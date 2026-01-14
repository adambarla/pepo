# python LLM/pepo_sampler.py --num_networks 4 --hf_username PessimisticDPO --model HuggingFaceTB/SmolLM2-1.7B

import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login

# Login to HuggingFace - only if not already logged in
# This will use the cached token if available, otherwise prompt
token = os.getenv("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
else:
    # This will use cached credentials if available
    try:
        login(add_to_git_credential=False)
    except Exception:
        # If no cached token, prompt for one
        login()

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

parser = argparse.ArgumentParser(description="Parser for loading ensemble models.")
parser.add_argument("--num_networks", type=int, default=3,
                    help="Number of networks in the ensemble (L).")
parser.add_argument("--cuda_index", type=int, default=0,
                    help="GPU Index")
parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B",
                    help="Base model ID")
parser.add_argument("--hf_username", type=str, required=True,
                    help="HuggingFace username where ensemble models are stored")
args = parser.parse_args()

# Configuration
BASE_MODEL_ID = args.model
L = args.num_networks  # Number of networks in the ensemble
name = args.model.rsplit('/', 1)[-1]  # Last substring
OUTPUT_DIR = f"{name}dpo_ensemble{L}"

# Device setup
if torch.cuda.is_available():
    DEVICE = f"cuda:{args.cuda_index}"
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
    print("Using MPS backend. Note: BFloat16 is not supported on MPS, using Float16.")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Selected device: {DEVICE} with dtype: {DTYPE}")
print(f"Base Model: {BASE_MODEL_ID}")
print(f"Loading {L} ensemble models...")

# Load tokenizer from the first model in the ensemble
hub_repo_id_0 = f"{args.hf_username}/{OUTPUT_DIR}_l0"
print(f"Loading tokenizer from: {hub_repo_id_0}")

tokenizer = AutoTokenizer.from_pretrained(hub_repo_id_0)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Tokenizer loaded successfully.")

ensemble_models = []

print(f"\n--- Loading {L} ensemble models into memory ---")

for l in tqdm(range(L), desc="Loading models"):
    # Construct the HuggingFace Hub repository ID for this model
    hub_repo_id = f"{args.hf_username}/{OUTPUT_DIR}_l{l}"
    
    print(f"\nLoading model {l} from: {hub_repo_id}")
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )
    
    # Load the LoRA adapter from HuggingFace Hub
    model = PeftModel.from_pretrained(base_model, hub_repo_id)
    
    # Merge the adapter weights into the base model
    model = model.merge_and_unload()
    
    # Set to evaluation mode
    model.eval()
    
    # Add to ensemble list
    ensemble_models.append(model)
    
    print(f"Model {l} loaded and merged successfully.")

print(f"\n✓ All {L} ensemble models loaded into memory!")
print(f"Ensemble models list contains {len(ensemble_models)} models.")


def sample_next_token(prompt, use_chat_template=True, t_max=10, use_idk_token=False):
    """
    Sample the next token using the ensemble of models with selective resampling.
    
    Args:
        prompt: A list of strings for batch processing.
        use_chat_template: Whether to apply chat template formatting.
        t_max: Maximum number of resampling attempts for "I don't know" tokens.
    
    Returns:
        A list of decoded next tokens, one for each prompt in the batch.
    """
    if use_chat_template:
        # Batch-process prompts by creating a list of message dicts
        messages_list = [[{"role": "user", "content": p}] for p in prompt]
        
        # apply_chat_template can't be batched, so we loop
        formatted_prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in messages_list
        ]
    else:
        formatted_prompts = prompt

    # tokenize the formatted prompts with padding
    tokenizer.padding_side = "left" # Use left padding for batch generation
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        probs_list = []
        for model in ensemble_models:
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            probs_list.append(probs)

        min_probs, _ = torch.min(torch.stack(probs_list), dim=0)
        if use_idk_token:
            extra_mass = 1.0 - torch.sum(min_probs, dim=-1, keepdim=True)
            adjusted_probs = torch.cat([min_probs, extra_mass], dim=-1)
            # The index for our "I don't know" token is the last one
            idk_token_id = adjusted_probs.shape[-1] - 1
            sampled_token_ids = torch.multinomial(adjusted_probs, num_samples=1)
            needs_resampling_mask = (sampled_token_ids == idk_token_id)
            
            # Print when IDK token is sampled initially
            if torch.any(needs_resampling_mask):
                idk_count = needs_resampling_mask.sum().item()
                idk_prob = extra_mass[needs_resampling_mask].mean().item()
                print(f"[IDK] Sampled IDK token for {idk_count}/{len(prompt)} prompt(s) (prob: {idk_prob:.4f})")
            
            # 3. Iteratively resample only the necessary elements
            resample_iteration = 0
            for _ in range(t_max - 1): # We already did 1 sample, so loop t_max-1 times
                # If no elements need resampling, break the loop early
                if not torch.any(needs_resampling_mask):
                    break
                
                resample_iteration += 1
                
                # Get the probabilities for only the elements that need resampling
                # .squeeze(-1) removes the last dimension to make the mask 1D for indexing
                probs_to_resample = adjusted_probs[needs_resampling_mask.squeeze(-1)]
                # generate new samples for these elements
                new_samples = torch.multinomial(probs_to_resample, num_samples=1)
                # update the original tensor with the new samples at the correct positions
                sampled_token_ids[needs_resampling_mask] = new_samples
                # update the mask for the next iteration
                needs_resampling_mask = (sampled_token_ids == idk_token_id)
                
                # Print resampling info
                if torch.any(needs_resampling_mask):
                    idk_count = needs_resampling_mask.sum().item()
                    print(f"[IDK] Resampling iteration {resample_iteration}: Still IDK for {idk_count} prompt(s)")
        else:
            # renormalize min_probs
            normalized_probs = min_probs / torch.sum(min_probs, dim=-1, keepdim=True)
            sampled_token_ids = torch.multinomial(normalized_probs, num_samples=1)


    return tokenizer.batch_decode(sampled_token_ids)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("--- Simple Iterative Generation Example ---")
    print("="*50)

    prompts = [
        "Switzerland is known for its",
        "The capital of France is",
        "In computer science, a binary tree is",
        "The theory of relativity was developed by",
    ]  
    max_new_tokens = 50

    print(f"Initial prompts ({len(prompts)} total):")
    for i, p in enumerate(prompts):
        print(f"  [{i}] '{p}'")
    print(f"\nGenerating up to {max_new_tokens} new tokens...\n")

    current_prompts = prompts.copy()  # Start with a copy of the initial prompts

    for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
        next_tokens = sample_next_token(current_prompts, use_chat_template=False, t_max=1, use_idk_token=True)
        
        # Check if any prompt hit EOS token
        if any(token == tokenizer.eos_token for token in next_tokens):
            print("\n[INFO] End-of-sequence token reached for at least one prompt.")
            break
        
        # Append each next token to its corresponding prompt
        current_prompts = [prompt + token for prompt, token in zip(current_prompts, next_tokens)]

    print("\n\n--- Final Generations ---")
    for i, text in enumerate(current_prompts):
        print(f"\n[{i}] {text}")