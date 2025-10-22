import torch
import argparse
import os
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset
from openai import OpenAI

# --- HuggingFace Login ---
# (Using your provided login logic)
token = os.getenv("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
else:
    try:
        login(add_to_git_credential=False)
    except Exception:
        login()
print("HuggingFace login successful.")

# ==============================================================================
# --- DEVICE & DTYPE SETUP ---
# ==============================================================================
if torch.cuda.is_available():
    # Detect default cuda index, but allow override
    DEFAULT_CUDA_INDEX = torch.cuda.current_device()
    DEVICE_NAME = f"cuda:{DEFAULT_CUDA_INDEX}"
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE_NAME = "mps"
    DTYPE = torch.float16
    print("Using MPS backend. Note: BFloat16 is not supported on MPS, using Float16.")
else:
    DEVICE_NAME = "cpu"
    DTYPE = torch.float32

print(f"Default device: {DEVICE_NAME} with dtype: {DTYPE}")


# ==============================================================================
# --- PEPO ENSEMBLE LOADER CLASS ---
# ==============================================================================

class PepoEnsemble:
    """
    Loads and manages the PEPO/χPO ensemble models.
    """
    def __init__(self, base_model_id, hf_username, num_networks, device, dtype):
        self.base_model_id = base_model_id
        self.hf_username = hf_username
        self.num_networks = num_networks
        self.device = device
        self.dtype = dtype
        self.ensemble_models = []
        self.tokenizer = None
        
        self._load_models()

    def _load_models(self):
        name = self.base_model_id.rsplit('/', 1)[-1]
        output_dir = f"{name}dpo_ensemble{self.num_networks}"
        
        # 1. Load tokenizer from the first model
        hub_repo_id_0 = f"{self.hf_username}/{output_dir}_l0"
        print(f"Loading tokenizer from: {hub_repo_id_0}")
        self.tokenizer = AutoTokenizer.from_pretrained(hub_repo_id_0)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Use left padding for batch generation
        print("Tokenizer loaded successfully.")

        # 2. Load all ensemble models
        print(f"\n--- Loading {self.num_networks} PEPO ensemble models ---")
        for l in tqdm(range(self.num_networks), desc="Loading PEPO models"):
            hub_repo_id = f"{self.hf_username}/{output_dir}_l{l}"
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=self.dtype,
                device_map=self.device
            )
            model = PeftModel.from_pretrained(base_model, hub_repo_id)
            model = model.merge_and_unload()
            model.eval()
            self.ensemble_models.append(model)
        
        print(f"\n✓ All {self.num_networks} PEPO models loaded to {self.device}!")

    def sample_next_token(self, prompts, t_max=10, use_idx_token=False):
        """
        Performs the core PEPO/χPO sampling logic for a batch of prompts.
        (This is your 'sample_next_token' function, integrated as a method)
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            probs_list = []
            for model in self.ensemble_models:
                outputs = model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                probs_list.append(probs)

            min_probs, _ = torch.min(torch.stack(probs_list), dim=0)
            
            if use_idx_token:
                extra_mass = 1.0 - torch.sum(min_probs, dim=-1, keepdim=True)
                adjusted_probs = torch.cat([min_probs, extra_mass], dim=-1)
                idk_token_id = adjusted_probs.shape[-1] - 1
                sampled_token_ids = torch.multinomial(adjusted_probs, num_samples=1)
                needs_resampling_mask = (sampled_token_ids == idk_token_id)
                
                for _ in range(t_max - 1):
                    if not torch.any(needs_resampling_mask):
                        break
                    probs_to_resample = adjusted_probs[needs_resampling_mask.squeeze(-1)]
                    new_samples = torch.multinomial(probs_to_resample, num_samples=1)
                    sampled_token_ids[needs_resampling_mask] = new_samples
                    needs_resampling_mask = (sampled_token_ids == idk_token_id)
            else:
                normalized_probs = min_probs / torch.sum(min_probs, dim=-1, keepdim=True)
                sampled_token_ids = torch.multinomial(normalized_probs, num_samples=1)

        return self.tokenizer.batch_decode(sampled_token_ids)

    def generate(self, initial_prompts, max_new_tokens, t_max, use_idx_token):
        """
        Generates full sequences for a batch of prompts using the PEPO ensemble.
        """
        current_prompts = initial_prompts.copy()
        finished = [False] * len(current_prompts)
        
        for _ in range(max_new_tokens):
            if all(finished):
                break
            
            active_prompts = [p for p, f in zip(current_prompts, finished) if not f]
            active_indices = [i for i, f in enumerate(finished) if not f]

            if not active_prompts:
                break

            next_tokens = self.sample_next_token(
                active_prompts, 
                t_max=t_max, 
                use_idx_token=use_idx_token
            )
            
            token_idx = 0
            for i in active_indices:
                token = next_tokens[token_idx]
                token_idx += 1
                if token == self.tokenizer.eos_token:
                    finished[i] = True
                else:
                    current_prompts[i] += token
        
        # Return only the newly generated part
        return [
            current_prompts[i][len(initial_prompts[i]):] 
            for i in range(len(initial_prompts))
        ]


# ==============================================================================
# --- BASELINE MODEL GENERATION ---
# ==============================================================================

def load_baseline_model(model_id, device, dtype):
    """
    Loads the DPO baseline model for comparison.
    """
    print(f"\n--- Loading DPO baseline model: {model_id} ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()
    print(f"✓ DPO baseline model loaded to {device}!")
    return model

def generate_baseline(prompts, model, tokenizer, max_new_tokens):
    """
    Generates full sequences from the baseline DPO model.
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Use sampling for a fair comparison
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode and strip the prompt
    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [
        decoded_texts[i][len(prompts[i]):] 
        for i in range(len(prompts))
    ]

# ==============================================================================
# --- GPT-4o JUDGING ---
# ==============================================================================

def get_gpt4o_judge(client, prompt, response_a, response_b):
    """
    Asks GPT-4o to judge which response is better.
    """
    judge_system_prompt = (
        "You are an expert evaluator for large language models. "
        "You will be given a prompt and two model responses, 'Response A' and 'Response B'. "
        "Your task is to determine which response is a better summary of the provided post. "
        "A good summary is coherent, accurate, and captures the main points of the post. "
        "Reply with only a single letter: 'A' if Response A is better, 'B' if Response B is better, or 'T' if they are tied or equally good."
    )
    
    judge_user_prompt = f"""
**PROMPT (Post):**
{prompt}

**Response A:**
{response_a}

**Response B:**
{response_b}

**Your judgment (A, B, or T):**
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": judge_user_prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        choice = response.choices[0].message.content.strip().upper()
        
        if "A" in choice:
            return "A"
        elif "B" in choice:
            return "B"
        else:
            return "TIE"
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "ERROR"

# ==============================================================================
# --- MAIN EVALUATION SCRIPT ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parser for PEPO/χPO Evaluation.")
    
    # --- PEPO Ensemble Args (from your script) ---
    parser.add_argument("--num_networks", type=int, default=4,
                        help="Number of networks in the ensemble (L).")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B",
                        help="Base model ID")
    parser.add_argument("--hf_username", type=str, required=True,
                        help="HuggingFace username where ensemble models are stored")
    parser.add_argument("--cuda_index", type=int, default=DEFAULT_CUDA_INDEX,
                        help=f"GPU Index (default: {DEFAULT_CUDA_INDEX})")

    # --- PEPO Sampling Args ---
    parser.add_argument("--pepo_t_max", type=int, default=10,
                        help="Maximum resampling attempts for 'I don't know' token.")
    parser.add_argument("--pepo_use_idx_token", action="store_true",
                        help="Use the 'I don't know' token sampling method.")

    # --- Evaluation Args (NEW) ---
    parser.add_argument("--dpo_baseline_model_id", type=str, required=True,
                        help="HuggingFace model ID for the DPO baseline comparison.")
    parser.add_argument("--dataset_name", type=str, default="CarperAI/tldr_prompts",
                        help="Dataset to use for evaluation.")
    parser.add_argument("--dataset_split", type=str, default="test",
                        help="Dataset split to use (e.g., 'test', 'train').")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of prompts to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Max tokens to generate for each summary.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl",
                        help="File to save generation results and judgments.")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key for GPT-4o judging. If not provided, script will only generate.")

    args = parser.parse_args()

    # --- Setup Device ---
    if torch.cuda.is_available():
        DEVICE = f"cuda:{args.cuda_index}"
    else:
        DEVICE = DEVICE_NAME # Use auto-detected MPS or CPU
    print(f"Using evaluation device: {DEVICE}")

    # 1. Load PEPO/χPO Ensemble Model
    pepo_model = PepoEnsemble(
        base_model_id=args.model,
        hf_username=args.hf_username,
        num_networks=args.num_networks,
        device=DEVICE,
        dtype=DTYPE
    )
    # Use the tokenizer from the PEPO model for both
    tokenizer = pepo_model.tokenizer

    # 2. Load DPO Baseline Model
    dpo_model = load_baseline_model(args.dpo_baseline_model_id, DEVICE, DTYPE)

    # 3. Load Dataset
    print(f"\n--- Loading Dataset: {args.dataset_name} ({args.dataset_split}) ---")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    prompts = [item['prompt'] for item in dataset]
    print(f"Loaded {len(prompts)} prompts for evaluation.")

    # 4. Setup GPT-4o Judger
    openai_client = None
    if args.openai_api_key:
        openai_client = OpenAI(api_key=args.openai_api_key)
        print("OpenAI client initialized for GPT-4o judging.")
    else:
        print("[Warning] No OpenAI API key provided. Script will only generate responses, not judge them.")

    # 5. Run Generation and Evaluation
    print(f"\n--- Starting Evaluation (Batch Size: {args.batch_size}) ---")
    pepo_wins = 0
    dpo_wins = 0
    ties = 0
    
    with open(args.output_file, 'w') as f_out:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating Batches"):
            batch_prompts = prompts[i:i+args.batch_size]
            
            # Generate from PEPO Ensemble
            pepo_responses = pepo_model.generate(
                batch_prompts, 
                args.max_new_tokens, 
                args.pepo_t_max, 
                args.pepo_use_idx_token
            )
            
            # Generate from DPO Baseline
            dpo_responses = generate_baseline(
                batch_prompts, 
                dpo_model, 
                tokenizer, 
                args.max_new_tokens
            )

            # Judge and save results for each item in the batch
            for j in range(len(batch_prompts)):
                prompt = batch_prompts[j]
                pepo_res = pepo_responses[j]
                dpo_res = dpo_responses[j]

                # Randomize A/B assignment to prevent bias
                is_pepo_a = random.choice([True, False])
                response_a = pepo_res if is_pepo_a else dpo_res
                response_b = dpo_res if is_pepo_a else pepo_res

                judgment = "NOT_JUDGED"
                if openai_client:
                    judgment = get_gpt4o_judge(openai_client, prompt, response_a, response_b)

                # Tally scores
                if (judgment == "A" and is_pepo_a) or (judgment == "B" and not is_pepo_a):
                    pepo_wins += 1
                elif (judgment == "B" and is_pepo_a) or (judgment == "A" and not is_pepo_a):
                    dpo_wins += 1
                elif judgment == "TIE":
                    ties += 1

                # Save result to file
                result_entry = {
                    "prompt": prompt,
                    "response_pepo": pepo_res,
                    "response_dpo": dpo_res,
                    "model_a": "pepo" if is_pepo_a else "dpo",
                    "model_b": "dpo" if is_pepo_a else "pepo",
                    "judgment": judgment
                }
                f_out.write(json.dumps(result_entry) + "\n")

    # 6. Print Final Report
    print("\n--- Evaluation Complete ---")
    print(f"Results saved to: {args.output_file}")
    
    total_judged = pepo_wins + dpo_wins + ties
    if total_judged > 0:
        pepo_winrate = (pepo_wins + 0.5 * ties) / total_judged
        dpo_winrate = (dpo_wins + 0.5 * ties) / total_judged

        print("\n--- GPT-4o Judgment Results ---")
        print(f"Total Judged Samples: {total_judged}")
        print(f"PEPO (χPO) Wins:    {pepo_wins}")
        print(f"DPO Baseline Wins: {dpo_wins}")
        print(f"Ties:                {ties}")
        print("-" * 30)
        print(f"PEPO (χPO) Win-Rate (vs DPO): {pepo_winrate:.2%}")
        print(f"DPO Baseline Win-Rate (vs PEPO): {dpo_winrate:.2%}")
    else:
        print("\nNo samples were judged by GPT-4o.")


if __name__ == "__main__":
    main()