from huggingface_hub import login

login()

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import os
import math
import argparse

parser = argparse.ArgumentParser(description="Parser for DPO training parameters.")

parser.add_argument("--num_train_examples", type=int, default=10,
                    help="Number of training examples to use.")
parser.add_argument("--num_eval_examples", type=int, default=2,
                    help="Number of evaluation examples to use.")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-5,
                    help="Learning rate for the optimizer.")
parser.add_argument("--beta", type=float, default=0.1,
                    help="DPO beta parameter, controlling the strength of the preference.")
parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for training.")
parser.add_argument("--num_networks", type=int, default=3,
                    help="Number of networks in the ensemble.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Number of steps to accumulate gradients before updating.")
parser.add_argument("--max_length", type=int, default=1024,
                    help="Maximum total sequence length (prompt + response).")
parser.add_argument("--cuda_index", type=int, default=0,
                    help="GPU Index")
parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length.")
args=parser.parse_args()
# Training parameters
NUM_TRAIN_EXAMPLES = args.num_train_examples # Use a small subset for demonstration
NUM_EVAL_EXAMPLES = args.num_eval_examples
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate # DPO often uses a lower learning rate than SFT
BETA = args.beta # DPO beta parameter, controls the strength of the preference. Common values: 0.1, 0.5, 0.8
BATCH_SIZE = args.batch_size
L=args.num_networks #Number of networks in the ensemble
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = args.max_length # Max total sequence length (prompt + response)
MAX_PROMPT_LENGTH = args.max_prompt_length  # Max prompt length
# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M" # A small model for quick demonstration
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized" #"HuggingFaceH4/ultrafeedback_binarized"
OUTPUT_DIR = f"./dpo_ensemble{L}"


# Device setup

if torch.cuda.is_available():
    DEVICE = f"cuda:{args.cuda_index}"
    DTYPE = torch.bfloat16 # bfloat16 is usually best for NVIDIA GPUs (Ampere architecture and newer)
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16 # MPS typically supports float16 (half-precision), but not bfloat16.
                         # If float16 causes issues, fall back to torch.float32
    print("Using MPS backend. Note: BFloat16 is not supported on MPS, using Float16.")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32 # CPU runs best with float32

print(f"Selected device: {DEVICE} with dtype: {DTYPE}")
# --- 1. Load Models and Tokenizer ---
print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Crucial for padding and chat templates
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- FIX: Manually set the chat template ---
# Using the popular ChatML format.
chat_template = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)
tokenizer.chat_template = chat_template
print("Chat template has been set manually.")

dataset = load_dataset(path=DATASET_ID, split="train_sft")

# For demonstration, select a small subset
if NUM_TRAIN_EXAMPLES:
    dataset = dataset.shuffle(seed=42)
    train_dataset_raw = dataset.select(range(NUM_TRAIN_EXAMPLES))
    eval_dataset_raw = dataset.select(range(NUM_TRAIN_EXAMPLES, NUM_TRAIN_EXAMPLES + NUM_EVAL_EXAMPLES))
else:
    train_dataset_raw = dataset.train_test_split(test_size=0.1, seed=42)['train']
    eval_dataset_raw = dataset.train_test_split(test_size=0.1, seed=42)['test']

print(f"Loaded {len(train_dataset_raw)} training examples and {len(eval_dataset_raw)} evaluation examples.")



def preprocess_function(examples):
    processed = {
        "prompt_input_ids": [],
        "chosen_input_ids": [],
        "rejected_input_ids": [],
        "prompt_attention_mask": [],
        "chosen_attention_mask": [],
        "rejected_attention_mask": [],
        "prompt_len": []
    }

    for i in range(len(examples['prompt'])):
        current_prompt_messages = examples['prompt'][i]
        current_chosen_messages = examples['chosen'][i]
        current_rejected_messages = examples['rejected'][i]

        # --- START FIX FOR TYPEERROR ---
        # Robustness check: Ensure messages are lists of dictionaries.
        # ultrafeedback-binarized is supposed to have this format.
        # If it's a string, it's malformed data, or something has corrupted the dataset.
        
        # Helper to validate and potentially fix message lists
        def ensure_message_list(messages, is_prompt=False, idx=i):
            if isinstance(messages, list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                return messages
            elif isinstance(messages, str):
                # If it's a string, try to wrap it as a simple user message.
                # This is a heuristic for malformed data; assumes the string is the user's input.
                if is_prompt:
                    print(f"Warning: Prompt entry {idx} is a string. Wrapping as 'user' message.")
                    return [{"role": "user", "content": messages}]
                else: # Chosen/rejected responses should not be simple strings
                    print(f"Warning: Chosen/Rejected entry {idx} is a string, which is unexpected. Skipping example.")
                    return None
            else:
                # If it's not a list or a string, it's an unrecognized format.
                print(f"Warning: Malformed entry {idx} (type: {type(messages)}). Skipping example.")
                return None

        current_prompt_messages = ensure_message_list(current_prompt_messages, is_prompt=True)
        current_chosen_messages = ensure_message_list(current_chosen_messages)
        current_rejected_messages = ensure_message_list(current_rejected_messages)

        if current_prompt_messages is None or current_chosen_messages is None or current_rejected_messages is None:
            continue # Skip this example if any part is malformed
        # --- END FIX FOR TYPEERROR ---


        # Format prompt for DPO: add empty assistant turn
        prompt_with_assistant_turn = current_prompt_messages + [{"role": "assistant", "content": ""}]

        prompt_str = tokenizer.apply_chat_template(
            prompt_with_assistant_turn,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Format chosen and rejected responses (full conversation)
        chosen_str = tokenizer.apply_chat_template(
            current_chosen_messages,
            tokenize=False
        )
        rejected_str = tokenizer.apply_chat_template(
            current_rejected_messages,
            tokenize=False
        )

        # Tokenize (don't pad here, DataCollator will handle it)
        prompt_encoded = tokenizer(prompt_str, truncation=True, max_length=MAX_PROMPT_LENGTH)
        chosen_encoded = tokenizer(chosen_str, truncation=True, max_length=MAX_LENGTH)
        rejected_encoded = tokenizer(rejected_str, truncation=True, max_length=MAX_LENGTH)

        # Filter out examples that are too long after tokenization
        if (len(prompt_encoded['input_ids']) >= MAX_PROMPT_LENGTH or
            len(chosen_encoded['input_ids']) >= MAX_LENGTH or
            len(rejected_encoded['input_ids']) >= MAX_LENGTH):
            # print(f"Skipping example due to length: Prompt {len(prompt_encoded['input_ids'])}, Chosen {len(chosen_encoded['input_ids'])}, Rejected {len(rejected_encoded['input_ids'])}")
            continue

        processed["prompt_input_ids"].append(prompt_encoded["input_ids"])
        processed["chosen_input_ids"].append(chosen_encoded["input_ids"])
        processed["rejected_input_ids"].append(rejected_encoded["input_ids"])
        processed["prompt_attention_mask"].append(prompt_encoded["attention_mask"])
        processed["chosen_attention_mask"].append(chosen_encoded["attention_mask"])
        processed["rejected_attention_mask"].append(rejected_encoded["attention_mask"])
        processed["prompt_len"].append(len(prompt_encoded["input_ids"]))

    return processed


print("Preprocessing dataset (applying chat template and tokenizing)...")
train_dataset = train_dataset_raw.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset_raw.column_names,
    num_proc=os.cpu_count(),
    desc="Preprocessing train dataset"
)
eval_dataset = eval_dataset_raw.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset_raw.column_names,
    num_proc=os.cpu_count(),
    desc="Preprocessing eval dataset"
)

# Convert lists to tensors for DataLoader
train_dataset.set_format(type="torch", columns=['prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids',
                                                'prompt_attention_mask', 'chosen_attention_mask', 'rejected_attention_mask', 'prompt_len'])
eval_dataset.set_format(type="torch", columns=['prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids',
                                               'prompt_attention_mask', 'chosen_attention_mask', 'rejected_attention_mask', 'prompt_len'])


print(f"After preprocessing: {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples.")

# Custom Data Collator for DPO
class DPODataCollator:
    def __init__(self, tokenizer, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            batch[key] = [f[key] for f in features]

        # Pad sequences
        batch['prompt_input_ids'] = self.tokenizer.pad(
            {'input_ids': batch['prompt_input_ids'], 'attention_mask': batch['prompt_attention_mask']},
            padding='longest',
            max_length=self.max_prompt_length,
            return_tensors='pt',
        )['input_ids']
        batch['chosen_input_ids'] = self.tokenizer.pad(
            {'input_ids': batch['chosen_input_ids'], 'attention_mask': batch['chosen_attention_mask']},
            padding='longest',
            max_length=self.max_length,
            return_tensors='pt',
        )['input_ids']
        batch['rejected_input_ids'] = self.tokenizer.pad(
            {'input_ids': batch['rejected_input_ids'], 'attention_mask': batch['rejected_attention_mask']},
            padding='longest',
            max_length=self.max_length,
            return_tensors='pt',
        )['input_ids']

        # Also pad attention masks
        batch['prompt_attention_mask'] = self.tokenizer.pad(
            {'input_ids': [torch.ones(len(ids), dtype=torch.long) for ids in [f['prompt_input_ids'] for f in features]], 'attention_mask': batch['prompt_attention_mask']}, # Use actual lengths here
            padding='longest',
            max_length=self.max_prompt_length,
            return_tensors='pt',
        )['attention_mask']
        batch['chosen_attention_mask'] = self.tokenizer.pad(
            {'input_ids': [torch.ones(len(ids), dtype=torch.long) for ids in [f['chosen_input_ids'] for f in features]], 'attention_mask': batch['chosen_attention_mask']},
            padding='longest',
            max_length=self.max_length,
            return_tensors='pt',
        )['attention_mask']
        batch['rejected_attention_mask'] = self.tokenizer.pad(
            {'input_ids': [torch.ones(len(ids), dtype=torch.long) for ids in [f['rejected_input_ids'] for f in features]], 'attention_mask': batch['rejected_attention_mask']},
            padding='longest',
            max_length=self.max_length,
            return_tensors='pt',
        )['attention_mask']

        # Convert prompt_len to tensor
        batch['prompt_len'] = torch.tensor(batch['prompt_len'], dtype=torch.long)

        return batch

data_collator = DPODataCollator(tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

# --- 3. Helper Function to Calculate Log Probabilities ---
def get_log_probs(model, input_ids, attention_mask, prompt_len):
    """
    Calculates the log probability of a sequence of tokens given a model,
    masking out the prompt part and padding.

    Args:
        model: The language model (policy or reference).
        input_ids: Tensor of tokenized sequence (prompt + response).
        attention_mask: Tensor of attention mask for the sequence.
        prompt_len: Tensor of lengths of the prompt for each example in batch.

    Returns:
        A tensor of shape (batch_size,) containing the sum of log probabilities
        for the response tokens only.
    """
    with torch.no_grad() if model.training is False else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits # (batch_size, sequence_length, vocab_size)

    # Shift logits and labels for causal LM
    # The loss is computed for token_i given token_0 to token_{i-1}
    # So, logits[:, :-1, :] corresponds to predicting token_1 to token_{length-1}
    # And input_ids[:, 1:] are the actual token_1 to token_{length-1}
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    
    # Calculate log_softmax over the vocabulary dimension
    log_probs = F.log_softmax(logits, dim=-1) # (batch_size, sequence_length - 1, vocab_size)

    # Gather the log probabilities for the actual next tokens
    # `labels.unsqueeze(-1)` makes it (batch_size, sequence_length - 1, 1)
    # `log_probs.gather(dim=-1, index=...)` picks the log_prob at the label index
    # `squeeze(-1)` removes the last dimension, resulting in (batch_size, sequence_length - 1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Create a mask to only consider response tokens
    # tokens corresponding to prompt_len[i] up to the end of the sequence.
    # The mask needs to be shifted by 1 because `token_log_probs` is also shifted.
    sequence_lengths = attention_mask.sum(dim=-1) # Actual length of each sequence before padding
    
    # Create an index tensor for each position in the shifted sequence
    indices = torch.arange(token_log_probs.shape[1], device=token_log_probs.device).unsqueeze(0) # (1, seq_len-1)

    # Mask for response tokens: True if index >= prompt_len (shifted by 1) AND index < sequence_length (shifted by 1)
    response_mask = (indices >= (prompt_len - 1).unsqueeze(1)) & \
                    (indices < (sequence_lengths - 1).unsqueeze(1)) & \
                    (labels != tokenizer.pad_token_id) # Also exclude padding tokens explicitly

    # Apply the mask
    masked_log_probs = token_log_probs * response_mask.float()
    
    # Sum the log probabilities for each example
    return masked_log_probs.sum(dim=-1) # (batch_size,)

for l in range(L):

    # Policy Model (will be trained with LoRA)
    # Use prepare_model_for_kbit_training if you're using quantization (e.g., 4-bit)
    print(DTYPE)
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )
    policy_model.config.use_cache = False # Required for gradient checkpointing, often helpful for training
    policy_model.train() # Set to train mode for gradients

    # Apply LoRA to the policy model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear", # or specify: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    policy_model = get_peft_model(policy_model, peft_config)
    policy_model.print_trainable_parameters()

    # Reference Model (frozen copy of the initial SFT model)
    # Ensure this model is *not* trained and is in eval mode.
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )

    ref_model.eval() # Set to eval mode for no gradients and no dropout
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Policy model and reference model loaded.")

    # --- 2. Data Preparation ---
    print(f"Loading dataset: {DATASET_ID}...")
    # --- 4. Optimizer and Scheduler ---
    optimizer = AdamW(policy_model.parameters(), lr=LEARNING_RATE)

    num_training_steps = (len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # --- 5. Training Loop ---
    print("Starting DPO training loop...")
    global_step = 0
    policy_model.zero_grad()

    for epoch in range(EPOCHS):
        policy_model.train() # Ensure policy model is in train mode
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Compute log probabilities for chosen responses
            log_prob_chosen_policy = get_log_probs(policy_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['prompt_len'])
            with torch.no_grad(): # Ensure no gradients for reference model
                log_prob_chosen_ref = get_log_probs(ref_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['prompt_len'])

            # Compute log probabilities for rejected responses
            log_prob_rejected_policy = get_log_probs(policy_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['prompt_len'])
            with torch.no_grad():
                log_prob_rejected_ref = get_log_probs(ref_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['prompt_len'])

            # Calculate the DPO loss components
            pi_log_ratio = log_prob_chosen_policy - log_prob_rejected_policy
            ref_log_ratio = log_prob_chosen_ref - log_prob_rejected_ref

            dpo_loss_components = -F.logsigmoid(BETA * (pi_log_ratio - ref_log_ratio))
            
            # Average loss over the batch
            loss = dpo_loss_components.mean()
            
            # Backward pass with gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS # Scale back up for logging

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                policy_model.zero_grad()
                global_step += 1
                
                progress_bar.set_postfix({
                    "loss": total_loss / (step + 1),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "global_step": global_step
                })

        print(f"Epoch {epoch+1} finished. Average Training Loss: {total_loss / len(train_dataloader)}")

        # --- Evaluation ---
        policy_model.eval()
        eval_loss = 0
        eval_progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Evaluation")
        with torch.no_grad():
            for batch in eval_progress_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                log_prob_chosen_policy = get_log_probs(policy_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['prompt_len'])
                log_prob_chosen_ref = get_log_probs(ref_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['prompt_len'])

                log_prob_rejected_policy = get_log_probs(policy_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['prompt_len'])
                log_prob_rejected_ref = get_log_probs(ref_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['prompt_len'])

                pi_log_ratio = log_prob_chosen_policy - log_prob_rejected_policy
                ref_log_ratio = log_prob_chosen_ref - log_prob_rejected_ref

                dpo_loss_components = -F.logsigmoid(BETA * (pi_log_ratio - ref_log_ratio))
                eval_loss += dpo_loss_components.mean().item()
                eval_progress_bar.set_postfix({"eval_loss": eval_loss / (eval_progress_bar.n + 1)})
                
        print(f"Epoch {epoch+1} finished. Average Evaluation Loss: {eval_loss / len(eval_dataloader)}")

        # --- Save the fine-tuned model ---
        # Save the LoRA adapter
        final_model_path = os.path.join(OUTPUT_DIR, f"final_checkpoint_{l}")
        policy_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Model saved to {final_model_path}")

        print(f"DPO l = {l} training complete!")

        # --- Optional: Test the trained model ---
        if DEVICE == f"cuda:{args.cuda_index}":
            print("\n--- Testing the trained model ---")
            from transformers import pipeline
            from peft import PeftModel

            # Load the base model
            test_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )
            # Load the LoRA adapter and merge
            test_model = PeftModel.from_pretrained(test_model, final_model_path)
            test_model = test_model.merge_and_unload() # Merge LoRA weights into base model
            test_model.eval()

            pipe = pipeline(
                "text-generation",
                model=test_model,
                tokenizer=tokenizer,
                torch_dtype=DTYPE
            )

            test_prompt_message = [{"role": "user", "content": "Write a short, heartwarming story about an old cat."}]
            test_prompt = tokenizer.apply_chat_template(
                test_prompt_message,
                tokenize=False,
                add_generation_prompt=True
            )

            print(f"Generating response for prompt:\n{test_prompt}")
            
            outputs = pipe(
                test_prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
            print("\nGenerated Response (full):")
            print(outputs[0]['generated_text'])
            
            generated_text_only = outputs[0]['generated_text'].replace(test_prompt, '').strip()
            print("\nGenerated Response (clean):")
            print(generated_text_only)
        else:
            print("\nSkipping model testing: CUDA not available. Run on GPU for testing.")