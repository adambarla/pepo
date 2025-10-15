from huggingface_hub import login

# login() # Uncomment if you need to log in

import torch
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
OUTPUT_DIR = "./mc_dpo_smollm_ultrafeedback"

# Training parameters
NUM_TRAIN_EXAMPLES = 10 
NUM_EVAL_EXAMPLES = 2
EPOCHS = 1
LEARNING_RATE = 1e-5 
BETA = 0.1 
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4 
MAX_LENGTH = 512 
MAX_PROMPT_LENGTH = 256 

# --- MC Dropout Configuration ---
# This is the number of forward passes to create our "virtual ensemble"
L_VIRTUAL_ENSEMBLE = 5

# Device setup
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
print(f"Selected device: {DEVICE} with dtype: {DTYPE}")

# --- 1. Load Models and Tokenizer ---
print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE
)
policy_model.config.use_cache = False
policy_model.train()

# --- The lora_dropout is what enables MC Dropout ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05, # This dropout is key for the MC method
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
policy_model = get_peft_model(policy_model, peft_config)
policy_model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE
)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
print("Policy model and reference model loaded.")


# --- 2. Data Preparation ---
print(f"Loading dataset: {DATASET_ID}...")

dataset = load_dataset(path=DATASET_ID, split="train_sft")
train_dataset_raw = dataset.train_test_split(test_size=0.1, seed=42)['train']
eval_dataset_raw = dataset.train_test_split(test_size=0.1, seed=42)['test']

print(f"Loaded {len(train_dataset_raw)} training examples and {len(eval_dataset_raw)} evaluation examples.")

def preprocess_function(examples):
    processed = {
        "prompt_input_ids": [], "chosen_input_ids": [], "rejected_input_ids": [],
        "prompt_attention_mask": [], "chosen_attention_mask": [], "rejected_attention_mask": [],
        "prompt_len": []
    }

    for i in range(len(examples['prompt'])):
        current_prompt_messages = examples['prompt'][i]
        current_chosen_messages = examples['chosen'][i]
        current_rejected_messages = examples['rejected'][i]

        def ensure_message_list(messages, is_prompt=False, idx=i):
            if isinstance(messages, list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                return messages
            elif isinstance(messages, str):
                if is_prompt: return [{"role": "user", "content": messages}]
                else: return None
            else:
                return None

        current_prompt_messages = ensure_message_list(current_prompt_messages, is_prompt=True)
        current_chosen_messages = ensure_message_list(current_chosen_messages)
        current_rejected_messages = ensure_message_list(current_rejected_messages)

        if current_prompt_messages is None or current_chosen_messages is None or current_rejected_messages is None:
            continue

        prompt_with_assistant_turn = current_prompt_messages + [{"role": "assistant", "content": ""}]
        prompt_str = tokenizer.apply_chat_template(prompt_with_assistant_turn, tokenize=False, add_generation_prompt=True)
        chosen_str = tokenizer.apply_chat_template(current_chosen_messages, tokenize=False)
        rejected_str = tokenizer.apply_chat_template(current_rejected_messages, tokenize=False)

        prompt_encoded = tokenizer(prompt_str, truncation=True, max_length=MAX_PROMPT_LENGTH)
        chosen_encoded = tokenizer(chosen_str, truncation=True, max_length=MAX_LENGTH)
        rejected_encoded = tokenizer(rejected_str, truncation=True, max_length=MAX_LENGTH)

        if (len(prompt_encoded['input_ids']) >= MAX_PROMPT_LENGTH or
            len(chosen_encoded['input_ids']) >= MAX_LENGTH or
            len(rejected_encoded['input_ids']) >= MAX_LENGTH):
            continue

        processed["prompt_input_ids"].append(prompt_encoded["input_ids"])
        processed["chosen_input_ids"].append(chosen_encoded["input_ids"])
        processed["rejected_input_ids"].append(rejected_encoded["input_ids"])
        processed["prompt_attention_mask"].append(prompt_encoded["attention_mask"])
        processed["chosen_attention_mask"].append(chosen_encoded["attention_mask"])
        processed["rejected_attention_mask"].append(rejected_encoded["attention_mask"])
        processed["prompt_len"].append(len(prompt_encoded["input_ids"]))

    return processed

print("Preprocessing dataset...")
train_dataset = train_dataset_raw.map(preprocess_function, batched=True, remove_columns=train_dataset_raw.column_names, num_proc=os.cpu_count())
eval_dataset = eval_dataset_raw.map(preprocess_function, batched=True, remove_columns=eval_dataset_raw.column_names, num_proc=os.cpu_count())

train_dataset.set_format(type="torch")
eval_dataset.set_format(type="torch")
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
            padding='longest', max_length=self.max_prompt_length, return_tensors='pt',
        )['input_ids']
        batch['chosen_input_ids'] = self.tokenizer.pad(
            {'input_ids': batch['chosen_input_ids'], 'attention_mask': batch['chosen_attention_mask']},
            padding='longest', max_length=self.max_length, return_tensors='pt',
        )['input_ids']
        batch['rejected_input_ids'] = self.tokenizer.pad(
            {'input_ids': batch['rejected_input_ids'], 'attention_mask': batch['rejected_attention_mask']},
            padding='longest', max_length=self.max_length, return_tensors='pt',
        )['input_ids']

        # Also pad attention masks
        batch['prompt_attention_mask'] = self.tokenizer.pad(
            {'input_ids': [torch.ones(len(ids), dtype=torch.long) for ids in [f['prompt_input_ids'] for f in features]], 'attention_mask': batch['prompt_attention_mask']},
            padding='longest', max_length=self.max_prompt_length, return_tensors='pt',
        )['attention_mask']
        batch['chosen_attention_mask'] = self.tokenizer.pad(
            {'input_ids': [torch.ones(len(ids), dtype=torch.long) for ids in [f['chosen_input_ids'] for f in features]], 'attention_mask': batch['chosen_attention_mask']},
            padding='longest', max_length=self.max_length, return_tensors='pt',
        )['attention_mask']
        batch['rejected_attention_mask'] = self.tokenizer.pad(
            {'input_ids': [torch.ones(len(ids), dtype=torch.long) for ids in [f['rejected_input_ids'] for f in features]], 'attention_mask': batch['rejected_attention_mask']},
            padding='longest', max_length=self.max_length, return_tensors='pt',
        )['attention_mask']

        batch['prompt_len'] = torch.tensor(batch['prompt_len'], dtype=torch.long)
        return batch

data_collator = DPODataCollator(tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

# --- 3. Helper Function to Calculate Log Probabilities ---
def get_log_probs(model, input_ids, attention_mask, prompt_len):
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    sequence_lengths = attention_mask.sum(dim=-1)
    indices = torch.arange(token_log_probs.shape[1], device=token_log_probs.device).unsqueeze(0)
    response_mask = (indices >= (prompt_len - 1).unsqueeze(1)) & \
                    (indices < (sequence_lengths - 1).unsqueeze(1)) & \
                    (labels != tokenizer.pad_token_id)

    masked_log_probs = token_log_probs * response_mask.float()
    return masked_log_probs.sum(dim=-1)

# --- 4. Optimizer and Scheduler ---
optimizer = AdamW(policy_model.parameters(), lr=LEARNING_RATE)
num_training_steps = (len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# --- 5. Training Loop ---
print("Starting DPO training loop...")
global_step = 0
policy_model.zero_grad()

for epoch in range(EPOCHS):
    policy_model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")

    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        log_prob_chosen_policy = get_log_probs(policy_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['prompt_len'])
        with torch.no_grad():
            log_prob_chosen_ref = get_log_probs(ref_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['prompt_len'])

        log_prob_rejected_policy = get_log_probs(policy_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['prompt_len'])
        with torch.no_grad():
            log_prob_rejected_ref = get_log_probs(ref_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['prompt_len'])

        pi_log_ratio = log_prob_chosen_policy - log_prob_rejected_policy
        ref_log_ratio = log_prob_chosen_ref - log_prob_rejected_ref

        dpo_loss_components = -F.logsigmoid(BETA * (pi_log_ratio - ref_log_ratio))
        loss = dpo_loss_components.mean()
        
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
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

# --- Save the trained model adapter ---
HF_REPO_NAME = "ema1234/mc-dpo-smollm-ultrafeedback"
policy_model.push_to_hub(HF_REPO_NAME, use_auth_token=True)
tokenizer.push_to_hub(HF_REPO_NAME, use_auth_token=True)

print("DPO training complete!")

# ------------------------------------------------------------------------------------
# --- NEW SECTION: Pessimistic Generation using MC Dropout ---
# ------------------------------------------------------------------------------------
# if DEVICE == "cuda":
#     print("\n--- Testing the trained model with Pessimistic MC Dropout ---")

#     # 1. Load the base model and merge the trained LoRA adapter
#     print("Loading base model and merging adapter...")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         torch_dtype=DTYPE,
#         device_map=DEVICE
#     )
#     pessimistic_model = PeftModel.from_pretrained(base_model, final_model_path)
#     pessimistic_model = pessimistic_model.merge_and_unload()
#     print("Model merged.")
    
#     def generate_pessimistic_response(model, tokenizer, prompt, num_tokens=100):
#         """
#         Generates a response token by token using the pessimistic MC Dropout method.
#         """
#         model.eval() # Start in eval mode for setup
        
#         # Prepare the initial input
#         chat_prompt = [{"role": "user", "content": prompt}]
#         formatted_prompt = tokenizer.apply_chat_template(
#             chat_prompt,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(DEVICE)
        
#         print(f"\nGenerating pessimistic response for prompt:\n'{prompt}'")
        
#         # Generate token by token
#         for _ in tqdm(range(num_tokens), desc="Generating tokens"):
            
#             # --- The MC Dropout Core Logic ---
#             # 1. Set model to TRAIN mode to activate dropout layers
#             model.train() 
            
#             # 2. Get L different next-token predictions
#             next_token_logits = []
#             with torch.no_grad():
#                 for _ in range(L_VIRTUAL_ENSEMBLE):
#                     # Each forward pass will have a different dropout mask
#                     outputs = model(input_ids)
#                     next_token_logits.append(outputs.logits[:, -1, :]) # Get logits for the last token

#             # 3. Apply pessimistic rule (min of probabilities)
#             # Stack logits and apply softmax to get L probability distributions
#             all_probs = F.softmax(torch.stack(next_token_logits), dim=-1) # Shape: [L, 1, vocab_size]
            
#             # Find the minimum probability for each token in the vocab across all L runs
#             min_probs, _ = torch.min(all_probs, dim=0) # Shape: [1, vocab_size]
            
#             # 4. Sample from the pessimistic distribution
#             # `torch.multinomial` samples an index based on the weights (probabilities)
#             next_token = torch.multinomial(min_probs.squeeze(0), num_samples=1)
            
#             # 5. Append the new token and continue the loop
#             input_ids = torch.cat([input_ids, next_token], dim=-1)

#             # Check for EOS token
#             if next_token.item() == tokenizer.eos_token_id:
#                 break
                
#         # Switch back to eval mode for clean decoding
#         model.eval()
#         generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

#         return generated_text

#     # Generate a response
#     test_prompt = "Write a short, heartwarming story about an old cat."
#     full_response = generate_pessimistic_response(pessimistic_model, tokenizer, test_prompt)
    
#     print("\n--- Generated Pessimistic Response (Full) ---")
#     print(full_response)
    
# else:
#     print("\nSkipping MC Dropout generation: CUDA not available. This part is best run on GPU.")