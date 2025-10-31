# python LLM/pepo_launcher.py --num_networks 1 --num_train_examples 0 -
# -num_eval_examples 0 --epochs 10 --batch_size 2 --cuda_index 0 --alpha 1.0
import torch
import os
import sys
import numpy as np
import random
from datetime import datetime
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
import math
import torch.multiprocessing as mp
from multiprocessing import set_start_method

# ============================================
# Set Random Seeds for Reproducibility
# ============================================
SEED = 42

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed at the very beginning
set_seed(SEED)
print(f"Random seed set to: {SEED}")

# ============================================
# Logging Utility: Redirect prints to file and console
# ============================================
class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', buffering=1)  # Line buffered
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

parser = argparse.ArgumentParser(description="Parser for DPO training parameters.")

parser.add_argument("--num_train_examples", type=int, default=20,
                    help="Number of training examples to use.")
parser.add_argument("--num_eval_examples", type=int, default=20,
                    help="Number of evaluation examples to use.")
parser.add_argument("--epochs", type=int, default=3,
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
                    help="GPU Index for policy model (trainable)")
parser.add_argument("--ref_cuda_index", type=int, default=None,
                    help="GPU Index for reference model (frozen). If None, uses same as cuda_index.")
parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length.")
parser.add_argument("--alpha", type=float, default=0.1,
                    help="Pessimistic margin alpha for the loss function.")
parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
parser.add_argument("--parallel", action="store_true",
                    help="Train models in parallel on different GPUs")
parser.add_argument("--gpu_ids", type=str, default=None,
                    help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). If not provided, will use all available GPUs.")
args=parser.parse_args()

# ============================================
# Initialize Logging
# ============================================
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = args.model.rsplit('/', 1)[-1]  # Extract model name
log_filename = f"logs/pepo_launcher_{model_name}_alpha{args.alpha}_L{args.num_networks}_{timestamp}.log"

# Initialize logger to redirect all prints to both console and file
logger = Logger(log_filename)
sys.stdout = logger
sys.stderr = logger

print("=" * 80)
print("PEPO Ensemble Training - Internal Log")
print("=" * 80)
print(f"Timestamp: {datetime.now()}")
print(f"Log file: {log_filename}")
print("=" * 80)
print()

# Training parameters
ALPHA = args.alpha
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
MODEL_ID = args.model# A small model for quick demonstration
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized" #"HuggingFaceH4/ultrafeedback_binarized"
name = last_substring = args.model.rsplit('/', 1)[-1] #Last substring
OUTPUT_DIR = f"PessimisticDPO/{name}dpo_ensemble_with_{ALPHA}alpha{L}"


# Print configuration summary
print("Training Configuration:")
print("-" * 80)
print(f"Model: {MODEL_ID}")
print(f"Dataset: {DATASET_ID}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Number of Networks (L): {L}")
print(f"Training Examples: {NUM_TRAIN_EXAMPLES if NUM_TRAIN_EXAMPLES > 0 else 'Full dataset'}")
print(f"Eval Examples: {NUM_EVAL_EXAMPLES if NUM_EVAL_EXAMPLES > 0 else 'Full dataset'}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Beta: {BETA}")
print(f"Alpha (Pessimistic Margin): {ALPHA}")
print(f"Max Length: {MAX_LENGTH}")
print(f"Max Prompt Length: {MAX_PROMPT_LENGTH}")
print("-" * 80)
print()

# Device setup
if torch.cuda.is_available():
    # Auto-detect available GPUs
    num_available_gpus = torch.cuda.device_count()
    
    # For sequential training, intelligently assign GPUs
    if not args.parallel:
        # Use cuda_index if specified, otherwise use GPU 0
        policy_gpu = args.cuda_index if args.cuda_index < num_available_gpus else 0
        
        # For reference model, try to use a different GPU if available
        if args.ref_cuda_index is not None:
            ref_gpu = args.ref_cuda_index if args.ref_cuda_index < num_available_gpus else policy_gpu
        elif num_available_gpus > 1:
            # If we have multiple GPUs, put reference model on a different one
            ref_gpu = (policy_gpu + 1) % num_available_gpus
        else:
            # Only one GPU available, share it
            ref_gpu = policy_gpu
        
        DEVICE = f"cuda:{policy_gpu}"
        REF_DEVICE = f"cuda:{ref_gpu}"
        print(f"Auto-detected {num_available_gpus} GPU(s)")
        print(f"Sequential training mode: Policy model on GPU {policy_gpu}, Reference model on GPU {ref_gpu}")
    else:
        # Parallel training mode - devices will be set per process
        DEVICE = f"cuda:{args.cuda_index}"
        REF_DEVICE = f"cuda:{args.ref_cuda_index}" if args.ref_cuda_index is not None else DEVICE
    
    DTYPE = torch.bfloat16 # bfloat16 is usually best for NVIDIA GPUs (Ampere architecture and newer)
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    REF_DEVICE = "mps"
    DTYPE = torch.float16 # MPS typically supports float16 (half-precision), but not bfloat16.
                         # If float16 causes issues, fall back to torch.float32
    print("Using MPS backend. Note: BFloat16 is not supported on MPS, using Float16.")
else:
    DEVICE = "cpu"
    REF_DEVICE = "cpu"
    DTYPE = torch.float32 # CPU runs best with float32

print(f"Selected device for policy model: {DEVICE} with dtype: {DTYPE}")
print(f"Selected device for reference model: {REF_DEVICE}")
# --- 1. Load Models and Tokenizer ---
print(f"Loading model: {MODEL_ID}...")
#if "HuggingFaceTB/SmolLM2-1.7B" in MODEL_ID and "Instruct" not in MODEL_ID:
#    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ID}-Instruct")
#else:
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, force_download=True)
# Crucial for padding and chat templates
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Manually set the chat template ---
# Using the popular ChatML format.
chat_template = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)
chat_template_smol = (
            "{% if messages[0]['role'] == 'system' %}"
                    "{{ messages[0]['content'] }}"
                        "{% else %}"
                                "{{ 'You are a helpful AI assistant.' }}"
                                    "{% endif %}"
                                        "{% for message in messages %}"
                                                "{% if message['role'] == 'user' %}"
                                                            "{{ '\n\n### User:\n' + message['content'] }}"
                                                                    "{% elif message['role'] == 'assistant' %}"
                                                                                "{{ '\n\n### Assistant:\n' + message['content'] }}"
                                                                                        "{% endif %}"
                                                                                            "{% endfor %}"
                                                                                            )
chat_template_gemma = (
        "{{ bos_token }}"
            "{% for message in messages %}"
                    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}"
                        "{% endfor %}"
                            "{% if add_generation_prompt %}"
                                    "{{ '<start_of_turn>model\n' }}"
                                        "{% endif %}"
                                        )
if MODEL_ID=="google/gemma-3-1b-pt": tokenizer.chat_template = chat_template_gemma
elif MODEL_ID=="HuggingFaceTB/SmolLM2-1.7B": tokenizer.chat_template = chat_template_smol
else: pass  #print("Chat template has been set manually.")

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

train_datasets_raw = {}
# Create an array of indices from 0 to len(dataset)-1
train_indices = np.arange(len(train_dataset_raw))
# Shuffle the indices for randomness
np.random.shuffle(train_indices)
# Split the indices into L roughly equal parts
train_split_indices = np.array_split(train_indices, L)

for i, indices in enumerate(train_split_indices):
    chunk_key = i + 1  # Keys from 1 to L
    # Select the examples corresponding to the current chunk of indices
    train_datasets_raw[chunk_key] = train_dataset_raw.select(indices)

# --- 2. Split the Evaluation Dataset ---
eval_datasets_raw = {}
eval_indices = np.arange(len(eval_dataset_raw))
# Shuffle the indices for randomness
np.random.shuffle(eval_indices)
eval_split_indices = np.array_split(eval_indices, L)

for i, indices in enumerate(eval_split_indices):
    chunk_key = i + 1 # Keys from 1 to L
    eval_datasets_raw[chunk_key] = eval_dataset_raw.select(indices)

def preprocess_function(examples):
    # examples['prompt'] is a list of lists of messages (dicts with 'role' and 'content')
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

        # Helper to validate and potentially fix message lists
        def ensure_message_list(messages, is_prompt=False, idx=i):
            if isinstance(messages, list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                return messages
            elif isinstance(messages, str):
                # If it's a string, try to wrap it as a simple user message.
                # This is a heuristic for malformed data; assumes the string is the user's input.
                if is_prompt:
                    # print(f"Warning: Prompt entry {idx} is a string. Wrapping as 'user' message.")
                    return [{"role": "user", "content": messages}]
                else: # Chosen/rejected responses should not be simple strings
                    # print(f"Warning: Chosen/Rejected entry {idx} is a string, which is unexpected. Skipping example.")
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
train_datasets = {}
eval_datasets = {}
print(train_datasets_raw)
for l,(train_dataset_raw, eval_dataset_raw) in enumerate(zip(train_datasets_raw.values(),eval_datasets_raw.values())):
    print(l, "l")
    print(train_dataset_raw, "train")
    print(eval_dataset_raw, "test")
    train_datasets[l] = train_dataset_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset_raw.column_names,
        num_proc=1,  # Disable multiprocessing to avoid subprocess crashes
        desc="Preprocessing train dataset"
    )
    eval_datasets[l] = eval_dataset_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset_raw.column_names,
        num_proc=1,  # Disable multiprocessing to avoid subprocess crashes
        desc="Preprocessing eval dataset"
    )

    # Convert lists to tensors for DataLoader
    train_datasets[l].set_format(type="torch", columns=['prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids',
                                                    'prompt_attention_mask', 'chosen_attention_mask', 'rejected_attention_mask', 'prompt_len'])
    eval_datasets[l].set_format(type="torch", columns=['prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids',
                                                'prompt_attention_mask', 'chosen_attention_mask', 'rejected_attention_mask', 'prompt_len'])

# Custom Data Collator for DPO (Optimized)
class DPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        # Helper function to pad a list of tensors efficiently
        def pad_tensors(tensors, pad_value=self.pad_token_id):
            max_len = max(len(t) for t in tensors)
            padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
            for i, t in enumerate(tensors):
                padded[i, :len(t)] = t
            return padded
        
        # Extract and pad all fields
        prompt_input_ids = pad_tensors([f['prompt_input_ids'] for f in features])
        chosen_input_ids = pad_tensors([f['chosen_input_ids'] for f in features])
        rejected_input_ids = pad_tensors([f['rejected_input_ids'] for f in features])
        
        prompt_attention_mask = pad_tensors([f['prompt_attention_mask'] for f in features], pad_value=0)
        chosen_attention_mask = pad_tensors([f['chosen_attention_mask'] for f in features], pad_value=0)
        rejected_attention_mask = pad_tensors([f['rejected_attention_mask'] for f in features], pad_value=0)
        
        prompt_len = torch.tensor([f['prompt_len'] for f in features], dtype=torch.long)

        return {
            'prompt_input_ids': prompt_input_ids,
            'chosen_input_ids': chosen_input_ids,
            'rejected_input_ids': rejected_input_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_attention_mask': rejected_attention_mask,
            'prompt_len': prompt_len
        }

data_collator = DPODataCollator(tokenizer)
train_dataloaders={}
eval_dataloaders={}
for l, (train_dataset,eval_dataset) in enumerate(zip(train_datasets.values(),eval_datasets.values())):
    train_dataloaders[l] = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    eval_dataloaders[l] = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator
    )

config = {
        'MODEL_ID': MODEL_ID,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'BETA': BETA,
        'BATCH_SIZE': BATCH_SIZE,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'ALPHA': ALPHA,
        'OUTPUT_DIR': OUTPUT_DIR,
        'DTYPE': DTYPE,
        'MAX_LENGTH': MAX_LENGTH,
        'MAX_PROMPT_LENGTH': MAX_PROMPT_LENGTH,
        'tokenizer': tokenizer,
        'data_collator': data_collator,
    }

# --- 3. Helper Function to Calculate Log Probabilities ---
def get_log_probs(model, input_ids, attention_mask, prompt_len, original_device=None):
    """
    Calculates the log probability of a sequence of tokens given a model,
    masking out the prompt part and padding.
    Handles cross-device operations (e.g., when reference model is on different GPU).

    Args:
        model: The language model (policy or reference).
        input_ids: Tensor of tokenized sequence (prompt + response).
        attention_mask: Tensor of attention mask for the sequence.
        prompt_len: Tensor of lengths of the prompt for each example in batch.
        original_device: Device to return results to (if None, uses input_ids device).

    Returns:
        A tensor of shape (batch_size,) containing the sum of log probabilities
        for the response tokens only.
    """
    # Store original device
    if original_device is None:
        original_device = input_ids.device
    
    # Get the device of the model
    model_device = next(model.parameters()).device
    
    # Move inputs to model's device if needed
    input_ids_on_model = input_ids.to(model_device)
    attention_mask_on_model = attention_mask.to(model_device)
    
    with torch.no_grad() if model.training is False else torch.enable_grad():
        outputs = model(input_ids=input_ids_on_model, attention_mask=attention_mask_on_model)
        logits = outputs.logits # (batch_size, sequence_length, vocab_size)

    # Shift logits and labels for causal LM
    logits = logits[:, :-1, :] # (batch_size, sequence_length - 1, vocab_size)
    labels = input_ids_on_model[:, 1:] # (batch_size, sequence_length - 1)

    # Calculate log_softmax over the vocabulary dimension
    log_probs = F.log_softmax(logits, dim=-1) # (batch_size, sequence_length - 1, vocab_size)

    # Gather the log probabilities for the actual next tokens
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, sequence_length - 1)

    # so far attention_mask_on_model corresponds to input_ids_on_model so it's 1 for prompt+response tokens and 0 for padding
    # compute sequence lengths (including prompt and response tokens and excluding padding)
    sequence_lengths = attention_mask_on_model.sum(dim=-1)
    
    # Create an index tensor for each position in the shifted sequence
    indices = torch.arange(token_log_probs.shape[1], device=model_device).unsqueeze(0) # (1, sequence_length - 1)

    # Move prompt_len to model device
    prompt_len_on_model = prompt_len.to(model_device)
    
    # Mask for response tokens
    response_mask = (indices >= (prompt_len_on_model - 1).unsqueeze(1)) & \
                    (indices < (sequence_lengths - 1).unsqueeze(1)) & \
                    (labels != tokenizer.pad_token_id) # defensive programming to avoid counting padding tokens

    # Apply the mask
    masked_log_probs = token_log_probs * response_mask.float()
    
    # Sum the log probabilities for each example and move back to original device
    result = masked_log_probs.sum(dim=-1)
    return result.to(original_device)

# --- Training Function for a Single Model ---
def train_single_model(l, gpu_id, train_dataset, eval_dataset, config):
    """
    Train a single model in the ensemble.
    
    Args:
        l: Model index (0 to L-1)
        gpu_id: GPU ID to use for this model
        train_dataset: Training dataset for this model
        eval_dataset: Evaluation dataset for this model
        config: Dictionary containing all training configuration
    """
    # Unpack configuration
    MODEL_ID = config['MODEL_ID']
    EPOCHS = config['EPOCHS']
    LEARNING_RATE = config['LEARNING_RATE']
    BETA = config['BETA']
    BATCH_SIZE = config['BATCH_SIZE']
    GRADIENT_ACCUMULATION_STEPS = config['GRADIENT_ACCUMULATION_STEPS']
    ALPHA = config['ALPHA']
    OUTPUT_DIR = config['OUTPUT_DIR']
    DTYPE = config['DTYPE']
    MAX_LENGTH = config['MAX_LENGTH']
    MAX_PROMPT_LENGTH = config['MAX_PROMPT_LENGTH']
    tokenizer = config['tokenizer']
    data_collator = config['data_collator']
    
    # Set device for this process
    DEVICE = f"cuda:{gpu_id}"
    REF_DEVICE = DEVICE # Using same GPU for reference model for simplicity; can be changed if needed
    
    # Setup logging for this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = MODEL_ID.rsplit('/', 1)[-1]
    log_filename = f"logs/pepo_model_{l}_gpu_{gpu_id}_{model_name}_{timestamp}.log"
    
    logger = Logger(log_filename)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    
    print("=" * 80)
    print(f"Training Model {l} on GPU {gpu_id}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    try:
        # Policy Model (will be trained with LoRA)
        print(f"Loading policy model on {DEVICE}...")
        policy_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE
        )
        policy_model.config.use_cache = False
        
        # Enable gradient checkpointing to save memory
        # policy_model.gradient_checkpointing_enable()
        # print("Gradient checkpointing enabled for policy model")
        
        policy_model.train()

        # Apply LoRA to the policy model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model.print_trainable_parameters()
        
        # Compile the policy model for faster training
        print("Compiling policy model with torch.compile()...")
        policy_model = torch.compile(policy_model, mode="reduce-overhead")
        print("Policy model compiled successfully")

        # Reference Model (frozen copy)
        print(f"Loading reference model on {REF_DEVICE}...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=REF_DEVICE
        )

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # Compile the reference model for faster inference
        print("Compiling reference model with torch.compile()...")
        ref_model = torch.compile(ref_model, mode="reduce-overhead")
        print("Reference model compiled successfully")
        
        print(f"Models loaded successfully on {DEVICE}")

        # Create dataloaders
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

        # Optimizer and Scheduler
        optimizer = AdamW(policy_model.parameters(), lr=LEARNING_RATE, fused=True) # we use fused optimizer for better performance on compatible GPUs

        num_batches = len(train_dataloader)
        if num_batches == 0:
            num_training_steps = 0
        else:
            num_training_steps = math.ceil(num_batches / GRADIENT_ACCUMULATION_STEPS) * EPOCHS
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # Training Loop
        print(f"Starting DPO training loop for model {l}...")
        global_step = 0
        policy_model.zero_grad() # Zero gradients before training

        for epoch in range(EPOCHS):
            policy_model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Model {l} - Epoch {epoch+1}/{EPOCHS}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                # Process chosen and rejected separately (they have different lengths)
                log_prob_chosen_policy = get_log_probs(
                    policy_model, 
                    batch['chosen_input_ids'], 
                    batch['chosen_attention_mask'], 
                    batch['prompt_len']
                )
                log_prob_rejected_policy = get_log_probs(
                    policy_model,
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    batch['prompt_len']
                )
                
                with torch.no_grad():
                    log_prob_chosen_ref = get_log_probs(
                        ref_model,
                        batch['chosen_input_ids'],
                        batch['chosen_attention_mask'],
                        batch['prompt_len']
                    )
                    log_prob_rejected_ref = get_log_probs(
                        ref_model,
                        batch['rejected_input_ids'],
                        batch['rejected_attention_mask'],
                        batch['prompt_len']
                    )

                # Calculate the DPO loss components
                pi_log_ratio = log_prob_chosen_policy - log_prob_rejected_policy
                ref_log_ratio = log_prob_chosen_ref - log_prob_rejected_ref
                alpha_offset = math.log(1.0 + ALPHA)
                argument = BETA * (pi_log_ratio - ref_log_ratio - alpha_offset)
                dpo_loss_components = -F.logsigmoid(argument)

                # Average loss over the batch
                loss = dpo_loss_components.mean()
                
                # Backward pass with gradient accumulation
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
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step
                    })

            avg_train_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
            print(f"Model {l} - Epoch {epoch+1} finished. Average Training Loss: {avg_train_loss}")
            
            # Evaluation
            policy_model.eval()
            eval_loss = 0
            eval_progress_bar = tqdm(eval_dataloader, desc=f"Model {l} - Epoch {epoch+1}/{EPOCHS} Eval")
            with torch.no_grad():
                for batch in eval_progress_bar:
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}

                    # Process chosen and rejected separately (they have different lengths)
                    log_prob_chosen_policy = get_log_probs(
                        policy_model,
                        batch['chosen_input_ids'],
                        batch['chosen_attention_mask'],
                        batch['prompt_len']
                    )
                    log_prob_rejected_policy = get_log_probs(
                        policy_model,
                        batch['rejected_input_ids'],
                        batch['rejected_attention_mask'],
                        batch['prompt_len']
                    )
                    log_prob_chosen_ref = get_log_probs(
                        ref_model,
                        batch['chosen_input_ids'],
                        batch['chosen_attention_mask'],
                        batch['prompt_len']
                    )
                    log_prob_rejected_ref = get_log_probs(
                        ref_model,
                        batch['rejected_input_ids'],
                        batch['rejected_attention_mask'],
                        batch['prompt_len']
                    )

                    pi_log_ratio = log_prob_chosen_policy - log_prob_rejected_policy
                    ref_log_ratio = log_prob_chosen_ref - log_prob_rejected_ref

                    alpha_offset = math.log(1.0 + ALPHA)
                    argument = BETA * (pi_log_ratio - ref_log_ratio - alpha_offset)
                    dpo_loss_components = -F.logsigmoid(argument)
                    eval_loss += dpo_loss_components.mean().item()
                    eval_progress_bar.set_postfix({"eval_loss": eval_loss / (eval_progress_bar.n + 1)})
                    
            avg_eval_loss = eval_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0.0
            print(f"Model {l} - Epoch {epoch+1} finished. Average Evaluation Loss: {avg_eval_loss}")

        # Save the fine-tuned model
        hub_repo_id = f"{OUTPUT_DIR}_l{l}"

        # Push the LoRA adapter model to the Hub
        policy_model.push_to_hub(
            hub_repo_id,
            commit_message=f"Upload LoRA adapter checkpoint {l}",
            private=True
        )

        # Push the tokenizer to the same repository
        tokenizer.push_to_hub(
            hub_repo_id,
            commit_message=f"Upload tokenizer for checkpoint {l}",
            private=True
        )

        print(f"Model {l} - LoRA adapter and tokenizer successfully pushed to: https://huggingface.co/{hub_repo_id}")
        print(f"Model {l} - DPO training complete!")
        
    except Exception as e:
        print(f"Error training model {l} on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.close()
        
        # Clean up GPU memory
        if 'policy_model' in locals():
            del policy_model
        if 'ref_model' in locals():
            del ref_model
        torch.cuda.empty_cache()

# Main execution logic
if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Determine available GPUs
    if args.parallel:
        if args.gpu_ids:
            # User explicitly specified GPU IDs
            assert False, "Explicit GPU IDs not supported in this version."
            available_gpus = [int(x.strip()) for x in args.gpu_ids.split(',')]
        else:
            # Auto-detect available GPUs
            # In SLURM, CUDA_VISIBLE_DEVICES is already set to renumber GPUs starting from 0
            num_gpus = torch.cuda.device_count()
            available_gpus = list(range(num_gpus))
            print(f"Auto-detected {num_gpus} GPUs from CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        print(f"Parallel training enabled with {len(available_gpus)} GPUs: {available_gpus}")
        print(f"Training {L} models across {len(available_gpus)} GPUs")
        

        # Create processes for parallel training
        processes = []
        for l in range(L):
            gpu_id = available_gpus[l % len(available_gpus)]  # Round-robin GPU assignment
            p = mp.Process(
                target=train_single_model,
                args=(l, gpu_id, train_datasets[l], eval_datasets[l], config)
            )
            p.start()
            processes.append(p)
            print(f"Started training process for model {l} on GPU {gpu_id}")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print("\n" + "=" * 80)
        print("All models trained successfully!")
        print("=" * 80)

    else:
        print("Warning: Sequential training mode not tested recently. Proceed with caution.")
        import sys; sys.exit(1)
        # --- Sequential Training ---
        print("Sequential training mode. Models will be trained one by one.")
        
        # Use the main GPU specified in args
        # The 'train_single_model' function will use this GPU ID to set its device
        main_gpu_id = args.cuda_index
        print(f"Using main GPU: {main_gpu_id} for all models.")

        for l in range(L):
            print("\n" + "=" * 80)
            print(f"Starting training for model {l} on GPU {main_gpu_id}")
            print("=" * 80)
            
            # Call the training function directly instead of duplicating logic.
            # This function handles loading, training, evaluation, and saving.
            train_single_model(
                l=l,
                gpu_id=main_gpu_id,
                train_dataset=train_datasets[l],
                eval_dataset=eval_datasets[l],
                config=config  # This 'config' dict must be defined outside the if/else block
                )
            
            print(f"Finished training and saving for model {l}.")

            # --- Optional: Test the trained model ---
            # This logic was in the original 'else' block and is preserved here.
            # It runs after each model is trained and saved.
            if DEVICE == f"cuda:{args.cuda_index}":
                # Imports needed for testing
                from transformers import pipeline, AutoModelForCausalLM
                from peft import PeftModel
                
                print("\n--- Testing the BASE model ---")

                # Load the original base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=DTYPE,
                    device_map=DEVICE
                )
                base_model.eval()

                # Create a pipeline for the base model
                base_pipe = pipeline(
                    "text-generation",
                    model=base_model,
                    tokenizer=tokenizer,
                    torch_dtype=DTYPE
                )

                # Prepare the prompt (it's the same for both models)
                test_prompt_message = [{"role": "user", "content": "Write a short, heartwarming story about an old cat."}]
                test_prompt = tokenizer.apply_chat_template(
                    test_prompt_message,
                    tokenize=False,
                    add_generation_prompt=True
                )

                print(f"Generating response for prompt with BASE model:\n{test_prompt}")

                base_outputs = base_pipe(
                    test_prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.3,
                    top_k=50,
                    top_p=0.995,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id
                )
                print("\nGenerated Response from BASE model (full):")
                print(base_outputs[0]['generated_text'])

                base_generated_text_only = base_outputs[0]['generated_text'].replace(test_prompt, '').strip()
                print("\nGenerated Response from BASE model (clean):")
                print(base_generated_text_only)

                # Clean up memory
                del base_model
                del base_pipe
                torch.cuda.empty_cache()

                # =================================================================
                #  SECTION 2: Testing the TRAINED (fine-tuned) model
                # =================================================================
                print("\n\n--- Testing the TRAINED model ---")
                
                # This is the Hub ID where train_single_model saved the adapter
                hub_repo_id = f"{OUTPUT_DIR}_l{l}" 

                # Load the base model again to apply the adapter
                trained_model_base = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=DTYPE,
                    device_map=DEVICE
                )
                # Load the LoRA adapter and merge
                trained_model = PeftModel.from_pretrained(trained_model_base, hub_repo_id)
                trained_model = trained_model.merge_and_unload() # Merge LoRA weights into base model
                trained_model.eval()

                pipe = pipeline(
                    "text-generation",
                    model=trained_model,
                    tokenizer=tokenizer,
                    torch_dtype=DTYPE
                )

                print(f"Generating response for prompt with TRAINED model:\n{test_prompt}")

                outputs = pipe(
                    test_prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.3,
                    top_k=50,
                    top_p=0.995,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id
                )
                print("\nGenerated Response from TRAINED model (full):")
                print(outputs[0]['generated_text'])

                generated_text_only = outputs[0]['generated_text'].replace(test_prompt, '').strip()
                print("\nGenerated Response from TRAINED model (clean):")
                print(generated_text_only)
                
                # Clean up
                del trained_model_base
                del trained_model
                del pipe
                torch.cuda.empty_cache()
                
            else:
                print("\nSkipping model testing: Not on main CUDA device. Run on GPU for testing.")

        print("\n" + "=" * 80)
        print("All models trained and tested successfully!")
        print("=" * 80)