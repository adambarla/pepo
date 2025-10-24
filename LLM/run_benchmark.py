import torch
import argparse
import os
import json
import random
import logging
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset

# ==============================================================================
# --- LOGGING SETUP ---
# ==============================================================================
def setup_logging(log_file=None):
    """
    Sets up logging to both file and console with timestamps.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"out/benchmark_run_{timestamp}.log"
    
    # Ensure out directory exists
    os.makedirs("out", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Also capture print statements
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
        
        def write(self, message):
            if message.strip():
                self.level(message.strip())
        
        def flush(self):
            pass
    
    # Uncomment to redirect print to logging (optional)
    # sys.stdout = LoggerWriter(logger.info)
    # sys.stderr = LoggerWriter(logger.error)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

# Initialize logging early (will be updated with proper filename in main)
log_file = setup_logging()

# --- HuggingFace Login ---
token = os.getenv("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
    logging.info("HuggingFace login successful (using HF_TOKEN environment variable).")
else:
    try:
        login(add_to_git_credential=False)
        logging.info("HuggingFace login successful (using cached credentials).")
    except Exception as e:
        logging.error("="*70)
        logging.error("ERROR: HuggingFace authentication failed!")
        logging.error("="*70)
        logging.error("\nFor non-interactive environments (SLURM), you must either:")
        logging.error("1. Set the HF_TOKEN environment variable:")
        logging.error("   export HF_TOKEN='your_token_here'")
        logging.error("   # Then run your command")
        logging.error("\n2. Or login once interactively on the compute node:")
        logging.error("   huggingface-cli login")
        logging.error("\nGet your token from: https://huggingface.co/settings/tokens")
        logging.error("="*70)
        exit(1)

# ==============================================================================
# --- PROMETHEUS JUDGE ---
# ==============================================================================
try:
    from vllm import LLM as VLLM_Engine
    from vllm import SamplingParams
except ImportError:
    logging.error("="*50)
    logging.error("ERROR: vllm not found.")
    logging.error("Please install with: pip install vllm")
    logging.error("="*50)
    exit(1)


class PrometheusJudge:
    """
    Wrapper for the open-source Prometheus judge model using vLLM directly.
    """
    def __init__(self, model_id, device, rubric_criteria, gpu_memory_utilization=0.9):
        logging.info(f"\n--- Loading Open-Source Judge: {model_id} on {device} ---")
        
        # Extract GPU ID from device string (e.g., "cuda:2" -> "2")
        if 'cuda' in device:
            gpu_id = device.split(':')[1] if ':' in device else '0'
            tensor_parallel_size = 1
        else:
            gpu_id = None
            tensor_parallel_size = 1
            
        vllm_kwargs = {
            'model': model_id,
            'trust_remote_code': True,
            'gpu_memory_utilization': gpu_memory_utilization,
            'dtype': 'auto',
            'tensor_parallel_size': tensor_parallel_size,
        }
        
        # Set CUDA_VISIBLE_DEVICES to use specific GPU
        if gpu_id is not None:
            logging.info(f"Setting judge to use GPU {gpu_id}")
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            
        self.model = VLLM_Engine(**vllm_kwargs)
        
        # Restore original CUDA_VISIBLE_DEVICES
        if gpu_id is not None:
            if original_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        self.rubric_criteria = rubric_criteria
        self.sampling_params = SamplingParams(
            temperature=0.01,
            max_tokens=1024,
            top_p=0.9
        )
        logging.info(f"✓ Open-source judge loaded with rubric: '{rubric_criteria}'")

    def _create_prometheus_prompt(self, instruction, response_a, response_b):
        """
        Creates a Prometheus-style prompt for relative grading.
        """
        prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criterion are given.
1. You will compare two responses (Response A and Response B) to the instruction and determine which one is better.
2. Your task is to rate the responses based on the given rubric and provide feedback.

###The instruction to evaluate:
{instruction}

###Response A to evaluate:
{response_a}

###Response B to evaluate:
{response_b}

###Score Rubric:
{self.rubric_criteria}

###Feedback:
Please compare Response A and Response B based on the rubric above.
First, provide detailed feedback explaining which response better follows the instruction.
Then, conclude with your judgment: 
- Output "A" if Response A is better
- Output "B" if Response B is better  
- Output "TIE" if both responses are equally good

Your judgment:"""
        return prompt

    def get_judgment(self, prompt, response_a, response_b):
        try:
            judge_prompt = self._create_prometheus_prompt(prompt, response_a, response_b)
            logging.info(f"      Sending prompt to judge (length: {len(judge_prompt)} chars)")
            outputs = self.model.generate([judge_prompt], self.sampling_params)
            
            if not outputs or len(outputs) == 0:
                logging.error("Error: No output from judge model")
                return "ERROR"
            
            judgment_text = outputs[0].outputs[0].text.strip().upper()
            logging.info(f"      Judge raw output (length: {len(judgment_text)} chars)")
            
            # Parse the judgment from the output
            if "A" in judgment_text[-100:]:  # Check last 100 chars for final judgment
                if "B" not in judgment_text[-100:] or judgment_text.rfind("A") > judgment_text.rfind("B"):
                    return "A"
            if "B" in judgment_text[-100:]:
                if "A" not in judgment_text[-100:] or judgment_text.rfind("B") > judgment_text.rfind("A"):
                    return "B"
            if "TIE" in judgment_text[-100:]:
                return "TIE"
            
            # Fallback: look anywhere in the text
            if "RESPONSE A IS BETTER" in judgment_text or "A IS BETTER" in judgment_text:
                return "A"
            elif "RESPONSE B IS BETTER" in judgment_text or "B IS BETTER" in judgment_text:
                return "B"
            elif "TIE" in judgment_text or "EQUAL" in judgment_text or "SAME" in judgment_text:
                return "TIE"
            
            logging.warning(f"Could not parse judgment from: {judgment_text[-200:]}")
            return "TIE"  # Default to tie if unclear
            
        except Exception as e:
            logging.error(f"Error calling judge: {e}")
            return "ERROR"

# ==============================================================================
# --- DEVICE & DTYPE SETUP ---
# ==============================================================================
if torch.cuda.is_available():
    DEFAULT_CUDA_INDEX = torch.cuda.current_device()
    DEVICE_NAME = f"cuda:{DEFAULT_CUDA_INDEX}"
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE_NAME = "mps"
    DTYPE = torch.float16
else:
    DEVICE_NAME = "cpu"
    DTYPE = torch.float32

# This will be configured when main() is called
# print(f"Default device: {DEVICE_NAME} with dtype: {DTYPE}")

# ==============================================================================
# --- PEPO ENSEMBLE LOADER CLASS ---
# ==============================================================================
class PepoEnsemble:
    """
    Loads and manages the PEPO ensemble models.
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
        
        hub_repo_id_0 = f"{self.hf_username}/{output_dir}_l0"
        logging.info(f"Loading tokenizer from: {hub_repo_id_0}")
        self.tokenizer = AutoTokenizer.from_pretrained(hub_repo_id_0)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logging.info("Tokenizer loaded successfully.")

        logging.info(f"\n--- Loading {self.num_networks} PEPO ensemble models ---")
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
        
        logging.info(f"\n✓ All {self.num_networks} PEPO models loaded to {self.device}!")

    def sample_next_token(self, prompts, t_max=10, use_idk_token=False):
        """
        Performs the core PEPO sampling logic.
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
            
            if use_idk_token:
                # Add an extra "I don't know" token as the last column of probabilities
                extra_mass = 1.0 - torch.sum(min_probs, dim=-1, keepdim=True)
                adjusted_probs = torch.cat([min_probs, extra_mass], dim=-1)
                idk_token_id = adjusted_probs.shape[-1] - 1

                # Draw initial samples (shape: [batch, 1]) and convert to 1D for easy masking
                sampled_token_ids = torch.multinomial(adjusted_probs, num_samples=1).squeeze(-1)

                # Mask where we sampled the IDK token
                needs_resample = (sampled_token_ids == idk_token_id)

                # Iteratively resample up to t_max-1 times for entries that picked IDK
                for _ in range(t_max - 1):
                    if not torch.any(needs_resample):
                        break
                    probs_to_resample = adjusted_probs[needs_resample]
                    # new_samples shape: [num_resample, 1] -> squeeze to [num_resample]
                    new_samples = torch.multinomial(probs_to_resample, num_samples=1).squeeze(-1)
                    # Assign back to the 1D sampled_token_ids
                    sampled_token_ids[needs_resample] = new_samples
                    needs_resample = (sampled_token_ids == idk_token_id)
                # For decoding, present token ids as list-of-lists [[id], [id], ...]
                sampled_token_ids = sampled_token_ids.unsqueeze(-1)
            else:
                normalized_probs = min_probs / torch.sum(min_probs, dim=-1, keepdim=True)
                sampled_token_ids = torch.multinomial(normalized_probs, num_samples=1)

        # Ensure we pass a list-of-lists to batch_decode
        if isinstance(sampled_token_ids, torch.Tensor):
            token_list = sampled_token_ids.cpu().tolist()
        else:
            token_list = sampled_token_ids
        return self.tokenizer.batch_decode(token_list)

    def generate(self, initial_prompts, max_new_tokens, t_max, use_idk_token):
        """
        Generates full sequences for a batch of prompts using the PEPO ensemble.
        """
        batch_size = len(initial_prompts)
        logging.info(f"      Starting PEPO generation for {batch_size} prompts (max_tokens={max_new_tokens})")
        
        current_prompts = initial_prompts.copy()
        finished = [False] * len(current_prompts)
        
        tokens_generated = 0
        for i in range(max_new_tokens):
            if all(finished):
                logging.info(f"      All sequences finished after {tokens_generated} tokens")
                break
            
            active_prompts = [p for p, f in zip(current_prompts, finished) if not f]
            active_indices = [i for i, f in enumerate(finished) if not f]

            if not active_prompts:
                break

            # Log every 50 tokens
            if i > 0 and i % 50 == 0:
                active_count = len(active_prompts)
                logging.info(f"      Generated {i}/{max_new_tokens} tokens, {active_count}/{batch_size} sequences still active")

            next_tokens = self.sample_next_token(
                active_prompts, 
                t_max=t_max, 
                use_idk_token=use_idk_token
            )
            
            token_idx = 0
            for idx in active_indices:
                token = next_tokens[token_idx]
                token_idx += 1
                if token == self.tokenizer.eos_token:
                    finished[idx] = True
                else:
                    current_prompts[idx] += token
            
            tokens_generated = i + 1
        
        logging.info(f"      PEPO generation complete: {tokens_generated} tokens, {sum(finished)}/{batch_size} finished")
        
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
    logging.info(f"\n--- Loading DPO baseline model: {model_id} ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()
    logging.info(f"✓ DPO baseline model loaded to {device}!")
    return model

def generate_baseline(prompts, model, tokenizer, max_new_tokens):
    """
    Generates full sequences from the baseline DPO model.
    """
    batch_size = len(prompts)
    logging.info(f"      Starting DPO generation for {batch_size} prompts (max_tokens={max_new_tokens})")
    
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    logging.info(f"      Input shape: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    logging.info(f"      DPO generation complete, decoding {outputs.shape[0]} sequences")
    
    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [
        decoded_texts[i][len(prompts[i]):] 
        for i in range(len(prompts))
    ]

# ==============================================================================
# --- MAIN EVALUATION SCRIPT ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Parser for PEPO Evaluation.")
    
    # --- PEPO Ensemble Args ---
    parser.add_argument("--num_networks", type=int, default=4,
                        help="Number of networks in the ensemble (L).")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B",
                        help="Base model ID")
    parser.add_argument("--hf_username", type=str, required=True,
                        help="HuggingFace username where ensemble models are stored")
    parser.add_argument("--pepo_cuda_index", type=int, default=0,
                        help="GPU index for PEPO ensemble models (default: 0)")
    parser.add_argument("--dpo_cuda_index", type=int, default=1,
                        help="GPU index for DPO baseline model (default: 1)")
    parser.add_argument("--judge_cuda_index", type=int, default=2,
                        help="GPU index for judge model (default: 2)")

    # --- PEPO Sampling Args ---
    parser.add_argument("--pepo_t_max", type=int, default=10,
                        help="Maximum resampling attempts for 'I don't know' token.")
    parser.add_argument("--pepo_use_idk_token", action="store_true",
                        help="Use the 'I don't know' token sampling method.")

    # --- Evaluation Args (MODIFIED) ---
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["tldr", "alpaca_eval", "arena_hard"],
                        help="Dataset to use for evaluation.")
    parser.add_argument("--dpo_baseline_model_id", type=str, required=True,
                        help="HuggingFace model ID for the DPO baseline comparison.")
    parser.add_argument("--judge_model_id", type=str, 
                        default="prometheus-eval/prometheus-7b-v2.0",
                        help="HuggingFace model ID for the open-source judge.")
    parser.add_argument("--judge_gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for judge model (0.0-1.0).")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of prompts to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max tokens to generate. Default varies by dataset.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save results. Default varies by dataset.")

    args = parser.parse_args()

    # Update log file name based on arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"out/benchmark_{args.evaluation_dataset}_L{args.num_networks}_{timestamp}.log"
    global log_file
    log_file = setup_logging(log_filename)
    
    logging.info("="*70)
    logging.info("PEPO Benchmark Evaluation Started")
    logging.info("="*70)
    logging.info(f"Evaluation Dataset: {args.evaluation_dataset}")
    logging.info(f"Number of Ensemble Networks: {args.num_networks}")
    logging.info(f"Base Model: {args.model}")
    logging.info(f"DPO Baseline: {args.dpo_baseline_model_id}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Number of Samples: {args.num_samples}")
    logging.info("="*70)

    # --- Setup Devices for Each Component ---
    if torch.cuda.is_available():
        PEPO_DEVICE = f"cuda:{args.pepo_cuda_index}"
        DPO_DEVICE = f"cuda:{args.dpo_cuda_index}"
        JUDGE_DEVICE = f"cuda:{args.judge_cuda_index}"
        logging.info(f"Multi-GPU Setup:")
        logging.info(f"  PEPO Ensemble: {PEPO_DEVICE}")
        logging.info(f"  DPO Baseline:  {DPO_DEVICE}")
        logging.info(f"  Judge Model:   {JUDGE_DEVICE}")
    else:
        PEPO_DEVICE = DPO_DEVICE = JUDGE_DEVICE = DEVICE_NAME
        logging.info(f"Using single device: {PEPO_DEVICE}")

    # 1. Load PEPO Ensemble Model (Needed for tokenizer)
    pepo_model = PepoEnsemble(
        base_model_id=args.model,
        hf_username=args.hf_username,
        num_networks=args.num_networks,
        device=PEPO_DEVICE,
        dtype=DTYPE
    )
    tokenizer = pepo_model.tokenizer
    
    # 2. Setup Dataset Configs
    logging.info(f"\n--- Configuring for dataset: {args.evaluation_dataset} ---")
    if args.evaluation_dataset == "tldr":
        dataset_id = "CarperAI/tldr_prompts"
        dataset_split = "test"
        prompt_column = "prompt"
        needs_chat_template = False
        rubric_criteria = "The summary should be coherent, accurate, and capture the main points of the post."
        if args.max_new_tokens is None:
            args.max_new_tokens = 100
            
    elif args.evaluation_dataset == "alpaca_eval":
        dataset_id = "tatsu-lab/alpaca_eval"
        dataset_split = "eval"
        prompt_column = "turns"
        needs_chat_template = False
        rubric_criteria = "The response should be helpful, relevant, and follow the user's instruction."
        if args.max_new_tokens is None:
            args.max_new_tokens = 512

    elif args.evaluation_dataset == "arena_hard":
        dataset_id = "lmarena-ai/arena-hard-auto-v0.1"
        dataset_split = "train" # Prompts are in the train split
        prompt_column = "turns"
        needs_chat_template = True # This is key
        rubric_criteria = "The response should be helpful, relevant, and follow the user's instruction in the conversation."
        if args.max_new_tokens is None:
            args.max_new_tokens = 512
            
    if args.output_file is None:
        args.output_file = f"out/evaluation_results_{args.evaluation_dataset}.jsonl"
        
    logging.info(f"Dataset: {dataset_id} ({dataset_split})")
    logging.info(f"Max New Tokens: {args.max_new_tokens}")
    logging.info(f"Output File: {args.output_file}")

    # 3. Load DPO Baseline Model
    dpo_model = load_baseline_model(args.dpo_baseline_model_id, DPO_DEVICE, DTYPE)

    # 4. Load Open-Source Judge
    judge = PrometheusJudge(
        args.judge_model_id, 
        JUDGE_DEVICE, 
        rubric_criteria,
        gpu_memory_utilization=args.judge_gpu_memory_utilization
    )

    # 5. Load and Process Dataset
    logging.info(f"\n--- Loading and Processing Dataset ---")
    dataset = load_dataset(dataset_id, split=dataset_split)
    dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    
    raw_prompts = [item[prompt_column] for item in dataset]
    prompts = []
    
    for p in tqdm(raw_prompts, desc="Formatting prompts"):
        if needs_chat_template:
            # Arena Hard prompts are lists of dicts, apply chat template
            if not tokenizer.chat_template:
                raise ValueError(
                    "Tokenizer is missing a chat_template, which is required for 'arena_hard'."
                    "Ensure you are using the tokenizer from the finetuned model."
                )
            prompts.append(
                tokenizer.apply_chat_template(
                    p, 
                    tokenize=False, 
                    add_generation_prompt=True # Crucial for multi-turn
                )
            )
        else:
            # TLDR and Alpaca Eval prompts are plain strings
            prompts.append(p)
    
    logging.info(f"Loaded and processed {len(prompts)} prompts for evaluation.")

    # 6. Run Generation and Evaluation
    logging.info(f"\n--- Starting Evaluation (Batch Size: {args.batch_size}) ---")
    pepo_wins = 0
    dpo_wins = 0
    ties = 0
    
    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    
    # Create a detailed generation log file
    generation_log_file = args.output_file.replace('.jsonl', '_generations.log')
    generation_logger = logging.getLogger('generations')
    generation_logger.setLevel(logging.INFO)
    gen_handler = logging.FileHandler(generation_log_file, mode='w')
    gen_handler.setFormatter(logging.Formatter('%(message)s'))
    generation_logger.addHandler(gen_handler)
    generation_logger.propagate = False  # Don't propagate to root logger
    
    logging.info(f"Detailed generations will be logged to: {generation_log_file}")
    
    with open(args.output_file, 'w') as f_out:
        for batch_idx, i in enumerate(tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating Batches")):
            batch_prompts = prompts[i:i+args.batch_size]
            
            import time
            batch_start_time = time.time()
            
            logging.info(f"\n[Batch {batch_idx+1}/{total_batches}] Processing samples {i+1}-{min(i+args.batch_size, len(prompts))}")
            logging.info(f"  Batch contains {len(batch_prompts)} prompts")
            logging.info(f"  Average prompt length: {sum(len(p) for p in batch_prompts) / len(batch_prompts):.0f} chars")
            
            logging.info(f"  Generating PEPO responses...")
            pepo_start = time.time()
            pepo_responses = pepo_model.generate(
                batch_prompts, 
                args.max_new_tokens, 
                args.pepo_t_max, 
                args.pepo_use_idk_token
            )
            pepo_time = time.time() - pepo_start
            logging.info(f"  PEPO generation took {pepo_time:.2f}s")
            
            # Log PEPO responses
            generation_logger.info(f"\n{'='*80}")
            generation_logger.info(f"BATCH {batch_idx+1} - PEPO RESPONSES")
            generation_logger.info(f"{'='*80}")
            for idx, (prompt, response) in enumerate(zip(batch_prompts, pepo_responses)):
                generation_logger.info(f"\n[Sample {i+idx+1}] PROMPT:")
                generation_logger.info(f"{prompt[:200]}..." if len(prompt) > 200 else prompt)
                generation_logger.info(f"\n[Sample {i+idx+1}] PEPO RESPONSE:")
                generation_logger.info(response)
                generation_logger.info("-"*80)
            
            logging.info(f"  Generating DPO baseline responses...")
            dpo_start = time.time()
            dpo_responses = generate_baseline(
                batch_prompts, 
                dpo_model, 
                tokenizer, 
                args.max_new_tokens
            )
            dpo_time = time.time() - dpo_start
            logging.info(f"  DPO generation took {dpo_time:.2f}s")
            
            # Log DPO responses
            generation_logger.info(f"\n{'='*80}")
            generation_logger.info(f"BATCH {batch_idx+1} - DPO RESPONSES")
            generation_logger.info(f"{'='*80}")
            for idx, (prompt, response) in enumerate(zip(batch_prompts, dpo_responses)):
                generation_logger.info(f"\n[Sample {i+idx+1}] PROMPT:")
                generation_logger.info(f"{prompt[:200]}..." if len(prompt) > 200 else prompt)
                generation_logger.info(f"\n[Sample {i+idx+1}] DPO RESPONSE:")
                generation_logger.info(response)
                generation_logger.info("-"*80)
            
            logging.info(f"  Running judge comparisons...")

            # Log judge evaluations header
            generation_logger.info(f"\n{'='*80}")
            generation_logger.info(f"BATCH {batch_idx+1} - JUDGE EVALUATIONS")
            generation_logger.info(f"{'='*80}")

            for j in range(len(batch_prompts)):
                prompt = batch_prompts[j]
                pepo_res = pepo_responses[j]
                dpo_res = dpo_responses[j]

                is_pepo_a = random.choice([True, False])
                response_a = pepo_res if is_pepo_a else dpo_res
                response_b = dpo_res if is_pepo_a else pepo_res

                sample_num = i + j + 1
                logging.info(f"    [Sample {sample_num}/{len(prompts)}] Calling judge (PEPO={'A' if is_pepo_a else 'B'})...")
                judgment = judge.get_judgment(prompt, response_a, response_b)
                logging.info(f"    [Sample {sample_num}/{len(prompts)}] Judge verdict: {judgment}")
                
                # Log judge evaluation details
                generation_logger.info(f"\n[Sample {sample_num}] JUDGE INPUT:")
                generation_logger.info(f"  PEPO position: {'A' if is_pepo_a else 'B'}")
                generation_logger.info(f"  Response A: {response_a[:100]}..." if len(response_a) > 100 else f"  Response A: {response_a}")
                generation_logger.info(f"  Response B: {response_b[:100]}..." if len(response_b) > 100 else f"  Response B: {response_b}")
                generation_logger.info(f"\n[Sample {sample_num}] PARSED VERDICT: {judgment}")
                generation_logger.info("-"*80)

                if (judgment == "A" and is_pepo_a) or (judgment == "B" and not is_pepo_a):
                    pepo_wins += 1
                elif (judgment == "B" and is_pepo_a) or (judgment == "A" and not is_pepo_a):
                    dpo_wins += 1
                elif judgment == "TIE":
                    ties += 1

                result_entry = {
                    "prompt": prompt,
                    "response_pepo": pepo_res,
                    "response_dpo": dpo_res,
                    "model_a": "pepo" if is_pepo_a else "dpo",
                    "model_b": "dpo" if is_pepo_a else "pepo",
                    "judgment": judgment
                }
                f_out.write(json.dumps(result_entry) + "\n")
            
            # Log progress after each batch
            f_out.flush()
            batch_total_time = time.time() - batch_start_time
            logging.info(f"  ✓ Batch {batch_idx+1} complete in {batch_total_time:.2f}s")
            logging.info(f"    Current scores - PEPO: {pepo_wins}, DPO: {dpo_wins}, Ties: {ties}")
            
            # Estimate remaining time
            avg_time_per_batch = batch_total_time  # Could keep running average
            remaining_batches = total_batches - (batch_idx + 1)
            est_remaining_time = avg_time_per_batch * remaining_batches
            logging.info(f"    Estimated time remaining: {est_remaining_time/60:.1f} minutes")

    # 7. Print Final Report
    logging.info("\n--- Evaluation Complete ---")
    logging.info(f"Results saved to: {args.output_file}")
    
    total_judged = pepo_wins + dpo_wins + ties
    if total_judged > 0:
        pepo_winrate = (pepo_wins + 0.5 * ties) / total_judged
        dpo_winrate = (dpo_wins + 0.5 * ties) / total_judged

        logging.info(f"\n--- Prometheus ({args.judge_model_id}) Judgment Results ---")
        logging.info(f"Evaluation Dataset:  {args.evaluation_dataset}")
        logging.info(f"Total Judged Samples: {total_judged}")
        logging.info(f"PEPO Wins:    {pepo_wins}")
        logging.info(f"DPO Baseline Wins: {dpo_wins}")
        logging.info(f"Ties:                {ties}")
        logging.info("-" * 30)
        logging.info(f"PEPO Win-Rate (vs DPO): {pepo_winrate:.2%}")
        logging.info(f"DPO Baseline Win-Rate (vs PEPO): {dpo_winrate:.2%}")
    else:
        logging.warning("\nNo samples were judged.")

if __name__ == "__main__":
    main()
