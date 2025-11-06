import json
import time
import shortuuid
import tiktoken
import re
from tqdm import tqdm

# Import from the repo's utils
# Make sure this script is run from the root of the arena-hard-auto repo
from utils.completion import load_questions
from utils.add_markdown_info import count_markdown_elements, remove_pattern
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


def setup_logging(log_file=None):
    """
    Sets up logging to both file and console with timestamps.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"out/benchmark_run_{timestamp}.log"
    
    os.makedirs("out", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
        
        def write(self, message):
            if message.strip():
                self.level(message.strip())
        
        def flush(self):
            pass
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

log_file = setup_logging()

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


class PepoEnsemble:
    """
    Loads and manages the PEPO ensemble models.
    """
    def __init__(self, base_model_id, hf_username, num_networks, device, dtype, ensemble_name=None):
        self.base_model_id = base_model_id
        self.hf_username = hf_username
        self.num_networks = num_networks
        self.device = device
        self.dtype = dtype
        self.ensemble_name = ensemble_name
        self.ensemble_models = []
        self.tokenizer = None
        
        self._load_models()

    def _load_models(self):
        if self.ensemble_name:
            # Use custom ensemble name if provided
            output_dir = self.ensemble_name
        else:
            # Use default naming convention
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

    def sample_next_token(self, prompts, temperature, top_p, t_max=10, use_idk_token=False):
        """
        Performs the core PEPO sampling logic.
        Applies temperature and top_p sampling to the selected distribution.
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
            idk_token_id = -1
            if use_idk_token:
                extra_mass = 1.0 - torch.sum(min_probs, dim=-1, keepdim=True)
                extra_mass = torch.clamp(extra_mass, min=0.0) # ensure no negative probs
                distribution_to_sample = torch.cat([min_probs, extra_mass], dim=-1)
                idk_token_id = distribution_to_sample.shape[-1] - 1
            else:
                distribution_to_sample = min_probs
            
            # Handle temperature=0 case (greedy sampling)
            if temperature == 0.0 or temperature < 1e-6:
                # Greedy sampling: select the token with highest probability
                sampled_token_ids = torch.argmax(distribution_to_sample, dim=-1, keepdim=True)
            else:
                # Temperature-based sampling
                scaled_logits = torch.log(distribution_to_sample + 1e-9) / temperature
                temp_scaled_probs = torch.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(temp_scaled_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0 # always keep at least one token
                indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
                    dim=1, 
                    index=sorted_indices, 
                    src=sorted_indices_to_remove
                )
                filtered_probs = temp_scaled_probs.clone()
                filtered_probs[indices_to_remove] = 0.0
                normalized_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
                normalized_probs = torch.nan_to_num(normalized_probs, nan=0.0)
                sampled_token_ids = torch.multinomial(normalized_probs, num_samples=1).squeeze(-1)

                if use_idk_token and idk_token_id != -1:
                    needs_resample = (sampled_token_ids == idk_token_id)
                    for _ in range(t_max - 1):
                        if not torch.any(needs_resample):
                            break
                        probs_to_resample = normalized_probs[needs_resample]
                        
                        # guard against empty/zero probability rows if filtering was too aggressive
                        if probs_to_resample.shape[0] > 0 and torch.sum(probs_to_resample) > 0:
                            new_samples = torch.multinomial(probs_to_resample, num_samples=1).squeeze(-1)
                            sampled_token_ids[needs_resample] = new_samples
                            needs_resample = (sampled_token_ids == idk_token_id)
                        else:
                            break
                sampled_token_ids = sampled_token_ids.unsqueeze(-1)

        if isinstance(sampled_token_ids, torch.Tensor):
            token_list = sampled_token_ids.cpu().tolist()
        else:
            token_list = sampled_token_ids
        
        final_token_list = []
        if idk_token_id != -1:
            eos_token_id = self.tokenizer.eos_token_id
            assert eos_token_id is not None, "Tokenizer must have an EOS token ID defined."
                
            for token_id_list in token_list:
                if token_id_list[0] == idk_token_id:
                    logging.warning("Warning: 'I don't know' token ID survived sampling. Replacing with EOS token.")
                    final_token_list.append([eos_token_id])
                else:
                    final_token_list.append(token_id_list)
        else:
            final_token_list = token_list

        return self.tokenizer.batch_decode(final_token_list)

    def generate(self, initial_prompts, max_new_tokens, t_max, use_idk_token, temperature, top_p):
        """
        Generates full sequences for a batch of prompts using the PEPO ensemble.
        """
        batch_size = len(initial_prompts)
        
        current_prompts = initial_prompts.copy()
        finished = [False] * len(current_prompts)
        
        tokens_generated = 0
        for i in range(max_new_tokens):
            if all(finished):
                break   
            active_prompts = [p for p, f in zip(current_prompts, finished) if not f]
            active_indices = [i for i, f in enumerate(finished) if not f]
            if not active_prompts:
                break

            next_tokens = self.sample_next_token(
                active_prompts,
                temperature=temperature,
                top_p=top_p,
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
        
        return [
            current_prompts[i][len(initial_prompts[i]):] 
            for i in range(len(initial_prompts))
        ]


def main_generate_pepo_answers(args):
    # --- Setup Devices ---
    PEPO_DEVICE = f"cuda:{args.pepo_cuda_index}"
    DTYPE = torch.bfloat16
    logging.info(f"Using device: {PEPO_DEVICE}")

    # 1. Load PEPO Ensemble Model
    pepo_model = PepoEnsemble(
        base_model_id=args.model,
        hf_username=args.hf_username,
        num_networks=args.num_networks,
        device=PEPO_DEVICE,
        dtype=DTYPE,
        ensemble_name=args.ensemble_name
    )
    tokenizer = pepo_model.tokenizer
    
    # 2. Define output file
    # Make sure to use the bench_name from your config
    bench_name = "arena-hard-v2.0" 
    model_name = "pepo-ensemble" # This will be the file name
    answer_file = f"data/{bench_name}/model_answer/{model_name}.jsonl"
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    logging.info(f"Generating PEPO answers to: {answer_file}")

    # 3. Load Arena-Hard Questions
    question_file = f"data/{bench_name}/question.jsonl"
    questions = load_questions(question_file)
    
    # Setup for metadata (optional but good to have)
    encoding = tiktoken.encoding_for_model("gpt-4o")

    # 4. Generate and Save Answers
    total_questions = len(questions)
    with open(answer_file, "w", encoding="utf-8") as fout:
        for idx, question in enumerate(questions, 1):
            # Print progress
            print(f"\rGenerating answers: {idx}/{total_questions} ({100*idx/total_questions:.1f}%)", end='', flush=True)
            
            # 'question["prompt"]' already contains the formatted chat template
            # from arena-hard's 'question.jsonl'
            # Note: Your script's `arena_hard` loading logic was correct
            
            messages = [{"role": "user", "content": question["prompt"]}]
            
            # Use your script's generation logic for a single prompt
            pepo_response_text = pepo_model.generate(
                initial_prompts=[question["prompt"]], 
                max_new_tokens=args.max_new_tokens, 
                t_max=args.pepo_t_max, 
                use_idk_token=args.pepo_use_idk_token,
                temperature=args.temperature,
                top_p=args.top_p
            )[0] # Get the first (and only) response

            # *** CRITICAL: Format the output to match gen_answer.py ***
            
            # The 'content' for the assistant is a dictionary
            assistant_output = {"answer": pepo_response_text}
            
            messages.append({"role": "assistant", "content": assistant_output})

            # Calculate metadata (same as gen_answer.py)
            token_len = len(encoding.encode(pepo_response_text, disallowed_special=()))
            metadata = {
                "token_len": token_len
            } | count_markdown_elements(
                remove_pattern(
                    pepo_response_text, 
                    re.compile("```([^`]*)```")
                ),
                suffix="",
            )

            # Dump the final answer object
            ans = {
                "uid": question["uid"],
                "ans_id": shortuuid.uuid(),
                "model": model_name,
                "messages": messages,
                "tstamp": time.time(),
                "metadata": metadata
            }
            fout.write(json.dumps(ans, ensure_ascii=False) + "\n")
    
    print()  # New line after progress
    logging.info(f"PEPO answer generation complete. Generated {total_questions} answers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for PEPO Evaluation.")
    
    # --- PEPO Ensemble Args ---
    parser.add_argument("--num_networks", type=int, default=4,
                        help="Number of networks in the ensemble (L).")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B",
                        help="Base model ID")
    parser.add_argument("--hf_username", type=str, required=True,
                        help="HuggingFace username where ensemble models are stored")
    parser.add_argument("--ensemble_name", type=str, default=None,
                        help="Custom ensemble name (e.g., 'SmolLM2-1.7Bdpo_ensemble_with_1.0alpha4'). If not provided, uses default naming.")
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
    
    # --- Shared Sampling Args ---
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (used by both DPO and PEPO).")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling (used by both DPO and PEPO).")

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
    
    # Set a default max_new_tokens if not provided
    if args.max_new_tokens is None:
        args.max_new_tokens = 4096 

    # Run the new main function
    main_generate_pepo_answers(args)