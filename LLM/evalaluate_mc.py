import torch
import os
import random
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Model Paths ---
# Ensure these paths point to the outputs of your training scripts.
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PESSIMISTIC_MODEL_PATH = "./mc_dpo_smollm_ultrafeedback/final_checkpoint" # Model A
DPO_MODEL_PATH = "./dpo_custom_tinyllama_ultrafeedback/final_checkpoint" # Model B

# --- Judge Model Configuration ---
# Using a free, open-source model from Hugging Face.
# Requires significant GPU memory (e.g., >16GB VRAM).
JUDGE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Device Configuration ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Base Model: {BASE_MODEL_ID}")
print(f"Model A (Pessimistic): {PESSIMISTIC_MODEL_PATH}")
print(f"Model B (Standard DPO): {DPO_MODEL_PATH}")
print(f"Judge Model: {JUDGE_MODEL_NAME}")
print(f"Using device: {DEVICE}")

# ==============================================================================
# --- 2. HELPER FUNCTION TO LOAD MODELS ---
# ==============================================================================

def load_and_merge_lora_model(base_model_id, adapter_path, device, dtype):
    """Loads a base model, applies a LoRA adapter, and merges the weights."""
    print(f"Loading and merging model from: {adapter_path}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    print("Model loaded and merged successfully.")
    return model

# ==============================================================================
# --- 3. LOAD MODELS AND TOKENIZERS ---
# ==============================================================================

print("\n--- Loading models for comparison ---")

# Tokenizer for the models being tested
tokenizer = AutoTokenizer.from_pretrained(PESSIMISTIC_MODEL_PATH)
tokenizer.padding_side = "left"

# Load Model A (Pessimistic) and Model B (Standard DPO)
model_A = load_and_merge_lora_model(BASE_MODEL_ID, PESSIMISTIC_MODEL_PATH, DEVICE, DTYPE)
model_B = load_and_merge_lora_model(BASE_MODEL_ID, DPO_MODEL_PATH, DEVICE, DTYPE)

# --- Load the Judge Model ---
print(f"\n--- Loading Judge Model: {JUDGE_MODEL_NAME} ---")
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
if judge_tokenizer.pad_token is None:
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
judge_tokenizer.padding_side = "left"

judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
judge_model.eval()
print("Judge model loaded.")

# --- Create pipelines for generation ---
pipe_A = pipeline("text-generation", model=model_A, tokenizer=tokenizer, torch_dtype=DTYPE, device=DEVICE)
pipe_B = pipeline("text-generation", model=model_B, tokenizer=tokenizer, torch_dtype=DTYPE, device=DEVICE)
judge_pipe = pipeline("text-generation", model=judge_model, tokenizer=judge_tokenizer, torch_dtype=DTYPE, device=DEVICE)


# ==============================================================================
# --- 4. EVALUATION SETUP ---
# ==============================================================================

evaluation_prompts = [
    "Write a short, heartwarming story about an old cat who finds a new friend.",
    "Explain the concept of quantum entanglement as if you were explaining it to a curious high school student.",
    "Describe a futuristic city in the year 2200 that is powered entirely by renewable energy. What does daily life look like?",
    "Generate three creative and catchy names for a new coffee shop that specializes in artisanal, single-origin beans.",
    "Write a haiku about the feeling of quiet solitude in a dense forest.",
    "I'm feeling unmotivated to exercise today. Can you give me some gentle encouragement?",
    "List the pros and cons of a four-day work week.",
    "Compose a brief, professional email declining a job offer but keeping the door open for future opportunities.",
    "What are some common logical fallacies? Please provide a simple example for two of them.",
    "Translate the following sentence into French: 'The sun shines brightly on the calm sea.'"
]

wins_model_A, wins_model_B, ties, invalid_judgements = 0, 0, 0, 0
total_comparisons = len(evaluation_prompts)

# ==============================================================================
# --- 5. BATTLE ARENA LOOP ---
# ==============================================================================

print(f"\n--- Starting comparison with {JUDGE_MODEL_NAME} as the judge ---")

for i, user_prompt_content in enumerate(evaluation_prompts):
    print("\n" + "="*50)
    print(f"--- PROMPT {i+1}/{total_comparisons} ---")
    print(f"Prompt: {user_prompt_content}")
    print("="*50)

    test_prompt_message = [{"role": "user", "content": user_prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(test_prompt_message, tokenize=False, add_generation_prompt=True)

    print("Generating response from Model A (Pessimistic)...")
    outputs_A = pipe_A(
        formatted_prompt, max_new_tokens=256, do_sample=True, temperature=0.7,
        top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id
    )
    generated_text_A = outputs_A[0]['generated_text'].replace(formatted_prompt, '').strip()
    print("\n--- Model A's Response ---\n", generated_text_A)

    print("\nGenerating response from Model B (Standard DPO)...")
    outputs_B = pipe_B(
        formatted_prompt, max_new_tokens=256, do_sample=True, temperature=0.7,
        top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id
    )
    generated_text_B = outputs_B[0]['generated_text'].replace(formatted_prompt, '').strip()
    print("\n--- Model B's Response ---\n", generated_text_B)

    is_swapped = random.choice([True, False])
    if is_swapped:
        response_for_judge_1, response_for_judge_2 = generated_text_B, generated_text_A
    else:
        response_for_judge_1, response_for_judge_2 = generated_text_A, generated_text_B

    judge_system_prompt = (
        "You are an impartial AI judge. Your task is to evaluate two responses (Response 1 and Response 2) to a given user prompt. "
        "Your evaluation should be based on helpfulness, accuracy, coherence, and adherence to instructions. "
        "You must choose only one of the following options: '[[Response 1 wins]]', '[[Response 2 wins]]', or '[[Tie]]'. "
        "Do not provide any other text or explanation."
    )
    judge_user_prompt = (
        f"USER PROMPT:\n{user_prompt_content}\n\n"
        f"--- RESPONSE 1 ---\n{response_for_judge_1}\n\n"
        f"--- RESPONSE 2 ---\n{response_for_judge_2}\n\n"
        "Which response is better? Respond with only the verdict inside double brackets."
    )
    
    # Format the prompt for the local judge model
    judge_messages = [{"role": "user", "content": judge_user_prompt}] # Mistral prefers a simpler user/assistant format
    formatted_judge_prompt = judge_tokenizer.apply_chat_template(judge_messages, tokenize=False, add_generation_prompt=True)


    print(f"\nQuerying judge model ({JUDGE_MODEL_NAME})...")
    try:
        outputs = judge_pipe(
            formatted_judge_prompt,
            max_new_tokens=20, # Only need a short response
            do_sample=False,   # Deterministic output
            temperature=0.0,
            eos_token_id=judge_tokenizer.eos_token_id
        )
        judge_decision = outputs[0]['generated_text'].replace(formatted_judge_prompt, '').strip()
        print(f"Judge Verdict: {judge_decision}")

        if "[[Response 1 wins]]" in judge_decision:
            if is_swapped: wins_model_B += 1
            else: wins_model_A += 1
        elif "[[Response 2 wins]]" in judge_decision:
            if is_swapped: wins_model_A += 1
            else: wins_model_B += 1
        elif "[[Tie]]" in judge_decision:
            ties += 1
        else:
            invalid_judgements += 1
            print("WARNING: Invalid judgement format.")

    except Exception as e:
        print(f"An error occurred during judge model inference: {e}")
        invalid_judgements += 1
    
    time.sleep(1)

# ==============================================================================
# --- 6. FINAL RESULTS ---
# ==============================================================================
print("\n" + "#"*50)
print("--- EVALUATION SUMMARY ---")
print("#"*50 + "\n")
print(f"Total Comparisons: {total_comparisons}")
print(f"Model A (Pessimistic) Wins: {wins_model_A}")
print(f"Model B (Standard DPO) Wins: {wins_model_B}")
print(f"Ties: {ties}")
print(f"Invalid Judgements: {invalid_judgements}")

if total_comparisons > invalid_judgements:
    effective_comparisons = total_comparisons - invalid_judgements
    if effective_comparisons > 0:
        win_rate_A = (wins_model_A / effective_comparisons) * 100
        win_rate_B = (wins_model_B / effective_comparisons) * 100
        tie_rate = (ties / effective_comparisons) * 100
        print("\n--- Win Rates (excluding ties and invalid judgements) ---")
        print(f"Win Rate for Model A (Pessimistic): {win_rate_A:.2f}%")
        print(f"Win Rate for Model B (Standard DPO): {win_rate_B:.2f}%")
        print(f"Tie Rate: {tie_rate:.2f}%")
    else:
        print("\nNo valid comparisons were made to compute win rates.")
else:
    print("\nAll comparisons resulted in invalid judgements.")