import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import os
import time
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
for llm1 in ["HuggingFaceTB/SmolLM2-135M-Instruct", 
            "./dpo_custom_tinyllama_ultrafeedback",
            "./ratings_dpo_custom_tinyllama_ultrafeedback"]:
    for llm2 in ["HuggingFaceTB/SmolLM2-135M-Instruct", 
            "./dpo_custom_tinyllama_ultrafeedback",
            "./ratings_dpo_custom_tinyllama_ultrafeedback"]:
        # --- Configuration (from your original snippet or similar) ---
        if torch.cuda.is_available():
            DEVICE = "cuda"
            DTYPE = torch.bfloat16 # bfloat16 is usually best for NVIDIA GPUs (Ampere architecture and newer)
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
            DTYPE = torch.float16 # MPS typically supports float16 (half-precision), but not bfloat16.
                                # If float16 causes issues, fall back to torch.float32
            print("Using MPS backend. Note: BFloat16 is not supported on MPS, using Float16.")
        else:
            DEVICE = "cpu"
            DTYPE = torch.float32 # CPU runs best with float32
        # --- Judge Model Configuration ---
        # For OpenAI API (Recommended for Judge)
        # Make sure to pip install openai
        import openai
        from openai import OpenAI
        # Set your OpenAI API key
        # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Better to set as environment variable
        # If you set it as an env variable, it's picked up automatically:
        # client = OpenAI()
        # Or pass it explicitly:
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client = OpenAI(api_key="YOUR_OPENAI_API_KEY") # Replace with your actual key or set env var
        JUDGE_MODEL_NAME = "gpt-4-turbo" # Or "gpt-3.5-turbo", "claude-3-opus-20240229", etc.

        print("\n--- Loading models for comparison ---")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token # Or specify a different pad_token if needed
        tokenizer.padding_side = "left" # Important for generation

        # Model 1: Your fine-tuned model
        if not llm1 == "HuggingFaceTB/SmolLM2-135M-Instruct": 
            model_A_base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )
            model_A = PeftModel.from_pretrained(model_A_base, llm1)
            model_A = model_A.merge_and_unload()
            
        else:
            model_A = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )
        model_A.eval()
        if not llm2 == "HuggingFaceTB/SmolLM2-135M-Instruct":
            model_B_base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )

            model_B = PeftModel.from_pretrained(model_B_base, llm2)
            model_B = model_B.merge_and_unload()
            
        else:
            model_B = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )
        model_B.eval()

        # Create pipelines for generation
        pipe_A = pipeline(
            "text-generation",
            model=model_A,
            tokenizer=tokenizer,
            torch_dtype=DTYPE
        )
        pipe_B = pipeline(
            "text-generation",
            model=model_B,
            tokenizer=tokenizer,
            torch_dtype=DTYPE
        )

        print("Models loaded successfully.")

        # --- Evaluation Prompts ---
        evaluation_prompts = [
            "Write a short, heartwarming story about an old cat.",
            "Explain the concept of quantum entanglement in simple terms.",
            "Describe a futuristic city powered by renewable energy.",
            "Generate a creative name for a new coffee shop.",
            "Write a haiku about a rainy day."
        ]

        # --- Evaluation Metrics ---
        wins_model_A = 0
        wins_model_B = 0
        ties = 0
        invalid_judgements = 0
        total_comparisons = len(evaluation_prompts)

        print(f"\n--- Starting comparison with {JUDGE_MODEL_NAME} ---")

        for i, user_prompt_content in enumerate(evaluation_prompts):
            print(f"\n--- Prompt {i+1}/{total_comparisons} ---")
            test_prompt_message = [{"role": "user", "content": user_prompt_content}]
            formatted_prompt = tokenizer.apply_chat_template(
                test_prompt_message,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate responses from both models
            print(f"Generating response from Model A...")
            outputs_A = pipe_A(
                formatted_prompt,
                max_new_tokens=150, # Adjust as needed
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text_A = outputs_A[0]['generated_text'].replace(formatted_prompt, '').strip()

            print(f"Generating response from Model B...")
            outputs_B = pipe_B(
                formatted_prompt,
                max_new_tokens=150, # Adjust as needed
                do_sample=True,
                temperature=0.7, # Use lower temp for base model often
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text_B = outputs_B[0]['generated_text'].replace(formatted_prompt, '').strip()

            # Randomly assign responses to A and B for the judge to avoid positional bias
            is_swapped = random.choice([True, False])
            if is_swapped:
                response_for_judge_A = generated_text_B
                response_for_judge_B = generated_text_A
                model_A_label = "Model B" # For internal tracking by judge
                model_B_label = "Model A" # For internal tracking by judge
                print(f" (Responses swapped for judge: Actual Model A is Response B, Actual Model B is Response A)")
            else:
                response_for_judge_A = generated_text_A
                response_for_judge_B = generated_text_B
                model_A_label = "Model A"
                model_B_label = "Model B"
                print(f" (Responses not swapped for judge)")
            JUDGE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Or "mistralai/Mixtral-8x7B-Instruct-v0.1", "google/gemma-7b-it" etc.
            # --- Load the Judge Model and Tokenizer ---
            print(f"Loading judge model: {JUDGE_MODEL_NAME}...")

            judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
            # Ensure the tokenizer has a pad_token, critical for batching in pipelines
            if judge_tokenizer.pad_token is None:
                judge_tokenizer.pad_token = judge_tokenizer.eos_token
            judge_tokenizer.padding_side = "left" # Often better for generation

            judge_model = AutoModelForCausalLM.from_pretrained(
                JUDGE_MODEL_NAME,
                torch_dtype=DTYPE,
                device_map=DEVICE
            )
            judge_model.eval() # Set to evaluation mode

            # Create a text generation pipeline
            judge_pipeline = pipeline(
                "text-generation",
                model=judge_model,
                tokenizer=judge_tokenizer,
                device=DEVICE, # Or directly use model.device if device_map="auto" was used
                torch_dtype=DTYPE,
                max_new_tokens=10, # Expect short output like "A wins"
                temperature=0.0,   # Make it deterministic
                do_sample=False,   # Ensure greedy decoding
                eos_token_id=judge_tokenizer.eos_token_id,
                pad_token_id=judge_tokenizer.pad_token_id # Important for batching, if you eventually judge multiple at once
            )

            # --- Inside your judging loop ---
            # (Assume user_prompt_content, response_for_judge_A, response_for_judge_B, is_swapped are defined)

            judge_system_message_content = (
                "You are an impartial AI judge. Your task is to evaluate two responses (Response A and Response B) "
                "to a given prompt. Your criteria are: adherence to prompt, coherence, creativity, and overall quality. "
                "You MUST choose only one of the following options: 'A wins', 'B wins', or 'Tie'. "
                "Do NOT provide any other text, explanation, or reasoning."
            )
            judge_user_message_content = (
                f"Original Prompt:\n{user_prompt_content}\n\n"
                f"Response A:\n{response_for_judge_A}\n\n"
                f"Response B:\n{response_for_judge_B}\n\n"
                "Which response is better? Respond with 'A wins', 'B wins', or 'Tie'."
            )

            # Format messages for the chosen model's chat template
            # This is crucial for instruct-tuned models
            judge_messages = [
                {"role": "system", "content": judge_system_message_content},
                {"role": "user", "content": judge_user_message_content}
            ]

            formatted_judge_prompt = judge_tokenizer.apply_chat_template(
                judge_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            print(f"Querying judge model ({JUDGE_MODEL_NAME})...")
            try:
                outputs = judge_pipeline(formatted_judge_prompt)
                
                # The output will contain the full prompt + generated text.
                # We only want the generated text.
                generated_text = outputs[0]['generated_text']
                judge_decision = generated_text.replace(formatted_judge_prompt, '').strip().lower()

                print(f"Judge decision: {judge_decision}")

                # Parse judge decision (remains the same)
                if "a wins" in judge_decision:
                    if is_swapped:
                        wins_model_B += 1
                        print(f"Result: Model B (Original) wins for this prompt.")
                    else:
                        wins_model_A += 1
                        print(f"Result: Model A (Fine-tuned) wins for this prompt.")
                elif "b wins" in judge_decision:
                    if is_swapped:
                        wins_model_A += 1
                        print(f"Result: Model A (Fine-tuned) wins for this prompt.")
                    else:
                        wins_model_B += 1
                        print(f"Result: Model B (Original) wins for this prompt.")
                elif "tie" in judge_decision:
                    ties += 1
                    print(f"Result: Tie for this prompt.")
                else:
                    invalid_judgements += 1
                    print(f"Result: Invalid judgement '{judge_decision}'. Skipping this comparison.")

            except Exception as e: # Catch any errors from pipeline execution
                print(f"An error occurred during judge model inference: {e}")
                invalid_judgements += 1

            # Small delay to avoid hitting API rate limits
            time.sleep(1)

        # --- Final Results ---
        print("\n--- Evaluation Summary ---")
        print(f"Total Comparisons: {total_comparisons}")
        print(f"Model A (Fine-tuned) Wins: {wins_model_A}")
        print(f"Model B (Original) Wins: {wins_model_B}")
        print(f"Ties: {ties}")
        print(f"Invalid Judgements: {invalid_judgements}")

        if total_comparisons > invalid_judgements:
            effective_comparisons = total_comparisons - invalid_judgements
            win_rate_A = (wins_model_A / effective_comparisons) * 100
            win_rate_B = (wins_model_B / effective_comparisons) * 100
            tie_rate = (ties / effective_comparisons) * 100
            print(f"\nWinning Rate for {llm1} (Fine-tuned): {win_rate_A:.2f}%")
            print(f"Winning Rate for {llm2} (Original): {win_rate_B:.2f}%")
            print(f"Tie Rate: {tie_rate:.2f}%")
        else:
            print("\nNot enough valid comparisons to compute rates.")
