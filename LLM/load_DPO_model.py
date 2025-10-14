# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

device = (
            "cuda"
                if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                    )
# Load the model and tokenizer
model_name = "SmolLM2-FT-DPO"
model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name
            ).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# Set up the chat format

prompt = "Hey! Do you like reading books?"

# Format with template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("Before training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
