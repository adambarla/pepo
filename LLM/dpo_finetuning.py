from huggingface_hub import login

login()

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

dataset = load_dataset(path="trl-lib/ultrafeedback_binarized", split="train")

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float32,
).to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-DPO"
finetune_tags = ["smol-course", "module_1"]

# Training arguments
training_args = DPOConfig(
    # Training batch size per GPU
    per_device_train_batch_size=4,
    # Number of updates steps to accumulate before performing a backward/update pass
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    gradient_accumulation_steps=4,
    # Saves memory by not storing activations during forward pass
    # Instead recomputes them during backward pass
    gradient_checkpointing=True,
    # Base learning rate for training
    learning_rate=5e-5,
    # Learning rate schedule - 'cosine' gradually decreases LR following cosine curve
    lr_scheduler_type="cosine",
    # Total number of training steps
    max_steps=200,
    # Disables model checkpointing during training
    save_strategy="no",
    # How often to log training metrics
    logging_steps=1,
    # Directory to save model outputs
    output_dir="smol_dpo_output",
    # Number of steps for learning rate warmup
    warmup_steps=100,
    # Use bfloat16 precision for faster training
    bf16=True,
    # Disable wandb/tensorboard logging
    report_to="none",
    # Keep all columns in dataset even if not used
    remove_unused_columns=False,
    # Enable MPS (Metal Performance Shaders) for Mac devices
    use_mps_device=device == "mps",
    # Model ID for HuggingFace Hub uploads
    hub_model_id=finetune_name,
    # DPO-specific temperature parameter that controls the strength of the preference model
    # Lower values (like 0.1) make the model more conservative in following preferences
    beta=0.1,
    # Maximum length of the input prompt in tokens
    max_prompt_length=1024,
    # Maximum combined length of prompt + response in tokens
    max_length=1536,
)

trainer = DPOTrainer(
    # The model to be trained
    model=model,
    # Training configuration from above
    args=training_args,
    # Dataset containing preferred/rejected response pairs
    train_dataset=dataset,
    # Tokenizer for processing inputs
    processing_class=tokenizer,
    # DPO-specific temperature parameter that controls the strength of the preference model
    # Lower values (like 0.1) make the model more conservative in following preferences
    # beta=0.1,
    # Maximum length of the input prompt in tokens
    # max_prompt_length=1024,
    # Maximum combined length of prompt + response in tokens
    # max_length=1536,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(f"./{finetune_name}")

# Save to the huggingface hub if login (HF_TOKEN is set)
if os.getenv("HF_TOKEN"):
    trainer.push_to_hub(tags=finetune_tags)




