from dataclasses import dataclass
from typing import Optional


@dataclass
class BackboneConfig:
    """Configuration for the backbone model."""

    model_id: str

    # LoRA / PEFT settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: str = "all-linear"

    # Tokenizer settings
    tokenizer_id: Optional[str] = None
    chat_template: Optional[str] = None

    # Compiler
    compile: bool = False

    # Batch sizes (informational for trainer, but kept with model config)
    train_batch_size: int = 32
    eval_batch_size: int = 64
    generator_batch_size: int = 128
