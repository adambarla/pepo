import logging
from typing import Optional, cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .utils import DeviceManager, HubManager

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        device_manager: DeviceManager,
        hub_manager: HubManager,
        compile_model: bool = False,
    ):
        self.device_manager = device_manager
        self.hub_manager = hub_manager
        self.compile_model = compile_model

    def load_tokenizer(
        self,
        model_id: str,
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
    ) -> PreTrainedTokenizerBase:
        tid = tokenizer_id or model_id
        tokenizer = AutoTokenizer.from_pretrained(tid)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if chat_template is not None:
            tokenizer.chat_template = chat_template
            logger.info("Using custom chat template from config")
        elif tokenizer.chat_template is not None:
            logger.info("Using model's built-in chat template")
        else:
            raise ValueError(
                f"Model {tid} has no built-in chat_template. "
                "Consider providing one in the model config."
            )

        return tokenizer

    def load_model(
        self,
        model_id: str,
        model_name: str,
        model_idx: int,
        lora_config: Optional[LoraConfig],
        init_new: bool = False,
        epoch: Optional[int] = None,
        custom_modules: Optional[dict[str, torch.nn.Module]] = None,
        model_class: type[PreTrainedModel] = AutoModelForCausalLM,  # type: ignore[assignment]
    ) -> PeftModel:
        """
        Load a single model (base + LoRA).

        Args:
            model_id: The base model ID (e.g. "meta-llama/Llama-2-7b-hf")
            model_name: Unique name for this specific model instance
                (for saving/loading from hub)
            model_idx: Index for device allocation
            lora_config: configuration for LoRA if initializing new
            init_new: If True, initialize new LoRA adapters.
                If False, try to load from Hub.
            epoch: If provided, load from specific epoch checkpoint.
            custom_modules: Optional dictionary of modules to attach to the base model
                under given attribute names before wrapping with PEFT.
                Useful for 'modules_to_save'.
            model_class: Class to use for loading base model.
                Defaults to AutoModelForCausalLM.
        """
        load_from_hub = not init_new and epoch is not None
        device_map = "cpu"  # Models start on CPU, moved to GPU during training
        dtype = self.device_manager.dtype

        logger.info(f"Loading base model {model_id} for {model_name} on CPU...")

        base_model = cast(
            PreTrainedModel,
            model_class.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device_map,
                attn_implementation="sdpa",
            ),
        )
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

        # Attach custom modules to base model before wrapping with PEFT
        if custom_modules:
            for attr_name, module in custom_modules.items():
                setattr(base_model, attr_name, module)

        if load_from_hub:
            logger.info(f"Loading adapter for {model_name} from hub (epoch={epoch})...")
            model: PeftModel = self.hub_manager.load_model(
                base_model, model_name, epoch=epoch
            )
        else:
            if lora_config is None:
                raise ValueError("lora_config must be provided when init_new=True")

            logger.info(f"Initializing new adapter for {model_name}...")
            model = cast(PeftModel, get_peft_model(base_model, lora_config))

            trainable, total = model.get_nb_trainable_parameters()
            trainable_m = trainable / 1000000
            total_m = total / 1000000
            logger.info(
                f"Model {model_name} has {trainable_m:.2f}M trainable "
                f"parameters out of {total_m:.2f}M total parameters "
                f"({trainable / total * 100:.2f}%)"
            )

        if self.compile_model:
            logger.warning(
                "Torch compile enabled, expect a slowdown on the first batch."
            )
            model = cast(PeftModel, torch.compile(model))

        return model

    def push_model(
        self,
        model: PeftModel,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        epochs: Optional[int] = None,
    ) -> None:
        """
        Push a single model to Hub.
        """
        self.hub_manager.push_model(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            epoch=epochs,
        )

    def load_adapter(
        self,
        model: PeftModel,
        model_name: str,
        adapter_name: str,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Load an additional adapter into an existing PeftModel.
        """
        if epoch is None:
            raise ValueError("epoch must be provided to load an existing adapter.")

        repo_id = self.hub_manager.get_repo_id(model_name, epoch)
        logger.info(f"Loading adapter {adapter_name} from {repo_id}...")
        model.load_adapter(repo_id, adapter_name=adapter_name)
