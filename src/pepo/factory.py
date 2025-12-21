import logging
from typing import Any, List, Optional, cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .utils import DeviceManager, HubManager

logger = logging.getLogger(__name__)


class PEPOFactory:
    def __init__(
        self,
        alpha: float,
        beta: float,
        num_networks: int,
        model_id: str,
        device_manager: DeviceManager,
        hub_manager: HubManager,
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        lora_target_modules: str = "all-linear",
        compile: bool = False,
    ):
        self.alpha = alpha
        self.beta = beta
        self.num_networks = num_networks
        self.model_id = model_id
        self.device_manager = device_manager
        self.hub_manager = hub_manager
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_task_type = lora_task_type
        self.lora_target_modules = lora_target_modules
        self.compile = compile

        self.tokenizer = self._init_tokenizer()

    def load_models(
        self, init_new: bool = False, epoch: Optional[int] = None
    ) -> List[PeftModel]:
        models = []
        logger.info(f"Loading {self.num_networks} models...")

        for model_idx in range(self.num_networks):
            models.append(self._load_model(model_idx, init_new=init_new, epoch=epoch))

        return models

    def save_model(self, models: List[Any], epochs: Optional[int] = None) -> None:
        """
        Push the model to the hub.
        """
        for i, submodel in enumerate(models):
            self.push_submodel(submodel, i, epochs)

    def push_submodel(
        self, model: Any, model_idx: int, epochs: Optional[int] = None
    ) -> None:
        """
        Push a single model to Hub.

        Args:
            model: The submodel instance.
            model_idx: Index of the model in the ensemble.
            epochs: Optional number of epochs. If provided, appends "-e{epochs}"
                to model name. Use None for final push without epoch suffix.
        """
        self.hub_manager.push_model(
            model_name=self.get_submodel_name(model_idx),
            model=model,
            tokenizer=self.tokenizer,
            model_idx=model_idx,
            epoch=epochs,
        )

    def _init_tokenizer(self) -> AutoTokenizer:
        tokenizer_id = self.tokenizer_id or self.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template
            logger.info("Using custom chat template from config")
        elif tokenizer.chat_template is not None:
            logger.info("Using model's built-in chat template")
        else:
            raise ValueError(
                f"Model {tokenizer_id} has no built-in chat_template. "
                "Consider providing one in the model config."
            )

        return tokenizer

    def get_repo_name(self, epoch: Optional[int] = None) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-pepo-a{self.alpha}-b{self.beta}-L{self.num_networks}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def get_model_name(self, epoch: Optional[int] = None) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-a{self.alpha}-b{self.beta}-L{self.num_networks}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def get_submodel_name(self, model_idx: int) -> str:
        model_name = self.get_model_name()
        return f"{model_name}-l{model_idx}"

    def _load_model(
        self, model_idx: int, init_new: bool = False, epoch: Optional[int] = None
    ) -> PeftModel:
        load_from_hub = not init_new and epoch is not None
        device_map = self.device_manager.get_device_for_model(model_idx)
        dtype = self.device_manager.dtype

        base_model = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=device_map,
                attn_implementation="sdpa",
            ),
        )
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

        model_name = self.get_submodel_name(model_idx)

        if load_from_hub:
            model: PeftModel = self.hub_manager.load_model(
                base_model, model_name, epoch=epoch
            )

        else:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type=self.lora_task_type,
                target_modules=self.lora_target_modules,
            )
            model = cast(PeftModel, get_peft_model(base_model, lora_config))

            trainable, total = model.get_nb_trainable_parameters()
            trainable = trainable // 1000000
            total = total // 1000000
            logger.info(
                f"Model {model_name} has {trainable:.2f}M trainable "
                f"parameters out of {total:.2f}M total parameters "
                f"({trainable / total * 100:.2f}%)"
            )

        if self.compile:
            logger.warning(
                "Torch compile enabled, expect a slowdown on the first batch."
            )
            model = cast(PeftModel, torch.compile(model))

        return model
