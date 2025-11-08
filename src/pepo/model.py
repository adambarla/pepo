import threading

import torch
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import DeviceManager, HubManager, Logger


class PEPOModel:
    def __init__(
        self,
        pepo_config: DictConfig,
        logger: Logger,
        device_manager: DeviceManager,
        hub_manager: HubManager,
    ):
        """
        Initialize PEPO Model.

        Args:
            pepo_config: PEPO-specific configuration (cfg.pepo).
            logger: Logger instance.
            device_manager: Device manager instance.
            hub_manager: Hub manager instance.
        """
        self.config = pepo_config
        self.logger = logger
        self.device_manager = device_manager
        self.hub_manager = hub_manager

        self.alpha = pepo_config.alpha
        self.beta = pepo_config.beta
        self.num_networks = pepo_config.num_networks

        self.models = self._load_models()
        self.tokenizer = self._init_tokenizer()

        self.logger.info(
            f"PEPOModel initialized with alpha={self.alpha}, beta={self.beta}, L={self.num_networks}"
        )

    def _init_tokenizer(self):
        """
        Initialize tokenizer for the PEPO ensemble.
        """
        tokenizer_id = self.config.model.tokenizer or self.config.model.id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _get_model_name(self, model_idx: int) -> str:
        """
        Generate model-specific repository name for a specific ensemble member.
        PEPO-specific logic lives here.

        Args:
            model_idx: Index of the ensemble model.

        Returns:
            Repository name (without base_dir, e.g., "model-name-pepo-a0.1-b0.1-L3-l0").
        """
        model_name = self.config.model.id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-pepo-a{self.alpha}-b{self.beta}-L{self.num_networks}-l{model_idx}"
        return repo_name

    def _load_models(self):
        """
        Load all ensemble models from Hub or initialize them from scratch.
        """
        models = []
        # check if all models can be loaded from hub
        load_from_hub = self.hub_manager.should_load_from_hub
        for model_idx in range(self.num_networks):
            if not load_from_hub:
                break
            if not self.hub_manager.model_exists(self._get_model_name(model_idx)):
                load_from_hub = False

        for model_idx in range(self.num_networks):
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.id,
                torch_dtype=self.device_manager.dtype,
                device_map=self.device_manager.get_device_for_model(model_idx),
            )
            base_model.config.use_cache = False
            if load_from_hub:
                repo_id = self.hub_manager.get_repo_id(self._get_model_name(model_idx))
                model = PeftModel.from_pretrained(base_model, repo_id)
                self.logger.info(
                    f"Submodel id:{model_idx} with LoRA adapter loaded successfully from {repo_id} on {self.device_manager.get_device_for_model(model_idx)}"
                )
            else:
                peft_config = LoraConfig(
                    r=self.config.model.lora.r,
                    lora_alpha=self.config.model.lora.alpha,
                    lora_dropout=self.config.model.lora.dropout,
                    bias=self.config.model.lora.bias,
                    task_type=self.config.model.lora.task_type,
                    target_modules=self.config.model.lora.target_modules,
                )
                model = get_peft_model(base_model, peft_config)
                self.logger.info(
                    f"Submodel id:{model_idx} with initialized successfully on {self.device_manager.get_device_for_model(model_idx)}"
                )
            models.append(model)
            # trainable parameters
            trainable, total = model.get_nb_trainable_parameters()
            self.logger.info(
                f"Submodel id:{model_idx} has {trainable} trainable parameters out of {total} total parameters ({trainable/total*100:.2f}%)"
            )
        return models

    def _push_models(self):
        """
        Push all ensemble models to Hub.
        """
        for model_idx in range(self.num_networks):
            self.hub_manager.push_model(
                model_name=self._get_model_name(model_idx),
                model=self.models[model_idx],
                commit_message=f"Upload PEPO ensemble model {model_idx}",
            )

    def _train_model(self, model_idx: int):
        """
        Train a single model in a thread. Each thread sets its CUDA device context
        to ensure proper GPU isolation.
        """
        device_str = self.device_manager.get_device_for_model(model_idx)
        device_idx = int(device_str.split(":")[1])
        torch.cuda.set_device(device_idx)

        model = self.models[model_idx]
        prompt = "Hello, how are you?"

        # Generate with base model (without adapters)
        base_model = model.base_model
        base_model.train()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_str)
        output = base_model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.logger.info(f"submodel {model_idx} base model output: {decoded_output}")

        # Generate with PEFT model (with adapters)
        model.train()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_str)
        output = model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.logger.info(f"submodel {model_idx} adapted model output: {decoded_output}")

    def train(self):
        """
        Train the PEPO ensemble models and save the models to the hub.
        Uses threading to run models in parallel on different GPUs.
        """
        self.logger.info("Training PEPO ensemble models...")

        # Launch training for each model in parallel threads
        threads = []
        for model_idx in range(self.num_networks):
            thread = threading.Thread(target=self._train_model, args=(model_idx,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        if self.hub_manager.should_push_to_hub:
            self._push_models()
