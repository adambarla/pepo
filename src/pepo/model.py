import threading
from typing import Optional

from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import DataManager, DeviceManager, HubManager, Logger

# TODO(adam): possible abstraction here is to define a PEPOSubModel base class that handles the logic of a single model in the ensemble
#             we could inherit this class to define submodels such as smollm, gemma, etc. which would handle their own chat template, tokenizer, etc.
#             this would allow for more modularity and easier to extend to new models.
#             we would define hydra configs for each submodel we support and in the pepo config just specify the submodel with a single word (smollm) which would load the appropriate config and instantiate the submodel.
#             ISSUE: instantiation of L submodels requires a factory/passing config to pepo


class PEPOModel:
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
        logger: Optional[Logger] = None,
    ):
        """
        Initialize PEPO Model.

        Args:
            alpha: Pessimistic margin alpha parameter.
            beta: DPO beta parameter, controls strength of preference.
            num_networks: Number of networks in the ensemble (L).
            model_id: HuggingFace model ID (e.g., "HuggingFaceTB/SmolLM2-1.7B").
            device_manager: Device manager instance.
            hub_manager: Hub manager instance.
            tokenizer_id: HuggingFace tokenizer ID. If None, uses model_id.
            chat_template: Custom chat template string. If None, uses model's built-in template.
            lora_r: LoRA rank parameter.
            lora_alpha: LoRA alpha parameter.
            lora_dropout: LoRA dropout rate.
            lora_bias: LoRA bias setting.
            lora_task_type: LoRA task type.
            lora_target_modules: LoRA target modules.
            logger: Optional logger instance.
        """
        self.alpha = alpha
        self.beta = beta
        self.num_networks = num_networks
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self.logger = logger
        self.device_manager = device_manager
        self.hub_manager = hub_manager

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=lora_target_modules,
        )

        self.models = self._load_models()
        self.tokenizer = self._init_tokenizer()

        if self.logger:
            self.logger.info(
                f"PEPOModel initialized with alpha={self.alpha}, beta={self.beta}, L={self.num_networks}"
            )

    def _init_tokenizer(self):
        """
        Initialize tokenizer for the PEPO ensemble.
        Handles chat template configuration from config or uses model's built-in template.
        """
        tokenizer_id = self.tokenizer_id or self.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Handle chat template
        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template
            if self.logger:
                self.logger.info("Using custom chat template from config")
                self.logger.debug(f"Chat template: {self.chat_template}")
        elif tokenizer.chat_template is not None:
            if self.logger:
                self.logger.info("Using model's built-in chat template")
        else:
            raise ValueError(
                f"Model {tokenizer_id} has no built-in chat_template. Consider providing one in the model config."
            )

        return tokenizer

    def get_tokenizer(self):
        """
        Get the tokenizer used by the PEPO ensemble.

        Returns:
            The tokenizer instance.
        """
        return self.tokenizer

    def _get_model_name(self, model_idx: int) -> str:
        """
        Generate model-specific repository name for a specific ensemble member.
        PEPO-specific logic lives here.

        Args:
            model_idx: Index of the ensemble model.

        Returns:
            Repository name (without base_dir, e.g., "model-name-pepo-a0.1-b0.1-L3-l0").
        """
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-pepo-a{self.alpha}-b{self.beta}-L{self.num_networks}-l{model_idx}"
        return repo_name

    def _load_models(self):
        """
        Load all ensemble models from Hub or initialize them from scratch.
        """
        models = []
        load_from_hub = self.hub_manager.should_load_from_hub
        for model_idx in range(self.num_networks):
            if not load_from_hub:
                break
            if not self.hub_manager.model_exists(self._get_model_name(model_idx)):
                load_from_hub = False

        for model_idx in range(self.num_networks):
            device_map = self.device_manager.get_device_for_model(model_idx)
            dtype = self.device_manager.dtype

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device_map,
            )
            base_model.config.use_cache = False
            if load_from_hub:
                repo_id = self.hub_manager.get_repo_id(self._get_model_name(model_idx))
                model = PeftModel.from_pretrained(base_model, repo_id)
            else:
                model = get_peft_model(base_model, self.lora_config)
            models.append(model)

            if self.logger:
                trainable, total = model.get_nb_trainable_parameters()
                trainable = trainable / 1000000
                total = total / 1000000
                self.logger.info(
                    f"Submodel id:{model_idx} on {device_map} has {trainable:.2f}M trainable parameters out of {total:.2f}M total parameters ({trainable/total*100:.2f}%)"
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

    def _train_model(
        self,
        model_idx: int,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        n_epochs: int = 1,
    ):
        """
        Train a single model in a thread. Each thread sets its CUDA device context
        to ensure proper GPU isolation.
        """
        device_str = self.device_manager.get_device_for_model(model_idx)
        model = self.models[model_idx]

        if self.logger:
            train_size = len(train_loader.dataset)  # type: ignore[arg-type]
            eval_size = len(eval_loader.dataset)  # type: ignore[arg-type]
            self.logger.info(
                f"Model {model_idx} - Train: size={train_size}, batches={len(train_loader)} - Eval: size={eval_size}, batches={len(eval_loader)}"
            )

        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device_str) for k, v in batch.items()}

                for k, v in batch.items():
                    self.logger.debug(
                        f"Model {model_idx} - Batch key: {k} - Shape: {v.shape}"
                    )
                # self.logger.debug(f"Model {model_idx} - Batch shapes: {batch}")

                break
            break

    def train(
        self,
        data_manager: DataManager,
        batch_size: int,
    ):
        """
        Train the PEPO ensemble models and save the models to the hub.
        Uses threading to run models in parallel on different GPUs.

        Args:
            data_manager: DataManager instance for getting dataloaders.
            batch_size: Batch size for training.
        """
        if self.logger:
            self.logger.info("Training PEPO ensemble models...")

        # Launch training for each model in parallel threads
        threads = []
        for model_idx in range(self.num_networks):
            train_loader = data_manager.get_dataloader(
                model_idx=model_idx,
                partition="train",
                batch_size=batch_size,
            )
            eval_loader = data_manager.get_dataloader(
                model_idx=model_idx,
                partition="eval",
                batch_size=batch_size,
            )

            thread = threading.Thread(
                target=self._train_model, args=(model_idx, train_loader, eval_loader)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.hub_manager.should_push_to_hub:
            self._push_models()
