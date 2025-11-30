import math
import threading
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import DeviceManager, HubManager, Logger

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
        compile: bool = False,
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
            compile: Whether to compile the models.
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
        self.compile = compile

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=lora_target_modules,
        )
        self.epochs_per_network = [0] * self.num_networks

        self.tokenizer = self._init_tokenizer()
        self._models = None
        self._init_epochs()

        if self.logger:
            self.logger.info(
                f"PEPOModel initialized with alpha={self.alpha}, beta={self.beta}, L={self.num_networks}"
            )

    def _init_epochs(self):
        """Initialize epoch information without loading models."""
        load_from_hub = self.hub_manager.should_load_from_hub
        if not load_from_hub:
            for model_idx in range(self.num_networks):
                self.epochs_per_network[model_idx] = 0
            return

        for model_idx in range(self.num_networks):
            submodel_name = self._get_submodel_name(model_idx)
            if not self.hub_manager.model_exists(
                submodel_name, self.hub_manager.load_epochs
            ):
                raise ValueError(
                    f"Model {submodel_name} (epoch {self.hub_manager.load_epochs}) not found in hub"
                )

            if self.hub_manager.load_epochs is not None:
                self.epochs_per_network[model_idx] = int(self.hub_manager.load_epochs)
            else:
                self.epochs_per_network[model_idx] = None

    def ensure_models_loaded(self):
        """Ensure models are loaded before use."""
        if self._models is None:
            if self.logger:
                self.logger.info("Lazy loading models...")
            self._models = self._load_models()

    @property
    def models(self):
        self.ensure_models_loaded()
        return self._models

    def unload_models(self):
        """
        Unload all submodels from GPU memory to free up resources.
        Useful when transitioning from generation to evaluation with a different model.
        Models can be reloaded later via lazy loading (they will be loaded on next access).
        """
        if self._models is None:
            if self.logger:
                self.logger.info("Models are already unloaded")
            return

        if self.logger:
            self.logger.info(
                f"Unloading {len(self._models)} submodels from GPU memory..."
            )

        for model in self._models:
            del model

        self._models = None
        self.device_manager.clear_cache()

        if self.logger:
            self.logger.info("All submodels unloaded from GPU memory")

    def _init_tokenizer(self):
        """
        Initialize tokenizer for the PEPO ensemble.
        Handles chat template configuration from config or uses model's built-in template.
        """
        tokenizer_id = self.tokenizer_id or self.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        return self.tokenizer

    def get_min_epochs(self) -> Optional[int]:
        """
        Get the minimum number of epochs across all networks in the ensemble.

        Returns:
            Minimum epochs. Returns None if any network has unknown epochs (loaded from best),
            0 if all are newly instantiated.
        """
        if not self.epochs_per_network:
            return 0

        if any(e is None for e in self.epochs_per_network):
            return None

        return min(self.epochs_per_network)

    def _get_model_name(self) -> str:
        """
        Generate model-specific repository name for the ensemble.
        e.g., "model-name-pepo-a0.1-b0.1-L3".
        """
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-pepo-a{self.alpha}-b{self.beta}-L{self.num_networks}"
        return repo_name

    def _get_submodel_name(self, model_idx: int) -> str:
        """
        Generate submodel name for a specific ensemble member.
        e.g., "model-name-pepo-a0.1-b0.1-L3-l0".
        """
        model_name = self._get_model_name()
        return f"{model_name}-l{model_idx}"

    def _get_base_model_name(self) -> str:
        """
        Generate base model name for the ensemble.
        e.g., "SmolLM2-1.7B".
        """
        return self.model_id.rsplit("/", 1)[-1]

    def _load_model(self, model_idx: int, load_from_hub: bool) -> AutoModelForCausalLM:
        device_map = self.device_manager.get_device_for_model(model_idx)
        dtype = self.device_manager.dtype

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        base_model.config.use_cache = False

        model_name = self._get_submodel_name(model_idx)
        if load_from_hub:
            model = self.hub_manager.load_model(base_model, model_name)
        else:
            model = get_peft_model(base_model, self.lora_config)
            if self.logger:
                trainable, total = model.get_nb_trainable_parameters()
                trainable = trainable / 1000000
                total = total / 1000000
                self.logger.info(
                    f"Model {model_name} has {trainable:.2f}M trainable parameters out of {total:.2f}M total parameters ({trainable/total*100:.2f}%)"
                )
        if self.compile:
            model = torch.compile(model)
        return model

    def _load_models(self):
        """
        Load all ensemble models from Hub or initialize them from scratch.
        """
        models = []
        load_from_hub = self.hub_manager.should_load_from_hub

        for model_idx in range(self.num_networks):
            models.append(self._load_model(model_idx, load_from_hub))
        return models

    def _push_model(self, model_idx: int, epochs: Optional[int] = None):
        """
        Push a single model to Hub.

        Args:
            model_idx: Index of the model in the ensemble.
            epochs: Optional number of epochs. If provided, appends "-e{epochs}" to model name.
                    Use None for final push without epoch suffix.
        """
        self.hub_manager.push_model(
            model_name=self._get_submodel_name(model_idx),
            model=self.models[model_idx],
            tokenizer=self.tokenizer,
            model_idx=model_idx,
            epochs=epochs,
        )

    def _push_models(self):
        """
        Push all ensemble models to Hub without epoch suffix (final version).
        """
        for model_idx in range(self.num_networks):
            self._push_model(model_idx, epochs=None)

    def _get_lprobs(
        self,
        model: AutoModelForCausalLM,
        device: torch.device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the log probabilities of the response tokens.
        Creates a mask of the responses and
        """
        # B, T = input_ids.shape
        # V = model.config.vocab_size

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad() if model.training is False else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T , V); shifted by 1 (removed first, added new)

        logits = logits[:, :-1, :]  # (B, T-1, V); remove the new token
        labels = input_ids[:, 1:]  # (B, T-1); remove the first token
        response_mask = response_mask[:, 1:]  # (B, T-1); remove the first token

        log_probs = F.log_softmax(logits, dim=-1)  # (B, T-1, V); log prob over V dim
        # select only the log probs for the labels, (B, T-1)
        log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs * response_mask.float()  # mask out the response tokens
        log_probs_sum = log_probs.sum(dim=-1)  # (B,) sum the log probs of response tokens
        return log_probs_sum  # (B,)

    def _loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: AutoModelForCausalLM,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {k: v.to(device) for k, v in batch.items()}

        chosen_ids = batch["chosen_input_ids"]
        chosen_amask = batch["chosen_attention_mask"]
        chosen_rmask = batch["chosen_response_mask"]
        reject_ids = batch["rejected_input_ids"]
        reject_amask = batch["rejected_attention_mask"]
        reject_rmask = batch["rejected_response_mask"]

        lprobs_chosen = self._get_lprobs(
            model, device, chosen_ids, chosen_amask, chosen_rmask
        )
        lprobs_reject = self._get_lprobs(
            model, device, reject_ids, reject_amask, reject_rmask
        )
        with model.disable_adapter():  # type: ignore[operator]
            lprobs_chosen_ref = self._get_lprobs(
                model,
                device,
                chosen_ids,
                chosen_amask,
                chosen_rmask,
            )
            lprobs_reject_ref = self._get_lprobs(
                model,
                device,
                reject_ids,
                reject_amask,
                reject_rmask,
            )

        pi_log_ratio = lprobs_chosen - lprobs_reject
        ref_log_ratio = lprobs_chosen_ref - lprobs_reject_ref
        alpha_offset = math.log(1.0 + self.alpha)
        argument = self.beta * (pi_log_ratio - ref_log_ratio - alpha_offset)
        dpo_loss_components = -F.logsigmoid(argument)
        loss = dpo_loss_components.mean()
        return loss, lprobs_chosen, lprobs_reject

    def _predict_submodel(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict log probabilities assuming input_ids and attention_mask are already on model.device.
        This avoids unnecessary device transfers during generation.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        last_logits = logits[:, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        return log_probs

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict using device-resident tensors. Each model uses its own input_ids tensor
        that stays on its device throughout generation, avoiding repeated CPU-GPU transfers.

        Args:
            device_input_ids: List of input_ids tensors, one per model, each on its model's device
            device_attention_masks: List of attention_mask tensors, one per model, each on its model's device

        Returns:
            Minimum log probabilities across ensemble (on CPU)
        """
        log_probs_ensemble: list[Optional[torch.Tensor]] = [None] * self.num_networks

        def predict_log_probs(model_idx, model, input_ids, attention_mask):
            with torch.no_grad():
                model.eval()
                log_probs = self._predict_submodel(model, input_ids, attention_mask)
                log_probs_ensemble[model_idx] = log_probs

        threads = []
        for model_idx in range(self.num_networks):
            thread = threading.Thread(
                target=predict_log_probs,
                args=(
                    model_idx,
                    self.models[model_idx],
                    device_input_ids[model_idx],
                    device_attention_masks[model_idx],
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if len(log_probs_ensemble) != self.num_networks:
            raise RuntimeError(
                f"Expected {self.num_networks} log prob tensors, got {len(log_probs_ensemble)}"
            )
        if len(log_probs_ensemble) == 1:
            return log_probs_ensemble[0]

        log_probs_ensemble = [log_probs.cpu() for log_probs in log_probs_ensemble]
        log_probs_tensor: torch.Tensor = torch.stack(
            log_probs_ensemble, dim=0
        )  # (L, B, V)
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        return min_log_probs

    def predict_base_model(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids must be a 2D tensor")

        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        model = self.models[0]
        device = torch.device(self.device_manager.get_device_for_model(0))
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        # disable adapter
        with torch.no_grad():
            with model.disable_adapter():
                model.eval()
                log_probs = self._predict_submodel(
                    model, input_ids, attention_mask
                )  # (B, V)
        return log_probs.cpu()

    def _top_p_sampling(self, logits, top_p=0.9, temperature=1.0):
        """
        Perform top-p (nucleus) sampling on the given logits.

        Args:
            logits (torch.Tensor): The logits from the model of shape (batch_size, vocab_size).
            top_p (float): The cumulative probability threshold for nucleus sampling.
            temperature (float): The temperature for scaling logits.
        Returns:
            torch.Tensor: The sampled token indices of shape (batch_size,).
        """
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        filtered_probs = F.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        return sampled_indices

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 1024,
        use_ensamble: bool = True,
        sample_missing_token: bool = False,
        greedy_sampling: bool = False,
        top_p_sampling: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if greedy_sampling and top_p_sampling:
            raise ValueError("Greedy sampling and top-p sampling cannot be used together")
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        batch_size = input_ids.shape[0]

        # Initialize device-resident tensors once at the start
        # This avoids repeated CPU-GPU transfers as sequences grow
        device_input_ids = []
        device_attention_masks = []
        for model_idx in range(self.num_networks):
            device = torch.device(self.device_manager.get_device_for_model(model_idx))
            device_input_ids.append(input_ids.to(device))
            device_attention_masks.append(attention_mask.to(device))

        stop_signal = torch.zeros(batch_size, dtype=torch.bool).cpu()

        pbar = tqdm(range(max_length - input_ids.shape[1]))
        for i in pbar:
            # Clear cache periodically to reduce fragmentation
            if i > 0 and i % 100 == 0:
                self.device_manager.clear_cache()

            if use_ensamble:
                log_probs = self.predict(device_input_ids, device_attention_masks)
            else:
                # For base model, use the first device's tensors
                model = self.models[0]
                with torch.no_grad():
                    with model.disable_adapter():
                        model.eval()
                        log_probs = self._predict_submodel(
                            model, device_input_ids[0], device_attention_masks[0]
                        )
                log_probs = log_probs
            min_probs = torch.exp(log_probs)

            # TODO(adam): handle top k sampling, temperature sampling, etc.
            # TODO(adam): handle resampling max attempts

            # resample where we got missing token until we get a non-missing token
            if sample_missing_token:
                missing_token_id = len(self.tokenizer)
                missing_probs = torch.clamp(1 - torch.sum(min_probs, dim=-1), min=0.0)
                min_probs = torch.cat([min_probs, missing_probs.unsqueeze(-1)], dim=-1)
                missing_mask = torch.ones(
                    batch_size, dtype=torch.bool, device=min_probs.device
                )
                sampled_token_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=min_probs.device
                )
                while True:
                    new_sampled_token_ids = torch.multinomial(
                        min_probs[missing_mask], num_samples=1
                    ).squeeze(-1)
                    sampled_token_ids[missing_mask] = new_sampled_token_ids
                    missing_mask = sampled_token_ids == missing_token_id
                    if not torch.any(missing_mask):
                        break
            else:
                if greedy_sampling:
                    sampled_token_ids = torch.argmax(min_probs, dim=-1)
                elif top_p_sampling:
                    sampled_token_ids = self._top_p_sampling(
                        log_probs,
                        top_p=top_p,
                        temperature=temperature,
                    )
                else:
                    min_probs = min_probs / torch.sum(min_probs, dim=-1, keepdim=True)
                    sampled_token_ids = torch.multinomial(min_probs, num_samples=1)

            stop_signal = stop_signal.to(device=sampled_token_ids.device) | (
                sampled_token_ids == self.tokenizer.eos_token_id
            )
            # Append new tokens directly on each device to avoid CPU-GPU transfers
            for model_idx in range(self.num_networks):
                device = torch.device(self.device_manager.get_device_for_model(model_idx))
                new_token_tensor = sampled_token_ids.to(device).unsqueeze(-1)
                device_input_ids[model_idx] = torch.cat(
                    [device_input_ids[model_idx], new_token_tensor], dim=1
                )
                # new maxk should be inverse stop signal
                device_attention_masks[model_idx] = torch.cat(
                    [
                        device_attention_masks[model_idx],
                        ~stop_signal.unsqueeze(-1).to(device),
                    ],
                    dim=1,
                )

            pbar.set_postfix({"stopped": f"{stop_signal.sum().item()}/{batch_size}"})

            if sampled_token_ids[0] == self.tokenizer.eos_token_id:
                if self.logger:
                    self.logger.debug(f"Generated EOS token at step {i}")
            if torch.all(stop_signal):
                break
        if self.logger:
            self.logger.debug(
                f"Generated sequence idx=0:\n{self.tokenizer.decode(device_input_ids[0].cpu()[0], skip_special_tokens=True)}"
            )

        # Return the first device's tensors (move to CPU for consistency with original API)
        return device_input_ids[0].cpu(), device_attention_masks[0].cpu()

    def generate_base_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            use_ensamble=False,
        )
