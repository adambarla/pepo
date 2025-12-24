"""DEPPO (Direct Ensemble Pessimistic Preference Optimization) Model."""

import functools
import logging
import math
import threading
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..factory import DEPPOFactory
from ..generator import Generator
from ..trainer import DEPPOTrainer
from ..utils import DeviceManager, HubManager
from ..utils.model_utils import get_log_probs
from .base import BaseModel

logger = logging.getLogger(__name__)

# Module-level flag to track if we've warned about missing precomputed logprobs
_warned_missing_ref_logprobs = False


class DEPPOModel(BaseModel):
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
        trainer: Optional[DEPPOTrainer] = None,
        generator: Optional[Generator] = None,
        debug: bool = False,
    ):
        """
        Initialize DEPPO Model.
        """
        self.alpha = alpha
        self.beta = beta
        self._num_models = num_networks
        self.model_id = model_id
        self._device_manager = device_manager
        self._hub_manager = hub_manager
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self._trainer = trainer
        self.generator = generator
        self.debug = debug

        # Initialize Factory
        self.factory = DEPPOFactory(
            alpha=alpha,
            beta=beta,
            num_networks=self._num_models,
            model_id=model_id,
            device_manager=device_manager,
            hub_manager=hub_manager,
            tokenizer_id=tokenizer_id,
            chat_template=chat_template,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            lora_task_type=lora_task_type,
            lora_target_modules=lora_target_modules,
            compile=compile,
        )

        self._tokenizer = self.factory.tokenizer
        self._models: list[PeftModel] | None = None  # lazy loaded
        self.epochs_per_model: list[Optional[int]] = [0] * self._num_models

        logger.info(
            f"DEPPOModel initialized with alpha={self.alpha}, "
            f"beta={self.beta}, L={self._num_models}"
        )

    @property
    def num_models(self) -> int:
        return self._num_models

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def device_manager(self) -> DeviceManager:
        return self._device_manager

    @property
    def hub_manager(self) -> HubManager:
        return self._hub_manager

    def init_trainer(self) -> None:
        """Initialize the trainer if it's a partial."""
        if isinstance(self._trainer, functools.partial):
            self._trainer = self._trainer()

    def train(
        self,
        data_manager: Any,
        max_epochs: int,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """
        Train the model using the configured trainer.

        Args:
            data_manager: Data manager for training data.
            max_epochs: Maximum number of epochs to train for.
            wandb_manager: Optional WandbManager instance for logging.
        """
        if self._trainer is None:
            raise ValueError("Trainer not configured in model config.")

        self.init_trainer()
        self._trainer.train(
            model=self,
            data_manager=data_manager,
            max_epochs=max_epochs,
            wandb_manager=wandb_manager,
            continue_training=continue_training,
        )

    def load(self, init_new: bool = False, epoch: Optional[int] = None) -> None:
        """
        Load models into memory.

        Args:
            init_new: If True, initialize new models instead of loading from hub.
            epoch: If provided, load models from this epoch checkpoint.
        """
        if self._models is not None:
            logger.warning(
                "Models are already loaded. Unload them first if you want to reload."
            )
            return
        self._models = self.factory.load_models(init_new=init_new, epoch=epoch)
        if epoch is not None:
            self.epochs_per_model = [epoch] * self._num_models

    def _check_models_loaded(self, expected_epoch: Optional[int] = None) -> None:
        """
        Check if models are loaded and optionally verify they're at the expected epoch.

        Args:
            expected_epoch: If provided, verify models are loaded from this epoch.

        Raises:
            RuntimeError: If models are not loaded or loaded from wrong epoch.
        """
        if self._models is None:
            epoch_msg = (
                f"Expected epoch: {expected_epoch}"
                if expected_epoch is not None
                else ""
            )
            raise RuntimeError(
                "Models are not loaded. Call model.load() "
                f"before using the model. {epoch_msg}"
            )

        if expected_epoch is not None:
            current_epoch = self.epochs_per_model[0] if self.epochs_per_model else None
            if current_epoch != expected_epoch:
                raise RuntimeError(
                    f"Models are loaded from epoch {current_epoch}, "
                    f"but expected epoch {expected_epoch}. "
                    f"Call model.load(epoch={expected_epoch}) "
                    "to load the correct checkpoint."
                )

    @property
    def models(self) -> list[PeftModel]:
        if self._models is None:
            raise RuntimeError(
                "Models are not loaded. Call model.load() before accessing model.models"
            )
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    def unload(self) -> None:
        """
        Unload all submodels from GPU memory to free up resources.
        """
        if not self._models:
            logger.info("Models are already unloaded")
            return

        logger.info(f"Unloading {len(self._models)} submodels from GPU memory...")

        for model in self._models:
            del model

        self._models = None
        self._device_manager.clear_cache()
        self.epochs_per_model = [0] * self._num_models

        logger.info("All submodels unloaded from GPU memory")

    def _push_models(self) -> None:
        """
        Push all ensemble models to Hub.
        Delegates to factory.
        """
        self.factory.save_model(self.models)

    def _push_model(self, model_idx: int, epochs: Optional[int] = None) -> None:
        """
        Push single model to hub.
        """
        self.factory.push_submodel(self.models[model_idx], model_idx, epochs)

    def get_tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def can_load_from_epoch(self, epoch: int) -> bool:
        """
        Check if all submodels have checkpoints at the specified epoch.

        Args:
            epoch: The epoch number to check.

        Returns:
            True if all submodels have checkpoints at the specified epoch,
            False otherwise.
        """
        for model_idx in range(self._num_models):
            submodel_name = self.get_submodel_name(model_idx)
            if not self._hub_manager.model_exists(submodel_name, epoch):
                return False
        return True

    def load_from_epoch(self, epoch: int) -> None:
        """
        Load models from a specific epoch checkpoint.

        Args:
            epoch: The epoch number to load from.
        """
        logger.info(f"Loading models from epoch {epoch} checkpoint...")

        if self._models is not None:
            logger.info("Unloading existing models before loading from checkpoint...")
            self.unload()

        self.load(init_new=False, epoch=epoch)

        logger.info(f"Successfully loaded models from epoch {epoch} checkpoint")

    def get_name(self, epoch: Optional[int] = None) -> str:
        return self.factory.get_model_name(epoch=epoch)

    def get_submodel_name(self, model_idx: int) -> str:
        return self.factory.get_submodel_name(model_idx)

    def _get_base_model_name(self) -> str:
        return self.model_id.rsplit("/", 1)[-1]

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {k: v.to(device) for k, v in batch.items()}

        chosen_ids = batch["chosen_input_ids"]
        chosen_amask = batch["chosen_attention_mask"]
        chosen_rmask = batch["chosen_response_mask"]
        reject_ids = batch["rejected_input_ids"]
        reject_amask = batch["rejected_attention_mask"]
        reject_rmask = batch["rejected_response_mask"]

        lprobs_chosen = get_log_probs(
            model, device, chosen_ids, chosen_amask, chosen_rmask, debug=self.debug
        )
        lprobs_reject = get_log_probs(
            model, device, reject_ids, reject_amask, reject_rmask, debug=self.debug
        )

        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            lprobs_chosen_ref = batch["reference_chosen_logps"]
            lprobs_reject_ref = batch["reference_rejected_logps"]
        else:
            global _warned_missing_ref_logprobs
            if not _warned_missing_ref_logprobs:
                logger.warning(
                    "Precomputed reference logprobs not found in batch. "
                    "Computing on-the-fly (2x slower). "
                    "Consider preprocessing dataset with ref_model_id "
                    "to cache them."
                )
                _warned_missing_ref_logprobs = True
            # if we don't have reference logprobs, we need to compute them
            with model.disable_adapter():
                with torch.no_grad():
                    lprobs_chosen_ref = get_log_probs(
                        model,
                        device,
                        chosen_ids,
                        chosen_amask,
                        chosen_rmask,
                        debug=self.debug,
                    )
                    lprobs_reject_ref = get_log_probs(
                        model,
                        device,
                        reject_ids,
                        reject_amask,
                        reject_rmask,
                        debug=self.debug,
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
        Predict using device-resident tensors.
        """
        self._check_models_loaded()

        log_probs_ensemble: list[Optional[torch.Tensor]] = [None] * self._num_models
        thread_exceptions: list[Optional[BaseException]] = [None] * self._num_models

        def predict_log_probs(
            model_idx: int,
            model: PeftModel,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> None:
            try:
                with torch.no_grad():
                    model.eval()
                    log_probs = self._predict_submodel(model, input_ids, attention_mask)
                    log_probs_ensemble[model_idx] = log_probs
            except BaseException as e:
                thread_exceptions[model_idx] = e

        threads = []
        for model_idx in range(self._num_models):
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

        # Propagate any thread exceptions to main thread
        for model_idx, exc in enumerate(thread_exceptions):
            if exc is not None:
                raise RuntimeError(
                    f"Exception in submodel {model_idx} prediction"
                ) from exc

        if len(log_probs_ensemble) != self._num_models:
            raise RuntimeError(
                f"Expected {self._num_models} log prob tensors, "
                f"got {len(log_probs_ensemble)}"
            )
        if len(log_probs_ensemble) == 1:
            result = log_probs_ensemble[0]
            if result is None:
                raise RuntimeError("Unexpected None in log_probs_ensemble")
            return result

        log_probs_ensemble_filtered = [
            log_probs.cpu() for log_probs in log_probs_ensemble if log_probs is not None
        ]
        log_probs_tensor: torch.Tensor = torch.stack(
            log_probs_ensemble_filtered, dim=0
        )  # (L, B, V)
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        return min_log_probs

    def predict_base_model(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self._check_models_loaded()

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

    def generate_responses(
        self,
        prompts: list[str],
        apply_chat_template: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for a list of prompts using the model's generator.
        """
        self._check_models_loaded()

        if self.generator is None:
            raise ValueError(
                "Generator not set on model. Set model.generator before "
                "calling generate_responses()."
            )
        return self.generator.generate_responses(
            model=self,
            prompts=prompts,
            apply_chat_template=apply_chat_template,
        )
