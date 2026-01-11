"""DEPPO (Direct Ensemble Pessimistic Preference Optimization) Model."""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, cast

if TYPE_CHECKING:
    from ..utils import DeviceManager, HubManager

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import PreTrainedTokenizerBase

from ..generator import Generator
from ..loader import CheckpointManager
from ..trainer import EnsembleTrainer
from ..utils import get_device_manager, get_hub_manager
from ..utils.model_utils import get_log_probs
from .base import BaseModel

logger = logging.getLogger(__name__)

# Module-level flag to track if we've warned about missing precomputed logprobs
_warned_missing_ref_logprobs = False


class DEPPOModel(BaseModel):
    """Direct Ensemble Pessimistic Preference Optimization Model."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        num_networks: int,
        model_id: str,
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        lora_target_modules: str = "all-linear",
        compile: bool = False,
        trainer: Optional[EnsembleTrainer] = None,
        generator: Optional[Generator] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize DEPPO Model.
        """
        self.alpha = alpha
        self.beta = beta
        self._num_models = num_networks
        self.model_id = model_id
        self._device_manager = get_device_manager()
        self._hub_manager = get_hub_manager()
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self._trainer = trainer
        self.generator = generator
        self.debug = debug

        self.compile_model = compile

        self._checkpoint_manager = CheckpointManager(
            device_manager=self._device_manager,
            hub_manager=self._hub_manager,
            compile_model=compile,
        )

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=cast(Literal["none", "all", "lora_only"], lora_bias),
            task_type=cast(Literal["CAUSAL_LM"], lora_task_type),
            target_modules=lora_target_modules,
        )

        self._tokenizer = self.checkpoint_manager.load_tokenizer(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            chat_template=chat_template,
        )

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
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def device_manager(self) -> DeviceManager:
        return self._device_manager

    @property
    def hub_manager(self) -> HubManager:
        return self._hub_manager

    def train(
        self,
        data_manager: Any,
        max_epochs: Optional[int] = None,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """
        Train the model using the configured trainer.

        Args:
            data_manager: Data manager for training data.
            max_epochs: Optional number of epochs to train for.
            wandb_manager: Optional WandbManager instance for logging.
            continue_training: Whether to continue from checkpoint.
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

    def load(
        self,
        init_new: bool = False,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
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
        self._models = self._load_models(init_new=init_new, epoch=epoch)
        if epoch is not None:
            self.epochs_per_model = [epoch] * self._num_models

    def _load_models(
        self, init_new: bool = False, epoch: Optional[int] = None
    ) -> list[PeftModel]:
        models = []
        logger.info(f"Loading {self._num_models} models...")

        for model_idx in range(self._num_models):
            models.append(
                self.checkpoint_manager.load_model(
                    model_id=self.model_id,
                    model_name=self.get_name(model_idx=model_idx),
                    model_idx=model_idx,
                    lora_config=self.lora_config,
                    init_new=init_new,
                    epoch=epoch,
                )
            )
        return models

    def _check_models_loaded(self) -> None:
        """Check if models are loaded."""
        if self._models is None:
            raise RuntimeError(
                "Models are not loaded. Call model.load() before using the model."
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

    def save(self) -> None:
        """Save all ensemble models to Hub."""
        for i in range(len(self.models)):
            self._push_model(i)

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch for a model."""
        if model_idx is not None:
            self.epochs_per_model[model_idx] = epoch
        else:
            self.epochs_per_model = [epoch] * self._num_models

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch for a model."""
        return self.epochs_per_model[model_idx] or 0

    def _push_model(self, model_idx: int, epochs: Optional[int] = None) -> None:
        """
        Push single model to hub.
        """
        self.checkpoint_manager.push_model(
            model=self.models[model_idx],
            model_name=self.get_name(model_idx=model_idx),
            tokenizer=self.tokenizer,
            epochs=epochs,
        )

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
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
            submodel_name = self.get_name(model_idx=model_idx)
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

    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-a{self.alpha}-b{self.beta}-L{self._num_models}"
        if model_idx is not None:
            repo_name = f"{repo_name}-l{model_idx}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def _get_base_model_name(self) -> str:
        return self.model_id.rsplit("/", 1)[-1]

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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

        # Calculate metrics
        with torch.no_grad():
            margins = pi_log_ratio - ref_log_ratio
            accuracy = (margins > alpha_offset).float().mean()
            metrics = {
                "loss": loss.item(),
                "rewards/chosen": (self.beta * (lprobs_chosen - lprobs_chosen_ref))
                .mean()
                .item(),
                "rewards/rejected": (self.beta * (lprobs_reject - lprobs_reject_ref))
                .mean()
                .item(),
                "rewards/margins": (self.beta * margins).mean().item(),
                "accuracy": accuracy.item(),
            }

        return loss, metrics

    def _predict_submodel(
        self,
        model: PeftModel,
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
