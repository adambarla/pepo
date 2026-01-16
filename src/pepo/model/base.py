"""Abstract base class for policy models."""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from ..generator import Generator
    from ..loader import CheckpointManager
    from ..trainer import BaseTrainer
    from ..utils import DeviceManager, HubManager

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for policy models (DEPPO, REPPO)."""

    def __init__(
        self,
        model_id: str,
        device_manager: "DeviceManager",
        hub_manager: "HubManager",
        checkpoint_manager: "CheckpointManager",
        tokenizer: PreTrainedTokenizerBase,
        trainer: Optional["BaseTrainer"] = None,
        generator: Optional["Generator"] = None,
    ) -> None:
        """Initialize base model attributes.

        Args:
            model_id: HuggingFace model ID.
            device_manager: Device manager for GPU allocation.
            hub_manager: Hub manager for model storage.
            checkpoint_manager: Checkpoint manager for saving/loading.
            tokenizer: Tokenizer for the model.
            trainer: Optional trainer instance.
            generator: Optional generator instance.
        """
        self.model_id = model_id
        self._device_manager = device_manager
        self._hub_manager = hub_manager
        self._checkpoint_manager = checkpoint_manager
        self._tokenizer = tokenizer
        self._trainer = trainer
        self.generator = generator

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer for the model."""
        return self._tokenizer

    @property
    def device_manager(self) -> "DeviceManager":
        """Device manager for GPU allocation."""
        return self._device_manager

    @property
    def hub_manager(self) -> "HubManager":
        """Hub manager for model storage."""
        return self._hub_manager

    @property
    def checkpoint_manager(self) -> "CheckpointManager":
        """Checkpoint manager for saving/loading models."""
        return self._checkpoint_manager

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if models are loaded."""

    @abstractmethod
    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss for training.

        Args:
            batch: Training batch with input tensors.
            model: The submodel to compute loss for.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, metrics_dict).
        """

    @abstractmethod
    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> Any:
        """Inference prediction.

        Args:
            device_input_ids: Input IDs per device/model.
            device_attention_masks: Attention masks per device/model.

        Returns:
            Aggregated predictions (e.g., min log probs for DEPPO).
        """

    @abstractmethod
    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if model can be loaded from epoch."""

    @abstractmethod
    def save(self) -> None:
        """Save all models to storage."""

    @abstractmethod
    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch. For ensembles, model_idx specifies which model."""

    @abstractmethod
    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch. For ensembles, model_idx specifies which model."""

    def find_latest_epoch(self, max_epoch: int) -> Optional[int]:
        """
        Find the latest epoch where all submodels have checkpoints.

        Args:
            max_epoch: Maximum epoch to check from (checks backwards).

        Returns:
            The latest epoch where all submodels exist, or None if none found.
        """
        for epoch in range(max_epoch, -1, -1):
            if self.can_load_from_epoch(epoch):
                logger.info(f"Found latest common epoch {epoch} for model")
                return epoch

        logger.warning(
            f"No common epoch found (checked from epoch {max_epoch} down to 0)"
        )
        return None

    @abstractmethod
    def load(
        self, init_new: bool = False, epoch: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Load models into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload models from memory."""

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer."""
        return self.tokenizer

    @abstractmethod
    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Get model name for identification.

        Args:
            epoch: Optional epoch number to include in name.
            model_idx: Optional model index for ensemble submodels.
            **kwargs: Additional optional arguments.
        """

    def _get_base_model_name(self) -> str:
        """Get the base model name (e.g. from model_id)."""
        return self.model_id.rsplit("/", 1)[-1]

    def init_trainer(self) -> None:
        """Initialize the trainer if it's a partial."""
        if isinstance(self._trainer, functools.partial):
            self._trainer = self._trainer()

    @property
    def trainer(self) -> Optional["BaseTrainer"]:
        """Get the trainer instance."""
        if isinstance(self._trainer, functools.partial):
            self.init_trainer()
        return self._trainer

    @trainer.setter
    def trainer(self, value: Optional["BaseTrainer"]) -> None:
        """Set the trainer instance."""
        self._trainer = value

    @abstractmethod
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
                Defaults to trainer config.
            wandb_manager: Optional WandbManager instance for logging.
            continue_training: Whether to continue from checkpoint.
        """

    def generate_responses(
        self,
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> list[dict[str, Any]]:
        """Generate responses for a list of prompts.

        Args:
            prompts: List of prompt strings or histories.
            apply_chat_template: Whether to apply chat template.
            token_callback: Optional callback for streaming tokens.

        Returns:
            List of dicts with 'prompt' and 'output' keys.
        """
        if self.generator is None:
            raise ValueError(
                "Generator not set on model. Set model.generator before "
                "calling generate_responses()."
            )
        return self.generator.generate_responses(
            model=self,
            prompts=prompts,
            apply_chat_template=apply_chat_template,
            token_callback=token_callback,
        )


class EnsembleModel(BaseModel):
    """Base class for ensemble models with multiple submodels (DEPPO, REPPOReward)."""

    def __init__(
        self,
        num_models: int,
        **kwargs: Any,
    ) -> None:
        """Initialize ensemble model.

        Args:
            num_models: Number of models in the ensemble.
            **kwargs: Arguments passed to BaseModel.__init__.
        """
        super().__init__(**kwargs)
        self._num_models = num_models
        self._models: Optional[list[PeftModel]] = None
        self.epochs_per_model: list[Optional[int]] = [0] * num_models

    @property
    def num_models(self) -> int:
        """Number of models in the ensemble."""
        return self._num_models

    @property
    def models(self) -> list[PeftModel]:
        """List of loaded models. Raises if not loaded."""
        if not self.is_loaded():
            raise RuntimeError(
                "Models not loaded. Call load() before accessing models."
            )
        assert self._models is not None  # for type narrowing
        return self._models

    @models.setter
    def models(self, value: list[PeftModel]) -> None:
        """Set the models list."""
        self._models = value

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models is not None

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch for a model."""
        if model_idx is not None:
            self.epochs_per_model[model_idx] = epoch
        else:
            self.epochs_per_model = [epoch] * self._num_models

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch for a model."""
        return self.epochs_per_model[model_idx] or 0

    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if all submodels have checkpoints at the specified epoch."""
        for model_idx in range(self._num_models):
            submodel_name = self.get_name(model_idx=model_idx)
            if not self.hub_manager.model_exists(submodel_name, epoch):
                return False
        return True

    def save(self) -> None:
        """Save all ensemble models to Hub."""
        for i in range(len(self.models)):
            self.checkpoint_manager.push_model(
                model=self.models[i],
                model_name=self.get_name(model_idx=i),
                tokenizer=self.tokenizer,
            )


class SingleModel(BaseModel):
    """Base class for single-model architectures (REPPOModel policy)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize single model.

        Args:
            **kwargs: Arguments passed to BaseModel.__init__.
        """
        super().__init__(**kwargs)
        self._model: Optional[PeftModel] = None
        self._epoch: int = 0

    @property
    def model(self) -> PeftModel:
        """The loaded model. Raises if not loaded."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        assert self._model is not None  # for type narrowing
        return self._model

    @model.setter
    def model(self, value: PeftModel) -> None:
        """Set the model."""
        self._model = value

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch (model_idx ignored for single model)."""
        self._epoch = epoch

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch (model_idx ignored for single model)."""
        return self._epoch

    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if model has checkpoint at the specified epoch."""
        return self.hub_manager.model_exists(self.get_name(), epoch)
