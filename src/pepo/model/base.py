"""Abstract base class for policy models."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
from peft import PeftModel
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from ..generator import Generator
    from ..loader import CheckpointManager
    from ..trainer import BaseTrainer
    from ..utils import DeviceManager, HubManager

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for policy models (DEPPO, REPPO).

    This defines the interface that Trainer, Generator, and Evaluator
    rely on. Subclasses implement loss_fn() and predict() differently.
    """

    _models: Optional[list[PeftModel]]
    _trainer: Optional["BaseTrainer"]
    _checkpoint_manager: "CheckpointManager"
    generator: Optional["Generator"]

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        """Tokenizer for the model."""

    @property
    @abstractmethod
    def device_manager(self) -> "DeviceManager":
        """Device manager for GPU allocation."""

    @property
    @abstractmethod
    def hub_manager(self) -> "HubManager":
        """Hub manager for model storage."""

    @property
    def checkpoint_manager(self) -> "CheckpointManager":
        """Checkpoint manager for saving/loading models."""
        return self._checkpoint_manager

    @abstractmethod
    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute loss for training.

        Args:
            batch: Training batch with input tensors.
            model: The specific submodel to compute loss for.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, metrics_dict) where metrics_dict contains scalars to be
            logged.
        """

    @abstractmethod
    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Inference prediction.

        For ensemble models, this handles parallel execution and aggregation.
        For single models, this is a simple forward pass.

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
    def load_from_epoch(self, epoch: int) -> None:
        """Load model from epoch."""

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
    def load(self, init_new: bool = False, epoch: Optional[int] = None) -> None:
        """Load models into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload models from memory."""

    def get_tokenizer(self) -> AutoTokenizer:
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

    @abstractmethod
    def _get_base_model_name(self) -> str:
        """Get the base model name (e.g. from model_id)."""

    def init_trainer(self) -> None:
        """Initialize the trainer if it's a partial."""
        if isinstance(self._trainer, functools.partial):
            self._trainer = self._trainer()

    @abstractmethod
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
            continue_training: Whether to continue from checkpoint.
        """

    @property
    def num_models(self) -> int:
        """Number of models (L for ensemble, 1 for single)."""
        if self._models is None:
            raise ValueError("Models not loaded. Call load() first.")
        return len(self.models)

    @property
    def models(self) -> list[PeftModel]:
        """List of loaded models. Raises if not loaded."""
        if self._models is None:
            raise ValueError("Models not loaded. Call load() first.")
        return self._models

    def generate_responses(
        self,
        prompts: list[str],
        apply_chat_template: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for a list of prompts.

        Args:
            prompts: List of prompt strings.
            apply_chat_template: Whether to apply chat template.

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
        )
