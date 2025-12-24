"""Abstract base class for policy models."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
from peft import PeftModel
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from .base_trainer import BaseTrainer
    from .generator import Generator
    from .utils import DeviceManager, HubManager

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for policy models (DEPPO, REPPO).

    This defines the interface that Trainer, Generator, and Evaluator
    rely on. Subclasses implement loss_fn() and predict() differently.
    """

    _models: Optional[list[PeftModel]]
    _trainer: Optional["BaseTrainer"]
    generator: Optional["Generator"]

    @property
    @abstractmethod
    def num_models(self) -> int:
        """Number of models (L for ensemble, 1 for single)."""

    @property
    @abstractmethod
    def models(self) -> list[PeftModel]:
        """List of loaded models. Raises if not loaded."""

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

    @abstractmethod
    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute loss for training.

        Args:
            batch: Training batch with input tensors.
            model: The specific submodel to compute loss for.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, *metrics) where metrics are logged.
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
    def load(self, init_new: bool = False, epoch: Optional[int] = None) -> None:
        """Load models into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload models from memory."""

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        return self.tokenizer

    @abstractmethod
    def get_name(self, epoch: Optional[int] = None) -> str:
        """Get model name for identification."""

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
