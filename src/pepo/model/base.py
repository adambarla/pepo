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

__all__ = ["BaseModel"]


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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[Any]]:
        """Single-token prediction.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            device: Device the model is on.
            past_key_values: Optional key values for caching.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (Log probs for next token (B, V) on CPU, past_key_values).
        """

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        greedy_sampling: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        token_callback: Optional[Callable[[str], None]] = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            max_new_tokens: Maximum tokens to generate.
            greedy_sampling: If True, use argmax. If False, use top-p.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling threshold.
            token_callback: Optional callback for streaming tokens.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (output_ids, output_mask) on CPU.
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
