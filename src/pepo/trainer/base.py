"""Abstract base class for trainers."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from ..data import DataManager
    from ..model import BaseModel
    from ..utils import WandbManager, WandbRun

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    Defined with shared configuration fields used by subclasses.
    """

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler_name: str,
        scheduler_num_warmup_steps: int,
        wandb_config: DictConfig,
        train_batch_size: int,
        eval_batch_size: int,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        log_interval: int = 100,
        skip_eval: bool = False,
        max_batches_per_epoch: Optional[int] = None,
        training_epochs: Optional[int] = None,
        force: bool = False,
    ) -> None:
        self.optimizer_factory = optimizer
        self.scheduler_name = scheduler_name
        self.scheduler_num_warmup_steps = scheduler_num_warmup_steps
        self.wandb_config = wandb_config
        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.log_interval = log_interval
        self.skip_eval = skip_eval
        self.max_batches_per_epoch = max_batches_per_epoch
        self.training_epochs = training_epochs
        self.force = force

        self.wandb_manager: Optional["WandbManager"] = None

    @abstractmethod
    def train(
        self,
        model: "BaseModel",
        data_manager: "DataManager",
        max_epochs: Optional[int] = None,
        wandb_manager: Optional["WandbManager"] = None,
        continue_training: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Train the model using the configured trainer.

        Args:
            model: The model to train.
            data_manager: Data manager for training data.
            max_epochs: Optional number of epochs to train for.
                Defaults to trainer config.
            wandb_manager: Optional WandbManager instance for logging.
            continue_training: Whether to continue from checkpoint.
        """
        pass

    def _compute_avg_metrics(
        self, accumulated_metrics: dict[str, float], count: int
    ) -> dict[str, float]:
        """Compute averages from accumulated metrics."""
        return {k: v / count for k, v in accumulated_metrics.items()}

    def _move_to_device(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {k: v.to(device) for k, v in batch.items()}

    def _log_metrics(
        self,
        wandb_run: Optional["WandbRun"],
        metrics: dict[str, float],
        step: int,
        prefix: str = "train",
        add_avg_prefix: bool = True,
        exclude_keys: Optional[list[str]] = None,
        additional_log_items: Optional[dict[str, Any]] = None,
    ) -> None:
        """Helper to log metrics to wandb with consistent naming."""
        if wandb_run is None:
            return

        if exclude_keys is None:
            exclude_keys = []

        log_dict: dict[str, Any] = {}
        if prefix == "train":
            log_dict[f"{prefix}/step"] = step
        if additional_log_items:
            log_dict.update(additional_log_items)

        for k, v in metrics.items():
            if k in exclude_keys:
                continue

            if add_avg_prefix:
                parts = k.split("/")
                parts[-1] = f"avg_{parts[-1]}"
                new_k = "/".join(parts)
                key_name = f"{prefix}/{new_k}"
            else:
                key_name = f"{prefix}/{k}"

            log_dict[key_name] = v

        wandb_run.log(log_dict, step=step)
