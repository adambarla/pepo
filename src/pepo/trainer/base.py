"""Abstract base class for trainers."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from ..model import BaseModel
    from ..utils import DataManager, WandbManager

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

        self.wandb_manager: Optional["WandbManager"] = None

    @abstractmethod
    def train(
        self,
        model: "BaseModel",
        data_manager: "DataManager",
        max_epochs: Optional[int] = None,
        wandb_manager: Optional["WandbManager"] = None,
        continue_training: bool = False,
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
