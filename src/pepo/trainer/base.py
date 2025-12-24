"""Abstract base class for trainers."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..model import BaseModel
    from ..utils import DataManager, WandbManager

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    This defines the interface that BaseModel.train() relies on.
    Subclasses implement training logic for specific model types.
    """

    @abstractmethod
    def train(
        self,
        model: "BaseModel",
        data_manager: "DataManager",
        max_epochs: int,
        wandb_manager: Optional["WandbManager"] = None,
        continue_training: bool = False,
    ) -> None:
        """
        Train the model.

        Args:
            model: Model instance to train.
            data_manager: Data manager for training data.
            max_epochs: Maximum number of epochs to train for.
            wandb_manager: Optional WandbManager instance for logging.
            continue_training: Whether to continue from checkpoint.
        """
