"""Trainer module - provides BaseTrainer and DEPPOTrainer."""

from .base import BaseTrainer
from .deppo import DEPPOTrainer

__all__ = ["BaseTrainer", "DEPPOTrainer"]
