"""PEPO module"""

__version__ = "0.1.0"

import warnings

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

from .evaluator import (  # noqa: E402
    AlpacaEvalEvaluator,
    BaseEvaluator,
    MTBenchEvaluator,
)
from .model import BaseModel, DEPPOModel  # noqa: E402
from .trainer import BaseTrainer, EnsembleTrainer, SingleModelTrainer  # noqa: E402

__all__ = [
    "BaseModel",
    "DEPPOModel",
    "BaseTrainer",
    "EnsembleTrainer",
    "SingleModelTrainer",
    "BaseEvaluator",
    "AlpacaEvalEvaluator",
    "MTBenchEvaluator",
]
