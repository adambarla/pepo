"""PEPO module"""

__version__ = "0.1.0"

import warnings

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

from .evaluator import AlpacaEvalEvaluator, BaseEvaluator  # noqa: E402
from .model import BaseModel, DEPPOModel  # noqa: E402
from .trainer import BaseTrainer, DEPPOTrainer  # noqa: E402

__all__ = [
    "BaseModel",
    "DEPPOModel",
    "BaseTrainer",
    "DEPPOTrainer",
    "BaseEvaluator",
    "AlpacaEvalEvaluator",
]
