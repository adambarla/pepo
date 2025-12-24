"""PEPO module"""

__version__ = "0.1.0"

import warnings

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

from .base_model import BaseModel  # noqa: E402
from .base_trainer import BaseTrainer  # noqa: E402
from .evaluator import AlpacaEvalEvaluator, BaseEvaluator  # noqa: E402
from .model import DEPPOModel  # noqa: E402
from .trainer import DEPPOTrainer  # noqa: E402

__all__ = [
    "BaseModel",
    "BaseTrainer",
    "DEPPOModel",
    "DEPPOTrainer",
    "BaseEvaluator",
    "AlpacaEvalEvaluator",
]
