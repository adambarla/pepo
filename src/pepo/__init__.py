"""PEPO module"""

__version__ = "0.1.0"

import warnings

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

from .evaluator import AlpacaEvalEvaluator, BaseEvaluator  # noqa: E402
from .model import PEPOModel  # noqa: E402
from .trainer import Trainer  # noqa: E402

__all__ = ["PEPOModel", "BaseEvaluator", "AlpacaEvalEvaluator", "Trainer"]
