from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..generate import Generator
from ..model import PEPOModel
from ..utils import Logger


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(
        self,
        model: PEPOModel,
        dataset_id: str,
        dataset_split: str,
        instruction_key: str,
        output_dir: str,
        generator: Generator,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize base evaluator.

        Args:
            dataset_id: HuggingFace dataset ID.
            dataset_split: Dataset split to use.
            instruction_key: Key to extract instruction from dataset items.
            output_dir: Directory to save outputs.
            responses_file: Filename for generated responses.
            results_file: Filename for evaluation results.
            logger: Optional logger instance.
        """
        self.model = model
        self.generator = generator
        self.dataset_id = dataset_id
        self.dataset_split = dataset_split
        self.instruction_key = instruction_key
        self.output_dir = Path(output_dir)
        self.logger = logger

    @abstractmethod
    def generate_responses(self, **kwargs):
        """
        Generate responses for the evaluation dataset.

        Args:
            **kwargs: Generation configuration parameters.

        Returns:
            Path to the responses file.
        """
        pass

    @abstractmethod
    def evaluate(self, responses_file: Path) -> Path:
        """
        Evaluate responses.

        Args:
            responses_file: Path to the responses file.

        Returns:
            Path to the results file.
        """
        pass

    @abstractmethod
    def responses_exist(self, **kwargs: Any) -> bool:
        """
        Check if responses file already exists.

        Args:
            **kwargs: Generation configuration parameters.

        Returns:
            True if responses file exists, False otherwise.
        """
        pass
