import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional

from datasets import load_dataset

from ..model import PEPOModel
from ..utils import sanitize_filename

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(
        self,
        dataset_id: str,
        dataset_split: str,
        output_dir: str,
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Initialize base evaluator.

        Args:
            dataset_id: HuggingFace dataset ID.
            dataset_split: Dataset split to use.
            output_dir: Directory to save outputs.
        """
        self.dataset_id = dataset_id
        self.dataset_split = dataset_split
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

    @abstractmethod
    def evaluate(
        self,
        model: PEPOModel,
        epoch: Optional[int] = None,
        ref_model: Optional[PEPOModel] = None,
        ref_epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Evaluate responses.

        Args:
            model: PEPOModel instance.
            epoch: Epoch of the model.
            ref_model: Optional reference PEPOModel instance.
            ref_epoch: Epoch of the reference model.
        """
        pass

    def _load_dataset(self) -> Any:
        """
        Load dataset from HuggingFace.
        """
        dataset = load_dataset(
            self.dataset_id,
            split=self.dataset_split,
            trust_remote_code=True,
        )
        if self.num_samples is not None and self.num_samples > 0:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        count = self.num_samples if self.num_samples else len(dataset)
        logger.info(f"Loaded {count} samples from {self.dataset_id}")
        return dataset

    def _get_filename(
        self,
        model: PEPOModel,
        epoch: Optional[int] = None,
    ) -> str:
        """
        Generate file names based on model name and generation configuration.

        Args:
            model: PEPOModel instance.
            epoch: Epoch of the model.
            **kwargs: Additional generation parameters.

        Returns:
            Tuple of (responses_filename, results_filename, leaderboard_filename).
        """
        model_name = sanitize_filename(model.get_name(epoch=epoch))
        parts = [model_name]
        if self.num_samples:
            parts.append(f"ns{self.num_samples}")
        if model.generator is not None:
            generator_name = model.generator.get_name()
            parts.append(generator_name)
        base_name = "_".join(parts)
        return base_name

    def _get_folder(
        self,
        ref_model: Optional[PEPOModel] = None,
        ref_epoch: Optional[int] = None,
    ) -> Path:
        """
        Generate file paths based on model name and generation configuration.
        """
        # outpus [/ref_model_name] / file_name
        folder = self.output_dir
        if ref_model is not None:
            folder = folder / ref_model.get_name(epoch=ref_epoch)
        return folder

    def _get_file_paths(
        self,
        model: PEPOModel,
        epoch: Optional[int] = None,
        ref_model: Optional[PEPOModel] = None,
        ref_epoch: Optional[int] = None,
        type: Literal["responses", "results", "leaderboard"] = "responses",
    ) -> Path:
        if type not in ["responses", "results", "leaderboard"]:
            raise ValueError(f"Invalid type: {type}")
        base_name = self._get_filename(model=model, epoch=epoch)
        folder = self._get_folder(ref_model=ref_model, ref_epoch=ref_epoch)
        path = folder / Path(base_name + f"_{type}.json")
        return path
