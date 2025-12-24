import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

from ..base_model import BaseModel
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
        model: BaseModel,
        epoch: Optional[int] = None,
        ref_model: Optional[BaseModel] = None,
        ref_epoch: Optional[int] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Evaluate responses.

        Args:
            model: BaseModel instance.
            epoch: Epoch of the model.
            ref_model: Optional reference BaseModel instance.
            ref_epoch: Epoch of the reference model.
            overwrite: Whether to overwrite existing outputs.
            **kwargs: Additional arguments.
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
        model: BaseModel,
        epoch: Optional[int] = None,
    ) -> str:
        """
        Generate file names based on model name and generation configuration.

        Args:
            model: BaseModel instance.
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

    def check_generator_consistency(
        self,
        model: BaseModel,
        ref_model: Optional[BaseModel] = None,
    ) -> None:
        """
        Check if the generator is consistent between model and reference model.
        """
        if ref_model is None:
            return

        if model.generator is None:
            raise ValueError("Model does not have a generator set.")
        if ref_model.generator is None:
            raise ValueError("Reference Model does not have a generator set.")

        gen_name = model.generator.get_name()
        ref_gen_name = ref_model.generator.get_name()

        if gen_name != ref_gen_name:
            raise ValueError(
                f"Generator mismatch: Model uses '{gen_name}'"
                f" but Reference Model uses '{ref_gen_name}'. "
            )

    def _get_folder(
        self,
        ref_model: Optional[BaseModel] = None,
        ref_epoch: Optional[int] = None,
    ) -> Path:
        """
        Generate folder path based on the reference model.
        """
        folder = self.output_dir
        if ref_model is not None:
            folder = folder / self._get_filename(ref_model, epoch=ref_epoch)
        else:
            folder = folder / "default"
        return folder
