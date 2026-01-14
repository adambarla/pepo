from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset


class BaseAnnotator(ABC):
    """Abstract base class for dataset annotators."""

    def __init__(self, force: bool = False) -> None:
        self.force = force

    @abstractmethod
    def annotate(self, dataset: Dataset, **kwargs: Any) -> Dataset:
        """
        Annotate the dataset with additional columns.

        Args:
            dataset: The dataset to annotate.
            **kwargs: Additional arguments.

        Returns:
            The annotated dataset.
        """
        ...

    @abstractmethod
    def get_signature(self) -> dict[str, Any]:
        """Return parameters that uniquely identify this annotator's effect."""
        ...
