from typing import Protocol

from datasets import Dataset
from transformers import AutoTokenizer


class DataProcessor(Protocol):
    """Protocol for dataset processors. Defines how to preprocess raw datasets."""

    def process(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        """Process dataset and return preprocessed version."""
        ...
