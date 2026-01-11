import copy
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from ..utils import DeviceManager

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ..utils import get_device_manager, get_hub_manager
from .annotators import BaseAnnotator
from .collators.base import DataCollator
from .processors.base import DataProcessor
from .sampler import LengthBasedBatchSampler

logger = logging.getLogger(__name__)


class DataManager:
    """Manages dataset loading, preprocessing, and splitting for PEPO."""

    def __init__(
        self,
        dataset_id: str,
        train_split_name: str,
        eval_split_name: str,
        seed: int,
        n_splits: int,
        tokenizer: PreTrainedTokenizerBase,
        processor: Optional[DataProcessor] = None,
        collator: Optional[DataCollator] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
        shuffle_train: bool = True,
        annotators: Optional[list[BaseAnnotator]] = None,
        inference_batch_size: int = 8,
        device_manager: Optional["DeviceManager"] = None,
        force_recompute: bool = False,
    ):
        self.dataset_id = dataset_id
        self.train_split_name = train_split_name
        self.eval_split_name = eval_split_name
        self.seed = seed
        self.n_splits = n_splits
        self.tokenizer = tokenizer
        self.processor = processor
        self.collator = collator
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.shuffle_train = shuffle_train
        self.annotators = annotators or []
        self.inference_batch_size = inference_batch_size
        self.device_manager = device_manager or get_device_manager()
        self.force_recompute = force_recompute
        self._hub_manager = get_hub_manager()

        self._initialize_dataset()

    def get_processed_name(self) -> str:
        """Generate unique name for processed dataset based on config."""

        # Sort annotator signatures for consistent hashing (commutative)
        annotator_sigs = [
            str(sorted(a.get_signature().items())) for a in self.annotators
        ]
        annotator_sigs.sort()

        hash_params = {
            "template": self.tokenizer.chat_template or "",
            "train_split": self.train_split_name,
            "eval_split": self.eval_split_name,
            "max_len": self.max_length,
            "max_prompt_len": self.max_prompt_length,
            "annotators": annotator_sigs,
        }
        param_hash = hashlib.md5(str(sorted(hash_params.items())).encode()).hexdigest()[
            :8
        ]

        proc_name = type(self.processor).__name__ if self.processor else "none"
        if proc_name.endswith("Processor"):
            proc_name = proc_name.replace("Processor", "")

        # Also sort names for the string representation
        annotator_names = sorted([type(a).__name__ for a in self.annotators])
        annotator_names = [
            n.replace("Annotator", "") if n.endswith("Annotator") else n
            for n in annotator_names
        ]
        annotator_str = "-".join(annotator_names) if annotator_names else "none"

        # Use only the last part of dataset_id (after /) for brevity
        short_dataset_id = self.dataset_id.rsplit("/", 1)[-1]

        # Omit processor name if it matches dataset pattern (redundant)
        if proc_name.lower() in short_dataset_id.lower():
            full_name = f"{short_dataset_id}-{annotator_str}-{param_hash}"
        else:
            full_name = f"{short_dataset_id}-{proc_name}-{annotator_str}-{param_hash}"

        return full_name

    def _initialize_dataset(self) -> None:
        processed_name = self.get_processed_name()

        # Check if processed dataset exists on Hub
        # Check if processed dataset exists on Hub
        if not self.force_recompute and self._hub_manager.dataset_exists(
            processed_name
        ):
            logger.info(f"Loading processed dataset from Hub: {processed_name}")
            repo_id = self._hub_manager.get_repo_id(processed_name)
            train_data = cast(
                Dataset, load_dataset(repo_id, split=self.train_split_name)
            )
            eval_data = cast(Dataset, load_dataset(repo_id, split=self.eval_split_name))
        else:
            # Load raw and process
            train_raw = cast(
                Dataset, load_dataset(self.dataset_id, split=self.train_split_name)
            )
            eval_raw = cast(
                Dataset, load_dataset(self.dataset_id, split=self.eval_split_name)
            )
            logger.info(
                f"Loaded {self.dataset_id}: "
                f"train={len(train_raw)}, eval={len(eval_raw)}"
            )

            if self.processor:
                train_data = self.processor.process(train_raw, self.tokenizer)
                eval_data = self.processor.process(eval_raw, self.tokenizer)
            else:
                train_data, eval_data = train_raw, eval_raw

            for annotator in self.annotators:
                logger.info(f"Applying annotator: {annotator.__class__.__name__}")
                train_data = annotator.annotate(
                    train_data,
                    tokenizer=self.tokenizer,
                    device_manager=self.device_manager,
                    force=self.force_recompute,
                )
                eval_data = annotator.annotate(
                    eval_data,
                    tokenizer=self.tokenizer,
                    device_manager=self.device_manager,
                    force=self.force_recompute,
                )

            # Push to Hub
            ds_dict = DatasetDict(
                {self.train_split_name: train_data, self.eval_split_name: eval_data}
            )
            self._hub_manager.push_dataset(ds_dict, processed_name)

        self._split_train(train_data)
        self.eval_dataset = self._sort_by_length(eval_data)

    def _split_train(self, dataset: Dataset) -> None:
        """Split train into n_splits for ensemble models."""
        if self.n_splits is None or self.n_splits <= 1:
            self.train_datasets = {0: self._sort_by_length(dataset)}
            return

        np.random.seed(self.seed)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        shuffled = dataset.select(indices)

        split_indices = np.array_split(np.arange(len(shuffled)), self.n_splits)
        self.train_datasets = {
            i: self._sort_by_length(shuffled.select(idx))
            for i, idx in enumerate(split_indices)
        }
        logger.info(f"Split train into {self.n_splits} splits")

    def _sort_by_length(self, dataset: Dataset) -> Dataset:
        if "chosen_text" not in dataset.column_names:
            return dataset
        lengths = [
            max(len(dataset[i]["chosen_text"]), len(dataset[i]["rejected_text"]))
            for i in range(len(dataset))
        ]
        sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        return dataset.select(sorted_indices)

    def set_num_splits(self, n_splits: int) -> None:
        """Update number of splits and re-partition train dataset."""
        if n_splits == self.n_splits:
            return

        self.n_splits = n_splits
        # Resplit using the current merged dataset
        self._split_train(self.merged_train_dataset)

    @property
    def merged_train_dataset(self) -> Dataset:
        from datasets import concatenate_datasets

        return concatenate_datasets(list(self.train_datasets.values()))

    def add_annotator(self, annotator: BaseAnnotator) -> None:
        """Add an annotator, update dataset (cache/compute), and re-split."""
        self.annotators.append(annotator)
        new_name = self.get_processed_name()

        # Check cache
        if (
            not self.force_recompute
            and not annotator.force
            and self._hub_manager.dataset_exists(new_name)
        ):
            logger.info(f"Loading annotated dataset from Hub: {new_name}")
            repo_id = self._hub_manager.get_repo_id(new_name)
            # Load and update internal state
            self._split_train(
                cast(Dataset, load_dataset(repo_id, split=self.train_split_name))
            )
            eval_ds = cast(Dataset, load_dataset(repo_id, split=self.eval_split_name))
            self.eval_dataset = self._sort_by_length(eval_ds)
            return

        # Compute
        logger.info(f"applying additional annotator: {type(annotator).__name__}...")

        # Use current state as base
        train_data = self.merged_train_dataset
        eval_data = self.eval_dataset

        train_data = annotator.annotate(
            train_data,
            tokenizer=self.tokenizer,
            device_manager=self.device_manager,
            force=self.force_recompute,
        )
        eval_data = annotator.annotate(
            eval_data,
            tokenizer=self.tokenizer,
            device_manager=self.device_manager,
            force=self.force_recompute,
        )

        # Push to Hub
        logger.info(f"Pushing updated dataset to Hub: {new_name}")
        ds_dict = DatasetDict(
            {self.train_split_name: train_data, self.eval_split_name: eval_data}
        )
        self._hub_manager.push_dataset(ds_dict, new_name)

        # Update internal state (re-split)
        self._split_train(train_data)
        self.eval_dataset = self._sort_by_length(eval_data)

    def get_dataloader(
        self,
        model_idx: int,
        partition: str,
        batch_size: int,
        shuffle: Optional[bool] = None,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        if partition == "train":
            dataset = self.train_datasets[model_idx]
            do_shuffle = self.shuffle_train if shuffle is None else shuffle
        else:
            dataset = self.eval_dataset
            do_shuffle = False if shuffle is None else shuffle

        if self.collator:
            collator = copy.deepcopy(self.collator)
            collator.set_tokenizer(copy.deepcopy(self.tokenizer))
        else:
            collator = DataCollator(
                tokenizer=copy.deepcopy(self.tokenizer),
                max_length=self.max_length,
                max_prompt_length=self.max_prompt_length,
            )
        sampler = LengthBasedBatchSampler(
            len(dataset), batch_size, shuffle=do_shuffle, seed=self.seed
        )
        return DataLoader(
            cast(torch.utils.data.Dataset[Any], dataset),
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
        )

    def get_name(self) -> str:
        return self.dataset_id.replace("/", "_")
