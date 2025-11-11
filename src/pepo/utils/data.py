import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .general import set_seed
from .logger import Logger


class DataCollator:
    """Tokenizes and pads sequences on-the-fly using fast tokenizer optimization."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prompt_texts = [f["prompt_text"] for f in features]
        chosen_texts = [f["chosen_text"] for f in features]
        rejected_texts = [f["rejected_text"] for f in features]

        prompt_encoded = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=self.max_prompt_length is not None,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        chosen_encoded = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected_encoded = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_encoded["input_ids"],
            "chosen_input_ids": chosen_encoded["input_ids"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "prompt_attention_mask": prompt_encoded["attention_mask"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_attention_mask": rejected_encoded["attention_mask"],
            "prompt_len": torch.tensor(
                [f["prompt_len"] for f in features], dtype=torch.long
            ),
        }


class DataManager:
    """Manages dataset loading, preprocessing, and splitting for PEPO ensemble training."""

    def __init__(
        self,
        dataset_id: str,
        split: str,
        train_split: float,
        eval_split: float,
        seed: int,
        n_splits: int,
        max_length: Optional[int],
        max_prompt_length: Optional[int],
        tokenizer: AutoTokenizer,
        num_proc: Optional[int] = None,
        cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
        dataloader_persistent_workers: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            dataset_id: HuggingFace dataset ID.
            split: Dataset split to use.
            train_split: Fraction of data for training (must sum to 1 with eval_split).
            eval_split: Fraction of data for evaluation.
            seed: Random seed for splitting.
            n_splits: Number of train splits (one per model in ensemble).
            max_length: Maximum sequence length for responses. None means no limit.
            max_prompt_length: Maximum sequence length for prompts. None means no limit.
            tokenizer: Tokenizer with chat template configured.
            num_proc: Number of processes for preprocessing. None uses all CPUs.
            cache_dir: Directory to cache preprocessed datasets. None means no caching.
            force_recompute: If True, skip loading from cache and recompute the dataset.
            dataloader_num_workers: Number of worker processes for data loading (0 = main thread only).
            dataloader_pin_memory: Pin memory for faster CPU->GPU transfer.
            dataloader_persistent_workers: Keep workers alive between epochs.
            logger: Optional logger instance.
        """
        self.logger = logger
        self.n_splits = n_splits
        self.seed = seed
        self.dataset_id = dataset_id
        self.split = split

        if train_split + eval_split != 1:
            raise ValueError("train_split and eval_split must sum to 1")
        self.train_split = train_split
        self.eval_split = eval_split

        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute

        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers

        self.cache_path = self._get_cache_path()
        self._initialize_dataset()

    def _get_cache_path(self) -> Optional[str]:
        if self.cache_dir is None:
            return None

        tokenizer_name = getattr(
            self.tokenizer, "name_or_path", str(type(self.tokenizer).__name__)
        )
        chat_template = getattr(self.tokenizer, "chat_template", None)
        chat_template_hash = (
            hashlib.md5(str(chat_template).encode()).hexdigest()
            if chat_template
            else "none"
        )

        cache_params = {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "seed": self.seed,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "tokenizer": tokenizer_name,
            "pad_token_id": self.tokenizer.pad_token_id,
            "chat_template": chat_template_hash,
        }

        cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()
        cache_path = Path(self.cache_dir) / f"preprocessed_{cache_key}"
        return str(cache_path)

    def _load_cache(self) -> Optional[Dataset]:
        if self.force_recompute:
            return None

        cache_exists = self.cache_path is not None and os.path.exists(self.cache_path)
        if not self.cache_path or not cache_exists:
            return None

        if self.logger:
            self.logger.info(
                f"Loading preprocessed dataset from cache: {self.cache_path}"
            )

        try:
            dataset = Dataset.load_from_disk(self.cache_path)
            if self.logger:
                self.logger.info(
                    f"Successfully loaded from cache - {len(dataset)} examples"
                )
            return dataset
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load from cache: {e}. Reprocessing...")
            return None

    def _save_cache(self, dataset: Dataset) -> None:
        if not self.cache_path:
            return

        if self.force_recompute and os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)

        if self.logger:
            self.logger.info(f"Saving preprocessed dataset to cache: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)
        dataset.save_to_disk(self.cache_path)

    def _load(self) -> Dataset:
        raw_dataset = load_dataset(path=self.dataset_id, split=self.split)
        if self.logger:
            self.logger.info(f"Loaded dataset with {len(raw_dataset)} examples")
        return raw_dataset

    def _ensure_message_list(self, messages, is_prompt: bool = False):
        """
        Validate and convert messages to list format.
        Matches logic from pepo_launcher.py ensure_message_list.

        Returns:
            Valid list of message dicts, or None if invalid.
        """
        if isinstance(messages, list) and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in messages
        ):
            return messages
        elif isinstance(messages, str):
            if is_prompt:
                return [{"role": "user", "content": messages}]
            else:
                return None
        else:
            return None

    def _process_helper(self, examples):
        """
        Apply chat templates and filter invalid/long examples.
        Stores formatted strings instead of tokenized sequences.
        Tokenization happens in the collator for better performance.
        """
        processed = {
            "prompt_text": [],
            "chosen_text": [],
            "rejected_text": [],
            "prompt_len": [],
        }

        for i in range(len(examples["prompt"])):
            current_prompt_messages = examples["prompt"][i]
            current_chosen_messages = examples["chosen"][i]
            current_rejected_messages = examples["rejected"][i]

            current_prompt_messages = self._ensure_message_list(
                current_prompt_messages, is_prompt=True
            )
            current_chosen_messages = self._ensure_message_list(current_chosen_messages)
            current_rejected_messages = self._ensure_message_list(
                current_rejected_messages
            )

            if (
                current_prompt_messages is None
                or current_chosen_messages is None
                or current_rejected_messages is None
            ):
                continue

            prompt_with_assistant_turn = current_prompt_messages + [
                {"role": "assistant", "content": ""}
            ]

            prompt_str = self.tokenizer.apply_chat_template(
                prompt_with_assistant_turn, tokenize=False, add_generation_prompt=True
            )

            chosen_str = self.tokenizer.apply_chat_template(
                current_chosen_messages, tokenize=False
            )
            rejected_str = self.tokenizer.apply_chat_template(
                current_rejected_messages, tokenize=False
            )

            # Tokenize just to check length for filtering
            # Use add_special_tokens=True to get accurate length (matches actual tokenization)
            prompt_encoded = self.tokenizer(
                prompt_str,
                truncation=False,
            )
            chosen_encoded = self.tokenizer(
                chosen_str,
                truncation=False,
            )
            rejected_encoded = self.tokenizer(
                rejected_str,
                truncation=False,
            )

            if (
                (
                    self.max_prompt_length is not None
                    and len(prompt_encoded["input_ids"]) > self.max_prompt_length
                )
                or (
                    self.max_length is not None
                    and len(chosen_encoded["input_ids"]) > self.max_length
                )
                or (
                    self.max_length is not None
                    and len(rejected_encoded["input_ids"]) > self.max_length
                )
            ):
                continue

            processed["prompt_text"].append(prompt_str)
            processed["chosen_text"].append(chosen_str)
            processed["rejected_text"].append(rejected_str)
            processed["prompt_len"].append(len(prompt_encoded["input_ids"]))

        return processed

    def _process(self, dataset: Dataset) -> Dataset:
        if self.logger:
            self.logger.info(
                "Preprocessing dataset (applying chat template and filtering)..."
            )

        original_size = len(dataset)

        dataset_preprocessed = dataset.map(
            self._process_helper,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_proc,
            desc="Preprocessing dataset",
        )

        if self.logger:
            self.logger.info(
                f"Dataset: {original_size} -> {len(dataset_preprocessed)} examples "
                f"(filtered {original_size - len(dataset_preprocessed)})"
            )

        return dataset_preprocessed

    def _split(self, dataset: Dataset) -> None:
        """
        Split preprocessed dataset into train/eval, then split train into n_splits.
        Format conversion is handled by the DataCollator.
        """
        dataset_split = dataset.train_test_split(
            test_size=self.eval_split, seed=self.seed
        )
        train_dataset_preprocessed = dataset_split["train"]
        eval_dataset_preprocessed = dataset_split["test"]

        if self.logger:
            self.logger.info(
                f"Train: {len(train_dataset_preprocessed)} examples, "
                f"Eval: {len(eval_dataset_preprocessed)} examples"
            )

        indices = np.arange(len(train_dataset_preprocessed))
        set_seed(self.seed)
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, self.n_splits)

        self.train_datasets = {}
        for model_idx, indices in enumerate(split_indices):
            self.train_datasets[model_idx] = train_dataset_preprocessed.select(indices)
            if self.logger:
                self.logger.info(
                    f"Train split {model_idx} has {len(self.train_datasets[model_idx])} examples"
                )

        self.eval_dataset = eval_dataset_preprocessed

        if self.logger:
            self.logger.info("Dataset preprocessing and splitting complete.")

    def _initialize_dataset(self) -> None:
        dataset_preprocessed = self._load_cache()

        if dataset_preprocessed is None:
            if self.logger:
                if self.force_recompute:
                    self.logger.info("Recomputing dataset (force_recompute=True)...")
                else:
                    self.logger.info(
                        f"No cached dataset found at {self.cache_path}. Creating new dataset..."
                    )
            dataset_raw = self._load()
            dataset_preprocessed = self._process(dataset_raw)
            self._save_cache(dataset_preprocessed)

        self._split(dataset_preprocessed)

    def get_dataloader(
        self,
        model_idx: int,
        partition: str,
        batch_size: int,
    ) -> DataLoader:
        """
        Get DataLoader for a specific model and partition.

        Args:
            model_idx: Index of the model in the ensemble (0 to n_splits-1).
            partition: Either "train" or "eval".
            batch_size: Batch size for the DataLoader.

        Returns:
            DataLoader for the specified model and partition.
        """
        if model_idx < 0 or model_idx >= self.n_splits:
            raise ValueError(
                f"Model index {model_idx} out of range [0, {self.n_splits - 1}]"
            )

        if partition == "train":
            dataset = self.train_datasets[model_idx]
            shuffle = True
        elif partition == "eval":
            dataset = self.eval_dataset
            shuffle = False
        else:
            raise ValueError(f"Partition must be 'train' or 'eval', got '{partition}'")

        collator = DataCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=(
                self.dataloader_persistent_workers
                if self.dataloader_num_workers > 0
                else False
            ),
        )
