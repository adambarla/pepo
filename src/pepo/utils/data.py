import hashlib
import os
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
    """Pads tokenized sequences. Tokenization happens during preprocessing."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        def pad_tensors(tensors, pad_value=self.pad_token_id):
            max_len = max(len(t) for t in tensors)
            padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
            for i, t in enumerate(tensors):
                if isinstance(t, torch.Tensor):
                    padded[i, : len(t)] = t.clone().detach().to(torch.long)
                else:
                    padded[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            return padded

        return {
            "prompt_input_ids": pad_tensors([f["prompt_input_ids"] for f in features]),
            "chosen_input_ids": pad_tensors([f["chosen_input_ids"] for f in features]),
            "rejected_input_ids": pad_tensors(
                [f["rejected_input_ids"] for f in features]
            ),
            "prompt_attention_mask": pad_tensors(
                [f["prompt_attention_mask"] for f in features], pad_value=0
            ),
            "chosen_attention_mask": pad_tensors(
                [f["chosen_attention_mask"] for f in features], pad_value=0
            ),
            "rejected_attention_mask": pad_tensors(
                [f["rejected_attention_mask"] for f in features], pad_value=0
            ),
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
        Tokenize examples and filter invalid ones.
        Filters out invalid message formats and sequences that are too long.
        """
        processed = {
            "prompt_input_ids": [],
            "chosen_input_ids": [],
            "rejected_input_ids": [],
            "prompt_attention_mask": [],
            "chosen_attention_mask": [],
            "rejected_attention_mask": [],
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

            prompt_encoded = self.tokenizer(
                prompt_str,
                truncation=self.max_prompt_length is not None,
                max_length=self.max_prompt_length,
            )
            chosen_encoded = self.tokenizer(
                chosen_str,
                truncation=self.max_length is not None,
                max_length=self.max_length,
            )
            rejected_encoded = self.tokenizer(
                rejected_str,
                truncation=self.max_length is not None,
                max_length=self.max_length,
            )

            if (
                (
                    self.max_prompt_length is not None
                    and len(prompt_encoded["input_ids"]) >= self.max_prompt_length
                )
                or (
                    self.max_length is not None
                    and len(chosen_encoded["input_ids"]) >= self.max_length
                )
                or (
                    self.max_length is not None
                    and len(rejected_encoded["input_ids"]) >= self.max_length
                )
            ):
                continue

            processed["prompt_input_ids"].append(prompt_encoded["input_ids"])
            processed["chosen_input_ids"].append(chosen_encoded["input_ids"])
            processed["rejected_input_ids"].append(rejected_encoded["input_ids"])
            processed["prompt_attention_mask"].append(prompt_encoded["attention_mask"])
            processed["chosen_attention_mask"].append(chosen_encoded["attention_mask"])
            processed["rejected_attention_mask"].append(
                rejected_encoded["attention_mask"]
            )
            processed["prompt_len"].append(len(prompt_encoded["input_ids"]))

        return processed

    def _process(self, dataset: Dataset) -> Dataset:
        if self.logger:
            self.logger.info(
                "Preprocessing dataset (applying chat template and tokenizing)..."
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
        Sets torch format for all datasets.
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

        eval_dataset_preprocessed.set_format(
            type="torch",
            columns=[
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "prompt_attention_mask",
                "chosen_attention_mask",
                "rejected_attention_mask",
                "prompt_len",
            ],
        )
        self.eval_dataset = eval_dataset_preprocessed

        for model_idx in range(self.n_splits):
            self.train_datasets[model_idx].set_format(
                type="torch",
                columns=[
                    "prompt_input_ids",
                    "chosen_input_ids",
                    "rejected_input_ids",
                    "prompt_attention_mask",
                    "chosen_attention_mask",
                    "rejected_attention_mask",
                    "prompt_len",
                ],
            )

        if self.logger:
            self.logger.info("Dataset preprocessing and splitting complete.")

    def _initialize_dataset(self) -> None:
        dataset_preprocessed = self._load_cache()
        if dataset_preprocessed is None:
            if self.logger:
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

        collator = DataCollator(pad_token_id=self.tokenizer.pad_token_id)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        )
