from typing import Optional

import torch
from datasets import Dataset, load_dataset

from .logger import Logger


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        # Helper function to pad a list of tensors efficiently
        def pad_tensors(tensors, pad_value=self.pad_token_id):
            max_len = max(len(t) for t in tensors)
            padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
            for i, t in enumerate(tensors):
                padded[i, : len(t)] = t
            return padded

        # Extract and pad all fields
        prompt_input_ids = pad_tensors([f["prompt_input_ids"] for f in features])
        chosen_input_ids = pad_tensors([f["chosen_input_ids"] for f in features])
        rejected_input_ids = pad_tensors([f["rejected_input_ids"] for f in features])

        prompt_attention_mask = pad_tensors(
            [f["prompt_attention_mask"] for f in features], pad_value=0
        )
        chosen_attention_mask = pad_tensors(
            [f["chosen_attention_mask"] for f in features], pad_value=0
        )
        rejected_attention_mask = pad_tensors(
            [f["rejected_attention_mask"] for f in features], pad_value=0
        )

        prompt_len = torch.tensor([f["prompt_len"] for f in features], dtype=torch.long)

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_attention_mask": rejected_attention_mask,
            "prompt_len": prompt_len,
        }


class DataManager:
    def __init__(
        self,
        dataset_id: str,
        split: str,
        train_split: float,
        eval_split: float,
        seed: int,
        n_splits: int,
        logger: Optional[Logger] = None,
    ):
        self.logger = logger
        self.n_splits = n_splits
        self.seed = seed
        self.dataset_id = dataset_id
        self.split = split

        if train_split is None or eval_split is None:
            raise ValueError("train_split and eval_split must be set")
        if train_split + eval_split != 1:
            raise ValueError("train_split and eval_split must sum to 1")
        self.train_split = train_split
        self.eval_split = eval_split

        self.dataset = self._load_dataset()

    def _load_dataset(self):
        dataset = load_dataset(path=self.dataset_id, split=self.split)
        if self.logger:
            self.logger.info(f"Loaded dataset with {len(dataset)} examples")
            # sample 1 example for debugging
            self.logger.debug(f"Dataset: {dataset[0]}")

        # TODO(adam): select a subset of the dataset based on the train_split and eval_split

        dataset = dataset.train_test_split(test_size=self.eval_split, seed=self.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        if self.logger:
            self.logger.info(f"Train dataset has {len(train_dataset)} examples")
            self.logger.info(f"Eval dataset has {len(eval_dataset)} examples")
            # self.logger.debug(f"Train dataset: {train_dataset[0]}")
            # self.logger.debug(f"Eval dataset: {eval_dataset[0]}")

        # preprocess dataset
        # dataset = self._split_dataset(self.num_models)

        return dataset

    def _preprocess_dataset(self, dataset: Dataset):
        # TODO(adam): implement dataset preprocessing
        return dataset

    def _split_dataset(self, num_models: int):
        # TODO(adam): implement dataset splitting for num_models
        pass

    def get_dataloader(self, model_idx: int, partition: str):
        # TODO(adam): implement dataloader retrieval for model_idx
        pass
