import copy
import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import DeviceManager
from .model_utils import get_log_probs

logger = logging.getLogger(__name__)


class LengthBasedBatchSampler(BatchSampler):
    """
    Batch sampler that groups consecutive examples (sorted by length) into batches,
    then shuffles the batches rather than individual examples.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        base_sampler = SequentialSampler(range(dataset_size))
        super().__init__(base_sampler, batch_size, drop_last=False)
        self.shuffle = shuffle
        self.seed = seed
        self._batches = [
            list(batch)
            for batch in BatchSampler(base_sampler, batch_size, drop_last=False)
        ]

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(batches)
        return iter(batches)


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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts = [f["prompt_text"] for f in features]
        chosen_texts = [f["chosen_text"] for f in features]
        reject_texts = [f["rejected_text"] for f in features]

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
        reject_encoded = self.tokenizer(
            reject_texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # response mask is the XOR of extended prompt_mask and chosen_att_mask
        # extend the prompt_attention_mask by 0s to chosen / rejected size
        prompt_mask = prompt_encoded["attention_mask"]  # B, T_p
        chosen_att_mask = chosen_encoded["attention_mask"]  # B, T_c
        reject_att_mask = reject_encoded["attention_mask"]  # B, T_r

        B, T_p = prompt_mask.shape  # B, T_p

        chosen_zero_mask = torch.zeros_like(chosen_att_mask[:, T_p:])  # B, T_c - T_p
        reject_zero_mask = torch.zeros_like(reject_att_mask[:, T_p:])  # B, T_r - T_p
        chosen_resp_mask = torch.cat([prompt_mask, chosen_zero_mask], dim=-1)  # B, T_c
        chosen_resp_mask ^= chosen_att_mask
        reject_resp_mask = torch.cat([prompt_mask, reject_zero_mask], dim=-1)  # B, T_r
        reject_resp_mask ^= reject_att_mask

        batch = {
            "prompt_input_ids": prompt_encoded["input_ids"],
            "chosen_input_ids": chosen_encoded["input_ids"],
            "rejected_input_ids": reject_encoded["input_ids"],
            "prompt_attention_mask": prompt_encoded["attention_mask"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_attention_mask": reject_encoded["attention_mask"],
            "chosen_response_mask": chosen_resp_mask,
            "rejected_response_mask": reject_resp_mask,
        }

        if "reference_chosen_logps" in features[0]:
            batch["reference_chosen_logps"] = torch.tensor(
                [f["reference_chosen_logps"] for f in features], dtype=torch.float
            )
        if "reference_rejected_logps" in features[0]:
            batch["reference_rejected_logps"] = torch.tensor(
                [f["reference_rejected_logps"] for f in features], dtype=torch.float
            )

        return batch


class DataManager:
    """Manages dataset loading, preprocessing, and splitting for PEPO ensemble."""

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
        dataloader_num_workers: Optional[int] = 0,
        dataloader_pin_memory: bool = False,
        dataloader_persistent_workers: bool = False,
        dataloader_prefetch_factor: Optional[int] = 2,
        ref_model_id: Optional[str] = None,
        inference_batch_size: int = 8,
        device_manager: Optional[DeviceManager] = None,
        shuffle_train: bool = True,
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
            dataloader_num_workers: Number of worker processes for data loading
                (0 = main thread only).
            dataloader_pin_memory: Pin memory for faster CPU->GPU transfer.
            dataloader_persistent_workers: Keep workers alive between epochs.
            dataloader_prefetch_factor: Number of batches to prefetch per worker.
            ref_model_id: ID of the reference model for caching logprobs.
            inference_batch_size: Batch size for computing reference logprobs.
            device_manager: Device manager for model loading.
        """
        self.n_splits = n_splits
        self.seed = seed
        self.dataset_id = dataset_id
        self.split = split
        self.ref_model_id = ref_model_id
        self.inference_batch_size = inference_batch_size
        self.device_manager = device_manager

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

        self.dataloader_num_workers = (
            dataloader_num_workers if dataloader_num_workers is not None else 0
        )
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = (
            dataloader_persistent_workers if self.dataloader_num_workers > 0 else False
        )
        self.dataloader_prefetch_factor = (
            dataloader_prefetch_factor if self.dataloader_num_workers > 0 else None
        )
        self.shuffle_train = shuffle_train

        logger.info(f"Dataloader num workers: {self.dataloader_num_workers}")
        logger.info(f"Dataloader pin memory: {self.dataloader_pin_memory}")
        logger.info(
            f"Dataloader persistent workers: {self.dataloader_persistent_workers}"
        )
        logger.info(f"Dataloader prefetch factor: {self.dataloader_prefetch_factor}")

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
            "ref_model_id": self.ref_model_id or "none",
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

        logger.info(f"Loading preprocessed dataset from cache: {self.cache_path}")

        try:
            dataset = Dataset.load_from_disk(self.cache_path)
            logger.info(f"Successfully loaded from cache - {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}. Reprocessing...")
            return None

    def _save_cache(self, dataset: Dataset) -> None:
        if not self.cache_path:
            return

        if self.force_recompute and os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)

        logger.info(f"Saving preprocessed dataset to cache: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)
        dataset.save_to_disk(self.cache_path)

    def _load(self) -> Dataset:
        raw_dataset = load_dataset(path=self.dataset_id, split=self.split)
        logger.info(f"Loaded dataset with {len(raw_dataset)} examples")
        return raw_dataset

    def _ensure_message_list(
        self, messages: Any, is_prompt: bool = False
    ) -> list[dict[str, str]] | None:
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

    def _is_valid_length(
        self, prompt_str: str, chosen_str: str, reject_str: str
    ) -> bool:
        """
        Checks the lenght of tokenized example to see if it's under the length limits.
        Tokenizes with add_special_tokens=True and truncation=False
        to get accurate length (matches actual tokenization)
        Returns:
            True if the example is under the length limits, False otherwise.
        """
        if self.max_prompt_length is None and self.max_length is None:
            return True

        if self.max_prompt_length is not None:
            prompt_tokens = self.tokenizer(prompt_str, truncation=False)
            prompt_len = len(prompt_tokens["input_ids"])
            if prompt_len > self.max_prompt_length:
                return False

        if self.max_length is not None:
            chosen_tokens = self.tokenizer(chosen_str, truncation=False)
            chosen_len = len(chosen_tokens["input_ids"])
            if chosen_len > self.max_length:
                return False
            reject_tokens = self.tokenizer(reject_str, truncation=False)
            reject_len = len(reject_tokens["input_ids"])
            if reject_len > self.max_length:
                return False
        return True

    def _process_helper(self, examples):
        """
        Apply chat templates and filter invalid/long examples.
        Stores formatted strings instead of tokenized sequences.
        Tokenization happens in the collator for better performance.
        """
        processed: dict[str, list[str]] = {
            "prompt_text": [],
            "chosen_text": [],
            "rejected_text": [],
        }

        for i in range(len(examples["prompt"])):
            curr_prompt = examples["prompt"][i]
            curr_chosen = examples["chosen"][i]
            curr_reject = examples["rejected"][i]

            curr_prompt = self._ensure_message_list(curr_prompt, is_prompt=True)
            curr_chosen = self._ensure_message_list(curr_chosen)
            curr_reject = self._ensure_message_list(curr_reject)

            if curr_prompt is None or curr_chosen is None or curr_reject is None:
                continue

            prompt_str = self.tokenizer.apply_chat_template(
                curr_prompt, tokenize=False, add_generation_prompt=True
            )
            chosen_str = (
                self.tokenizer.apply_chat_template(curr_chosen, tokenize=False)
                + self.tokenizer.eos_token
            )
            reject_str = (
                self.tokenizer.apply_chat_template(curr_reject, tokenize=False)
                + self.tokenizer.eos_token
            )

            if not self._is_valid_length(prompt_str, chosen_str, reject_str):
                continue  # Skip this example if it's too long

            processed["prompt_text"].append(prompt_str)
            processed["chosen_text"].append(chosen_str)
            processed["rejected_text"].append(reject_str)

        return processed

    def _process(self, dataset: Dataset) -> Dataset:
        logger.info("Preprocessing dataset (applying chat template and filtering)...")

        original_size = len(dataset)

        dataset_preprocessed = dataset.map(
            self._process_helper,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_proc,
            desc="Preprocessing dataset",
        )

        logger.info(
            f"Dataset: {original_size} -> {len(dataset_preprocessed)} examples "
            f"(filtered {original_size - len(dataset_preprocessed)})"
        )

        return dataset_preprocessed

    def _compute_example_length(self, example: Dict[str, Any]) -> int:
        """
        Compute the length of an example for sorting purposes.
        Uses the maximum of chosen_text and rejected_text lengths since both
        are in the same batch and need to be padded to the same length.
        """
        chosen_len = len(example["chosen_text"])
        rejected_len = len(example["rejected_text"])
        return max(chosen_len, rejected_len)

    def _sort_dataset_by_length(self, dataset: Dataset) -> Dataset:
        """
        Sort dataset by example length (descending, largest first).
        This reduces padding in batches by grouping similar-length examples.
        """
        lengths = [
            self._compute_example_length(dataset[i]) for i in range(len(dataset))
        ]
        sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        sorted_dataset = dataset.select(sorted_indices)
        return sorted_dataset

    def _split(self, dataset: Dataset) -> None:
        """
        Split preprocessed dataset into train/eval, then split train into n_splits.
        Shuffles train dataset before splitting to ensure random distribution
        across submodels. Each split is then sorted by length for efficient batching.
        Format conversion is handled by the DataCollator.
        """
        dataset_split = dataset.train_test_split(
            test_size=self.eval_split, seed=self.seed
        )
        train_dataset_preprocessed = dataset_split["train"]
        eval_dataset_preprocessed = dataset_split["test"]

        logger.info(
            f"Train: {len(train_dataset_preprocessed)} examples, "
            f"Eval: {len(eval_dataset_preprocessed)} examples"
        )

        np.random.seed(self.seed)
        train_indices = np.arange(len(train_dataset_preprocessed))
        np.random.shuffle(train_indices)
        train_dataset_shuffled = train_dataset_preprocessed.select(train_indices)

        split_indices = np.array_split(
            np.arange(len(train_dataset_shuffled)), self.n_splits
        )
        self.train_datasets = {}
        for model_idx, indices in enumerate(split_indices):
            split_dataset = train_dataset_shuffled.select(indices)
            self.train_datasets[model_idx] = self._sort_dataset_by_length(split_dataset)
        self.eval_dataset = self._sort_dataset_by_length(eval_dataset_preprocessed)

        logger.info("Dataset preprocessing and splitting complete.")

    def _initialize_dataset(self) -> None:
        dataset_preprocessed = self._load_cache()

        if dataset_preprocessed is None:
            if self.force_recompute:
                logger.info("Recomputing dataset (force_recompute=True)...")
            else:
                logger.info(
                    f"No cached dataset found at {self.cache_path}. "
                    f"Creating new dataset..."
                )
            dataset_raw = self._load()
            dataset_preprocessed = self._process(dataset_raw)

            if self.ref_model_id:
                dataset_preprocessed = self._add_ref_logprobs(dataset_preprocessed)
            else:
                logger.warning(
                    "No reference model ID provided."
                    "Training will be 2x slower without precomputed reference logprobs."
                )

            self._save_cache(dataset_preprocessed)

        self._split(dataset_preprocessed)

    def _add_ref_logprobs(self, dataset: Dataset) -> Dataset:
        if self.device_manager is None:
            raise ValueError(
                "DeviceManager is required for reference lprob computation"
            )
        if self.ref_model_id is None:
            raise ValueError("ref_model_id is required for reference lprob computation")

        available_gpus = self.device_manager._available_gpus
        num_gpus = len(available_gpus)
        logger.info(f"Computing reference lprobs with {self.ref_model_id}...")

        # Load models sequentially (following trainer pattern)
        models: list[AutoModelForCausalLM] = []
        for gpu_id in available_gpus:
            device_str = f"cuda:{gpu_id}"
            logger.info(f"Loading reference model on {device_str}...")
            model = cast(
                AutoModelForCausalLM,
                AutoModelForCausalLM.from_pretrained(
                    self.ref_model_id,
                    dtype=self.device_manager.dtype,
                    device_map=device_str,
                    attn_implementation="sdpa",  # Match training config
                ),
            )
            model.config.use_cache = False  # Match training config
            model.eval()
            models.append(model)

        total_size = len(dataset)
        chunk_size = (total_size + num_gpus - 1) // num_gpus

        results: list[tuple[list[float], list[float]] | Exception | None] = [
            None
        ] * num_gpus
        threads: list[threading.Thread] = []

        def worker(
            model: AutoModelForCausalLM,
            gpu_id: int,
            shard_idx: int,
            start_idx: int,
            end_idx: int,
        ) -> None:
            try:
                device = torch.device(f"cuda:{gpu_id}")
                sub_dataset = dataset.select(range(start_idx, end_idx))

                collator = DataCollator(
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    max_prompt_length=self.max_prompt_length,
                )

                dataloader = DataLoader(
                    sub_dataset,
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=self.dataloader_num_workers,
                    pin_memory=self.dataloader_pin_memory,
                    persistent_workers=self.dataloader_persistent_workers,
                    prefetch_factor=self.dataloader_prefetch_factor,
                )

                chosen_logps_shard: list[float] = []
                rejected_logps_shard: list[float] = []

                desc = f"GPU {gpu_id} ({start_idx}-{end_idx})"
                for batch in tqdm(
                    dataloader, desc=desc, position=shard_idx, leave=False
                ):
                    with torch.no_grad():
                        chosen_logps = get_log_probs(
                            model,
                            device,
                            batch["chosen_input_ids"],
                            batch["chosen_attention_mask"],
                            batch["chosen_response_mask"],
                        )
                        rejected_logps = get_log_probs(
                            model,
                            device,
                            batch["rejected_input_ids"],
                            batch["rejected_attention_mask"],
                            batch["rejected_response_mask"],
                        )
                    chosen_logps_shard.extend(chosen_logps.cpu().tolist())
                    rejected_logps_shard.extend(rejected_logps.cpu().tolist())

                results[shard_idx] = (chosen_logps_shard, rejected_logps_shard)

            except Exception as e:
                logger.error(f"Worker on GPU {gpu_id} failed: {e}")
                results[shard_idx] = e

        # Launch threads for processing (models already loaded)
        for i, gpu_id in enumerate(available_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            if start >= total_size:
                break

            t = threading.Thread(target=worker, args=(models[i], gpu_id, i, start, end))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Cleanup models
        for model in models:
            del model
        if self.device_manager:
            self.device_manager.clear_cache()

        # Check for errors and combine results
        final_chosen: list[float] = []
        final_rejected: list[float] = []

        for res in results:
            if isinstance(res, Exception):
                raise res
            if res is None:
                continue
            chosen, rejected = res
            final_chosen.extend(chosen)
            final_rejected.extend(rejected)

        dataset = dataset.add_column("reference_chosen_logps", final_chosen)
        dataset = dataset.add_column("reference_rejected_logps", final_rejected)

        return dataset

    def get_dataloader(
        self,
        model_idx: int,
        partition: str,
        batch_size: int,
        shuffle: Optional[bool] = None,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """
        Get DataLoader for a specific model and partition.

        Args:
            model_idx: Index of the model in the ensemble (0 to n_splits-1).
            partition: Either "train" or "eval".
            batch_size: Batch size for the DataLoader.
            shuffle: Override shuffle behavior. None uses default (True for train).

        Returns:
            DataLoader for the specified model and partition.
        """
        if model_idx < 0 or model_idx >= self.n_splits:
            raise ValueError(
                f"Model index {model_idx} out of range [0, {self.n_splits - 1}]"
            )

        if partition == "train":
            dataset = self.train_datasets[model_idx]
            shuffle_batches = self.shuffle_train if shuffle is None else shuffle
        elif partition == "eval":
            dataset = self.eval_dataset
            shuffle_batches = False if shuffle is None else shuffle
        else:
            raise ValueError(f"Partition must be 'train' or 'eval', got '{partition}'")

        tokenizer_copy = copy.deepcopy(self.tokenizer)

        collator = DataCollator(
            tokenizer=tokenizer_copy,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
        )

        batch_sampler = LengthBasedBatchSampler(
            dataset_size=len(dataset),
            batch_size=batch_size,
            shuffle=shuffle_batches,
            seed=self.seed if shuffle_batches else None,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=self.dataloader_persistent_workers,
            prefetch_factor=self.dataloader_prefetch_factor,
        )
