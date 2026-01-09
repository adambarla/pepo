import copy
import hashlib
import logging
import threading
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from ..utils import DeviceManager

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import get_device_manager, get_hub_manager
from ..utils.model_utils import get_log_probs
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
        tokenizer: AutoTokenizer,
        processor: Optional[DataProcessor] = None,
        collator: Optional[DataCollator] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
        shuffle_train: bool = True,
        ref_model_id: Optional[str] = None,
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
        self.ref_model_id = ref_model_id
        self.inference_batch_size = inference_batch_size
        self.device_manager = device_manager or get_device_manager()
        self.force_recompute = force_recompute
        self._hub_manager = get_hub_manager()

        self._initialize_dataset()

    def _get_processed_name(self) -> str:
        """Generate unique name for processed dataset based on config."""

        # Include critical filtering params in hash
        hash_params = {
            "template": getattr(self.tokenizer, "chat_template", ""),
            "train_split": self.train_split_name,
            "eval_split": self.eval_split_name,
            "max_len": self.max_length,
            "max_prompt_len": self.max_prompt_length,
            "ref_model": self.ref_model_id,
        }
        param_hash = hashlib.md5(str(sorted(hash_params.items())).encode()).hexdigest()[
            :8
        ]

        proc_name = type(self.processor).__name__ if self.processor else "none"
        return f"{self.dataset_id.replace('/', '_')}-{proc_name}-{param_hash}"

    def _initialize_dataset(self) -> None:
        processed_name = self._get_processed_name()

        # Check if processed dataset exists on Hub
        if not self.force_recompute and self._hub_manager.dataset_exists(
            processed_name
        ):
            logger.info(f"Loading processed dataset from Hub: {processed_name}")
            repo_id = self._hub_manager.get_repo_id(processed_name)
            train_data = load_dataset(repo_id, split="train")
            eval_data = load_dataset(repo_id, split="eval")
        else:
            # Load raw and process
            train_raw = load_dataset(self.dataset_id, split=self.train_split_name)
            eval_raw = load_dataset(self.dataset_id, split=self.eval_split_name)
            logger.info(
                f"Loaded {self.dataset_id}: "
                f"train={len(train_raw)}, eval={len(eval_raw)}"
            )

            if self.processor:
                train_data = self.processor.process(train_raw, self.tokenizer)
                eval_data = self.processor.process(eval_raw, self.tokenizer)
            else:
                train_data, eval_data = train_raw, eval_raw

            if self.ref_model_id:
                logger.info(f"Adding reference logprobs using {self.ref_model_id}...")
                train_data = self._add_ref_logprobs(train_data)
                eval_data = self._add_ref_logprobs(eval_data)

            # Push to Hub
            ds_dict = DatasetDict({"train": train_data, "eval": eval_data})
            self._hub_manager.push_dataset(ds_dict, processed_name)

        self._split_train(train_data)
        self.eval_dataset = self._sort_by_length(eval_data)

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
                    attn_implementation="sdpa",
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

                # Use fresh collator
                if self.collator:
                    collator = copy.deepcopy(self.collator)
                    collator.set_tokenizer(copy.deepcopy(self.tokenizer))
                else:
                    collator = DataCollator(
                        tokenizer=copy.deepcopy(self.tokenizer),
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
                    # persistent_workers not available in signature yet
                )

                chosen_logps_shard: list[float] = []
                rejected_logps_shard: list[float] = []

                desc = f"GPU {gpu_id} ({start_idx}-{end_idx})"
                for batch in tqdm(
                    dataloader, desc=desc, position=shard_idx, leave=False
                ):
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
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

        # Launch threads for processing
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

        return dataset.add_column("reference_chosen_logps", final_chosen).add_column(
            "reference_rejected_logps", final_rejected
        )

    def _split_train(self, dataset: Dataset) -> None:
        """Split train into n_splits for ensemble models."""
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

    @property
    def merged_train_dataset(self) -> Dataset:
        from datasets import concatenate_datasets

        return concatenate_datasets(list(self.train_datasets.values()))

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
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
        )

    def get_name(self) -> str:
        return self.dataset_id.replace("/", "_")
