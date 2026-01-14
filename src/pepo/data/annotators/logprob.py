import copy
import logging
import threading
from typing import Any, Optional, cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from pepo.data.collators.base import DataCollator
from pepo.utils.device import DeviceManager
from pepo.utils.model_utils import get_log_probs

from .base import BaseAnnotator

logger = logging.getLogger(__name__)


class LogprobAnnotator(BaseAnnotator):
    """
    Annotates dataset with reference log probabilities using a pretrained model.
    Uses multi-GPU parallel inference if available via DeviceManager.
    """

    def __init__(
        self,
        ref_model_id: str,
        inference_batch_size: int = 8,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
        dataloader_persistent_workers: bool = False,
        dataloader_prefetch_factor: Optional[int] = None,
        force: bool = False,
    ):
        super().__init__(force=force)
        self.ref_model_id = ref_model_id
        self.inference_batch_size = inference_batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor

    def annotate(self, dataset: Dataset, **kwargs: Any) -> Dataset:
        tokenizer: PreTrainedTokenizerBase = kwargs["tokenizer"]
        device_manager: DeviceManager = kwargs["device_manager"]
        force: bool = kwargs.get("force", False)
        if not device_manager:
            raise ValueError("DeviceManager is required for LogprobAnnotator")

        effective_force = self.force or force

        if not effective_force and "reference_chosen_logps" in dataset.column_names:
            logger.info(
                f"Dataset already contains logprobs. Skipping {self.ref_model_id}."
            )
            return dataset

        available_gpus = device_manager._available_gpus
        num_gpus = len(available_gpus)
        logger.info(
            f"Computing reference lprobs with {self.ref_model_id} on {num_gpus} GPUs..."
        )

        # Load models sequentially
        models: list[PreTrainedModel] = []
        for gpu_id in available_gpus:
            device_str = f"cuda:{gpu_id}"
            logger.info(f"Loading reference model on {device_str}...")
            model = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    self.ref_model_id,
                    dtype=device_manager.dtype,
                    device_map=device_str,
                    attn_implementation="sdpa",
                ),
            )
            model.config.use_cache = False
            model.eval()
            models.append(model)

        total_size = len(dataset)
        chunk_size = (total_size + num_gpus - 1) // num_gpus

        results: list[tuple[list[float], list[float]] | Exception | None] = [
            None
        ] * num_gpus
        threads: list[threading.Thread] = []

        # Split dataset into shards
        shards: list[Optional[Dataset]] = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            if start < total_size:
                shards.append(dataset.select(range(start, end)))
            else:
                shards.append(None)

        def worker(
            model: PreTrainedModel,
            gpu_id: int,
            shard_idx: int,
            sub_dataset: Dataset,
        ) -> None:
            try:
                device = torch.device(f"cuda:{gpu_id}")
                tokenizer_copy = copy.deepcopy(tokenizer)

                collator = DataCollator(
                    tokenizer=tokenizer_copy,
                    max_length=self.max_length,
                    max_prompt_length=self.max_prompt_length,
                )

                # Check persistent workers logic: must be False if num_workers is 0
                persistent = (
                    self.dataloader_persistent_workers
                    if self.dataloader_num_workers > 0
                    else False
                )
                prefetch = (
                    self.dataloader_prefetch_factor
                    if self.dataloader_num_workers > 0
                    else None
                )

                dataloader = DataLoader(
                    cast(torch.utils.data.Dataset[Any], sub_dataset),
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=self.dataloader_num_workers,
                    pin_memory=self.dataloader_pin_memory,
                    persistent_workers=persistent,
                    prefetch_factor=prefetch,
                )

                chosen_logps_shard: list[float] = []
                rejected_logps_shard: list[float] = []

                desc = f"GPU {gpu_id}"
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

                # Synchronize to ensure all CUDA operations complete
                torch.cuda.synchronize()

            except Exception as e:
                logger.error(f"Worker on GPU {gpu_id} failed: {e}")
                results[shard_idx] = e

        # Launch threads
        for i, gpu_id in enumerate(available_gpus):
            if shards[i] is None:
                continue

            # Using casting because shards[i] is definitely Dataset here
            shard = cast(Dataset, shards[i])
            t = threading.Thread(target=worker, args=(models[i], gpu_id, i, shard))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Cleanup - synchronize all GPUs first to ensure all operations complete
        for gpu_id in available_gpus:
            with torch.cuda.device(gpu_id):
                torch.cuda.synchronize()

        for model in models:
            del model
        device_manager.clear_cache()

        # Combine results
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

    def get_signature(self) -> dict[str, Any]:
        return {
            "type": "logprob",
            "model_id": self.ref_model_id,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
        }
