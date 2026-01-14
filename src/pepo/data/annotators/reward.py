import copy
import logging
import threading
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from pepo.data.collators.base import DataCollator
from pepo.utils.device import DeviceManager

from .base import BaseAnnotator

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pepo.model.reppo import REPPORewardModel


class RewardAnnotator(BaseAnnotator):
    """
    Annotates dataset with rewards from an ensemble of models.
    Supports parallel inference across multiple GPUs.
    """

    def __init__(
        self,
        reward_model: "REPPORewardModel",
        device_manager: DeviceManager,
        tokenizer: PreTrainedTokenizerBase,
        force: bool = False,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
    ):
        super().__init__(force=force)
        self.reward_model = reward_model
        self.device_manager = device_manager
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def annotate(self, dataset: Dataset, **kwargs: Any) -> Dataset:
        """Annotate dataset with rewards.

        Args:
            dataset: Dataset to annotate.
            force: Whether to force recomputation (combines with self.force).

        Returns:
            Annotated dataset.
        """
        force: bool = kwargs.get("force", False)
        effective_force = self.force or force

        # Check if first model's rewards exist as a heuristic
        num_models = self.reward_model.num_models
        cols = dataset.column_names
        if not effective_force and "rewards_0_chosen" in cols:
            has_all = all(f"rewards_{i}_chosen" in cols for i in range(num_models))
            if has_all:
                logger.info("Dataset already has rewards. Skipping.")
                return dataset

        trainer = self.reward_model.trainer
        batch_size = getattr(trainer, "eval_batch_size", 8) if trainer else 8

        dataset_lock = threading.Lock()

        logger.info(
            f"Starting reward annotation with {num_models} models "
            f"({self.device_manager.num_available_gpus} GPUs available)..."
        )

        # Container for safe update in threads
        shared_dataset = [dataset]
        threads = []

        for i in range(num_models):
            args = (i, dataset, batch_size, dataset_lock, shared_dataset)
            t = threading.Thread(target=self._annotate_single_model, args=args)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return shared_dataset[0]

    def _annotate_single_model(
        self,
        model_idx: int,
        dataset: Dataset,
        batch_size: int,
        dataset_lock: threading.Lock,
        shared_dataset: list[Dataset],
    ) -> None:
        with self.device_manager.request_gpu() as device:
            model = self.reward_model.models[model_idx]
            reward_head = self.reward_model.reward_heads[model_idx]
            model.to(device)

            tokenizer_copy = copy.deepcopy(self.tokenizer)
            collator = DataCollator(
                tokenizer=tokenizer_copy,
                max_length=self.max_length,
                max_prompt_length=self.max_prompt_length,
            )

            dataloader = DataLoader(
                cast(torch.utils.data.Dataset[Any], dataset),
                batch_size=batch_size,
                collate_fn=collator,
                pin_memory=True,
                num_workers=0,
            )

            rewards_chosen = []
            rewards_rejected = []

            desc = f"Model {model_idx} Inference"
            for batch in tqdm(dataloader, desc=desc, position=model_idx, leave=False):
                with torch.no_grad():
                    r_c = self.reward_model._compute_reward(
                        batch["chosen_input_ids"].to(device),
                        batch["chosen_attention_mask"].to(device),
                        model,
                        reward_head,
                        device,
                    )
                    r_r = self.reward_model._compute_reward(
                        batch["rejected_input_ids"].to(device),
                        batch["rejected_attention_mask"].to(device),
                        model,
                        reward_head,
                        device,
                    )

                    rewards_chosen.extend(r_c.cpu().tolist())
                    rewards_rejected.extend(r_r.cpu().tolist())

            with dataset_lock:
                current_ds = shared_dataset[0]
                current_ds = current_ds.add_column(
                    f"rewards_{model_idx}_chosen", rewards_chosen
                )
                current_ds = current_ds.add_column(
                    f"rewards_{model_idx}_rejected", rewards_rejected
                )
                shared_dataset[0] = current_ds
                logger.info(f"Model {model_idx} annotation merged.")

            # Move model back to CPU
            model.to("cpu")
            torch.cuda.empty_cache()

    def get_signature(self) -> dict[str, Any]:
        # Hash based on model name and epoch (which captures parameters state)
        epoch = self.reward_model.get_epoch(0)
        return {
            "type": "reward",
            "model_name": self.reward_model.get_name(epoch=epoch),
        }
