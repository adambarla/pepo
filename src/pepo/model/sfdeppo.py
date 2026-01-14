"""SFDEPPO (Simplified DPO) Model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, cast

if TYPE_CHECKING:
    from .config import BackboneConfig

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel

from ..generator import Generator
from ..loader import CheckpointManager
from ..trainer import SingleModelTrainer
from ..utils import get_device_manager, get_hub_manager
from ..utils.model_utils import get_log_probs, get_next_token_log_probs
from .base import SingleModel

logger = logging.getLogger(__name__)

# Module-level flag to track if we've warned about missing precomputed logprobs
_warned_missing_ref_logprobs = False


class SFDEPPOModel(SingleModel):
    """Simplified DPO Model."""

    def __init__(
        self,
        backbone: "BackboneConfig",
        beta: float = 0.1,
        sft_weight: float = 1.0,
        trainer: Optional[SingleModelTrainer] = None,
        generator: Optional[Generator] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize SFDEPPO Model."""
        self.beta = beta
        self.sft_weight = sft_weight
        self.debug = debug

        device_manager = get_device_manager()
        hub_manager = get_hub_manager()

        checkpoint_manager = CheckpointManager(
            device_manager=device_manager,
            hub_manager=hub_manager,
            compile_model=backbone.compile,
        )

        tokenizer = checkpoint_manager.load_tokenizer(
            model_id=backbone.model_id,
            tokenizer_id=backbone.tokenizer_id,
            chat_template=backbone.chat_template,
        )

        super().__init__(
            model_id=backbone.model_id,
            device_manager=device_manager,
            hub_manager=hub_manager,
            checkpoint_manager=checkpoint_manager,
            tokenizer=tokenizer,
            trainer=trainer,
            generator=generator,
        )

        self.tokenizer_id = backbone.tokenizer_id
        self.chat_template = backbone.chat_template
        self.compile_model = backbone.compile

        self.lora_config = LoraConfig(
            r=backbone.lora_r,
            lora_alpha=backbone.lora_alpha,
            lora_dropout=backbone.lora_dropout,
            bias=cast(Literal["none", "all", "lora_only"], backbone.lora_bias),
            task_type=cast(Literal["CAUSAL_LM"], backbone.lora_task_type),
            target_modules=backbone.lora_target_modules,
        )

        logger.info(f"SFDEPPOModel initialized with beta={self.beta}")

    def train(
        self,
        data_manager: Any,
        max_epochs: Optional[int] = None,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """Train the model using the configured trainer.

        Args:
            data_manager: Data manager for training data.
            max_epochs: Optional number of epochs to train for.
            wandb_manager: Optional WandbManager instance for logging.
            continue_training: Whether to continue from checkpoint.
        """
        if self._trainer is None:
            raise ValueError("Trainer not configured in model config.")

        self.init_trainer()
        self._trainer.train(
            model=self,
            data_manager=data_manager,
            max_epochs=max_epochs,
            wandb_manager=wandb_manager,
            continue_training=continue_training,
        )

    def load(
        self,
        init_new: bool = False,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Load model into memory.

        Args:
            init_new: If True, initialize new model instead of loading from hub.
            epoch: If provided, load model from this epoch checkpoint.
        """
        if self._model is not None:
            logger.warning(
                "Model is already loaded. Unload first if you want to reload."
            )
            return

        self._model = self.checkpoint_manager.load_model(
            model_id=self.model_id,
            model_name=self.get_name(),
            model_idx=0,
            lora_config=self.lora_config,
            init_new=init_new,
            epoch=epoch,
        )
        if epoch is not None:
            self._epoch = epoch

        logger.info("Loaded SFDEPPO model")

    def unload(self) -> None:
        """Unload model from GPU memory to free up resources."""
        if self._model is None:
            logger.info("Model is already unloaded")
            return

        logger.info("Unloading SFDEPPO model from GPU memory...")
        del self._model
        self._model = None
        self._device_manager.clear_cache()
        self._epoch = 0
        logger.info("SFDEPPO model unloaded from GPU memory")

    def save(self) -> None:
        """Save model to Hub."""
        self.checkpoint_manager.push_model(
            model=self.model,
            model_name=self.get_name(),
            tokenizer=self.tokenizer,
            epochs=self._epoch,
        )

    def load_from_epoch(self, epoch: int) -> None:
        """Load model from a specific epoch checkpoint.

        Args:
            epoch: The epoch number to load from.
        """
        logger.info(f"Loading model from epoch {epoch} checkpoint...")

        if self._model is not None:
            logger.info("Unloading existing model before loading from checkpoint...")
            self.unload()
        self.load(init_new=False, epoch=epoch)

        logger.info(f"Successfully loaded model from epoch {epoch} checkpoint")

    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,  # Ignored - single model
        **kwargs: Any,
    ) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-b{self.beta}-sfdeppo"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch = {k: v.to(device) for k, v in batch.items()}

        chosen_ids = batch["chosen_input_ids"]
        chosen_amask = batch["chosen_attention_mask"]
        chosen_rmask = batch["chosen_response_mask"]
        reject_ids = batch["rejected_input_ids"]
        reject_amask = batch["rejected_attention_mask"]
        reject_rmask = batch["rejected_response_mask"]

        lprobs_chosen = get_log_probs(
            model, device, chosen_ids, chosen_amask, chosen_rmask, debug=self.debug
        )
        lprobs_reject = get_log_probs(
            model, device, reject_ids, reject_amask, reject_rmask, debug=self.debug
        )

        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            lprobs_chosen_ref = batch["reference_chosen_logps"]
            lprobs_reject_ref = batch["reference_rejected_logps"]
        else:
            global _warned_missing_ref_logprobs
            if not _warned_missing_ref_logprobs:
                logger.warning(
                    "Precomputed reference logprobs not found in batch. "
                    "Computing on-the-fly (2x slower). "
                    "Consider preprocessing dataset with ref_model_id "
                    "to cache them."
                )
                _warned_missing_ref_logprobs = True
            # if we don't have reference logprobs, we need to compute them
            with model.disable_adapter():
                with torch.no_grad():
                    lprobs_chosen_ref = get_log_probs(
                        model,
                        device,
                        chosen_ids,
                        chosen_amask,
                        chosen_rmask,
                        debug=self.debug,
                    )
                    lprobs_reject_ref = get_log_probs(
                        model,
                        device,
                        reject_ids,
                        reject_amask,
                        reject_rmask,
                        debug=self.debug,
                    )

        # Calculate logits: chosen - ref_chosen - rejected + ref_rejected
        logits = lprobs_chosen - lprobs_chosen_ref - lprobs_reject + lprobs_reject_ref

        sft_loss = -lprobs_chosen
        losses = -F.logsigmoid(self.beta * logits) + (self.sft_weight * sft_loss)
        loss = losses.mean()

        # Calculate metrics
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            log_ratio_chosen = lprobs_chosen - lprobs_chosen_ref
            log_ratio_rejected = lprobs_reject - lprobs_reject_ref
            dpo_margins = log_ratio_chosen - log_ratio_rejected

            metrics = {
                "loss": loss.item(),
                "rewards/chosen": (self.beta * log_ratio_chosen).mean().item(),
                "rewards/rejected": (self.beta * log_ratio_rejected).mean().item(),
                "rewards/margins": (self.beta * dpo_margins).mean().item(),
                "accuracy": accuracy.item(),
            }

        return loss, metrics

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Inference prediction using the single model.

        Args:
            device_input_ids: Input IDs (list of length 1).
            device_attention_masks: Attention masks (list of length 1).

        Returns:
            Log probs for the next token (B, V).
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call model.load() before using the model."
            )

        if len(device_input_ids) != 1 or len(device_attention_masks) != 1:
            raise ValueError(
                f"SFDEPPOModel.predict expects input lists of length 1, "
                f"got {len(device_input_ids)} and {len(device_attention_masks)}"
            )

        with torch.no_grad():
            self._model.eval()
            log_probs = get_next_token_log_probs(
                self._model,
                device_input_ids[0],
                device_attention_masks[0],
            )

        return log_probs
