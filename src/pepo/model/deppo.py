"""DEPPO (Direct Ensemble Pessimistic Preference Optimization) Model."""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, cast

if TYPE_CHECKING:
    from .config import BackboneConfig

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel

from ..generator import Generator
from ..loader import CheckpointManager
from ..trainer import EnsembleTrainer
from ..utils import get_device_manager, get_hub_manager
from ..utils.model_utils import get_log_probs, get_next_token_log_probs
from .base import EnsembleModel

logger = logging.getLogger(__name__)

# Module-level flag to track if we've warned about missing precomputed logprobs
_warned_missing_ref_logprobs = False


class DEPPOModel(EnsembleModel):
    """Direct Ensemble Pessimistic Preference Optimization Model."""

    def __init__(
        self,
        backbone: "BackboneConfig",
        num_networks: int,  # This will be overridden by L in config usually
        alpha: float,
        beta: float = 0.1,  # Default for safety
        trainer: Optional[EnsembleTrainer] = None,
        generator: Optional[Generator] = None,
        debug: bool = False,
        shared_backbone: bool = False,
        **kwargs: Any,
    ):
        """Initialize DEPPO Model."""
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
            num_models=num_networks,
            model_id=backbone.model_id,
            device_manager=device_manager,
            hub_manager=hub_manager,
            checkpoint_manager=checkpoint_manager,
            tokenizer=tokenizer,
            trainer=trainer,
            generator=generator,
        )

        self.alpha = alpha
        self.beta = beta
        self.tokenizer_id = backbone.tokenizer_id
        self.chat_template = backbone.chat_template
        self.debug = debug
        self.shared_backbone = shared_backbone
        self.compile_model = backbone.compile

        self.lora_config = LoraConfig(
            r=backbone.lora_r,
            lora_alpha=backbone.lora_alpha,
            lora_dropout=backbone.lora_dropout,
            bias=cast(Literal["none", "all", "lora_only"], backbone.lora_bias),
            task_type=cast(Literal["CAUSAL_LM"], backbone.lora_task_type),
            target_modules=backbone.lora_target_modules,
        )

        logger.info(
            f"DEPPOModel initialized with alpha={self.alpha}, "
            f"beta={self.beta}, L={self._num_models}"
        )

    def train(
        self,
        data_manager: Any,
        max_epochs: Optional[int] = None,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """
        Train the model using the configured trainer.

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
        """
        Load models into memory.

        Args:
            init_new: If True, initialize new models instead of loading from hub.
            epoch: If provided, load models from this epoch checkpoint.
        """
        if self._models is not None:
            logger.warning(
                "Models are already loaded. Unload them first if you want to reload."
            )
            return
        self._models = self._load_models(init_new=init_new, epoch=epoch)
        if epoch is not None:
            self.epochs_per_model = [epoch] * self._num_models

    def _load_models(
        self, init_new: bool = False, epoch: Optional[int] = None
    ) -> list[PeftModel]:
        logger.info(
            f"Loading {self._num_models} models "
            f"(shared_backbone={self.shared_backbone})..."
        )

        if self.shared_backbone:
            # Load the first model (base + adapter 0)
            base_model = self.checkpoint_manager.load_model(
                model_id=self.model_id,
                model_name=self.get_name(model_idx=0),
                model_idx=0,
                lora_config=self.lora_config,
                init_new=init_new,
                epoch=epoch,
            )

            # Load remaining adapters into the same base model
            for model_idx in range(1, self._num_models):
                adapter_name = f"adapter_{model_idx}"
                if not init_new:
                    self.checkpoint_manager.load_adapter(
                        model=base_model,
                        model_name=self.get_name(model_idx=model_idx),
                        adapter_name=adapter_name,
                        epoch=epoch,
                    )
                else:
                    raise NotImplementedError(
                        "shared_backbone=True only supported "
                        "for loading trained models."
                    )

            return [base_model] * self._num_models

        models = []
        for model_idx in range(self._num_models):
            models.append(
                self.checkpoint_manager.load_model(
                    model_id=self.model_id,
                    model_name=self.get_name(model_idx=model_idx),
                    model_idx=model_idx,
                    lora_config=self.lora_config,
                    init_new=init_new,
                    epoch=epoch,
                )
            )
        return models

    def unload(self) -> None:
        """
        Unload all submodels from GPU memory to free up resources.
        """
        if not self._models:
            logger.info("Models are already unloaded")
            return

        logger.info(f"Unloading {len(self._models)} submodels from GPU memory...")

        for model in self._models:
            del model

        self._models = None
        self._device_manager.clear_cache()
        self.epochs_per_model = [0] * self._num_models

        logger.info("All submodels unloaded from GPU memory")

    def load_from_epoch(self, epoch: int) -> None:
        """
        Load models from a specific epoch checkpoint.

        Args:
            epoch: The epoch number to load from.
        """
        logger.info(f"Loading models from epoch {epoch} checkpoint...")

        if self._models is not None:
            logger.info("Unloading existing models before loading from checkpoint...")
            self.unload()
        self.load(init_new=False, epoch=epoch)

        logger.info(f"Successfully loaded models from epoch {epoch} checkpoint")

    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-a{self.alpha}-b{self.beta}-L{self._num_models}"
        if model_idx is not None:
            repo_name = f"{repo_name}-l{model_idx}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss for the ensemble expert.

        Args:
            batch: Training batch.
            model: Submodel (expert) to train.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, metrics).
        """
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

        pi_log_ratio = lprobs_chosen - lprobs_reject
        ref_log_ratio = lprobs_chosen_ref - lprobs_reject_ref
        alpha_offset = math.log(1.0 + self.alpha)
        argument = self.beta * (pi_log_ratio - ref_log_ratio - alpha_offset)
        dpo_loss_components = -F.logsigmoid(argument)
        loss = dpo_loss_components.mean()

        # Calculate metrics
        with torch.no_grad():
            margins = pi_log_ratio - ref_log_ratio
            accuracy = (margins > alpha_offset).float().mean()
            metrics = {
                "loss": loss.item(),
                "rewards/chosen": (self.beta * (lprobs_chosen - lprobs_chosen_ref))
                .mean()
                .item(),
                "rewards/rejected": (self.beta * (lprobs_reject - lprobs_reject_ref))
                .mean()
                .item(),
                "rewards/margins": (self.beta * margins).mean().item(),
                "accuracy": accuracy.item(),
            }

        return loss, metrics

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Pessimistic prediction across ensemble members.

        Args:
            device_input_ids: Input IDs per model.
            device_attention_masks: Attention masks per model.

        Returns:
            Min log probs (B, V).
        """
        if not self.is_loaded():
            raise RuntimeError("Models not loaded. Call load() first.")

        if self.shared_backbone:
            # Sequential Execution for shared backbone
            # The model should already be on a GPU before the generation loop.
            # We don't acquire GPU here since this is called every token.
            model = self.models[0]
            device = model.device

            with torch.no_grad():
                log_probs_list = []

                for model_idx in range(self._num_models):
                    adapter_name = (
                        "default" if model_idx == 0 else f"adapter_{model_idx}"
                    )
                    model.set_adapter(adapter_name)

                    # Full forward pass (no KV cache as requested for stability)
                    inp = device_input_ids[model_idx].to(device)
                    mask = device_attention_masks[model_idx].to(device)

                    log_probs = get_next_token_log_probs(model, inp, mask)
                    log_probs_list.append(
                        log_probs.cpu()
                    )  # Move to CPU for aggregation

            log_probs_tensor = torch.stack(log_probs_list, dim=0)  # (L, B, V)
            min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
            return min_log_probs

        # Parallel Execution (Separate Models)
        log_probs_ensemble: list[Optional[torch.Tensor]] = [None] * self._num_models
        thread_exceptions: list[Optional[BaseException]] = [None] * self._num_models

        def predict_log_probs(
            model_idx: int,
            model: PeftModel,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> None:
            try:
                # Use semaphore to acquire a GPU
                with self.device_manager.request_gpu() as device:
                    with torch.no_grad():
                        model.eval()
                        # Ensure inputs are on the assigned device
                        inp = input_ids.to(device)
                        mask = attention_mask.to(device)

                        # Ensure model is on the device (if not already)
                        # Note: For generation, we assume models fit in VRAM.
                        # Strict offloading per-token is too slow.
                        if model.device != device:
                            model.to(device)

                        log_probs = get_next_token_log_probs(model, inp, mask)
                        log_probs_ensemble[model_idx] = log_probs.cpu()
            except BaseException as e:
                thread_exceptions[model_idx] = e

        threads = []
        for model_idx in range(self._num_models):
            # We assume inputs are on CPU initially or we move them inside the thread
            # To avoid pickling large tensors if they are on GPU, we pass them as is.
            # But the caller of predict usually passes 'device_input_ids'.
            # If they are already on GPU, passing to thread is fine (share memory).

            thread = threading.Thread(
                target=predict_log_probs,
                args=(
                    model_idx,
                    self.models[model_idx],
                    device_input_ids[model_idx],
                    device_attention_masks[model_idx],
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # Propagate any thread exceptions to main thread
        for model_idx, exc in enumerate(thread_exceptions):
            if exc is not None:
                raise RuntimeError(
                    f"Exception in submodel {model_idx} prediction"
                ) from exc

        if len(log_probs_ensemble) != self._num_models:
            raise RuntimeError(
                f"Expected {self._num_models} log prob tensors, "
                f"got {len(log_probs_ensemble)}"
            )
        if len(log_probs_ensemble) == 1:
            result = log_probs_ensemble[0]
            if result is None:
                raise RuntimeError("Unexpected None in log_probs_ensemble")
            return result

        log_probs_ensemble_filtered = [
            log_probs.cpu() for log_probs in log_probs_ensemble if log_probs is not None
        ]
        log_probs_tensor: torch.Tensor = torch.stack(
            log_probs_ensemble_filtered, dim=0
        )  # (L, B, V)
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        return min_log_probs
