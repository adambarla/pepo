"""CHIPPO (Chi^2 Preference Optimization) Model."""

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
from ..utils.model_utils import get_log_probs
from .single_base import SingleModel

logger = logging.getLogger(__name__)

# Module-level flag to track if we've warned about missing precomputed logprobs
_warned_missing_ref_logprobs = False


class CHIPPOModel(SingleModel):
    """Chi^2 Preference Optimization Model."""

    def __init__(
        self,
        backbone: "BackboneConfig",
        alpha_chi: float,
        beta_chi: float = 0.1,  # Default for safety
        gamma_chi: float = 1.0,
        r_max_chi: float = 10.0,
        use_internal_clipping: bool = True,
        trainer: Optional[SingleModelTrainer] = None,
        generator: Optional[Generator] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize Chi^2 Model."""
        self.alpha_chi = alpha_chi
        self.beta_chi = beta_chi
        self.gamma_chi = gamma_chi
        self.r_max_chi = r_max_chi
        self.use_internal_clipping = use_internal_clipping
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
            train_batch_size=backbone.train_batch_size,
            eval_batch_size=backbone.eval_batch_size,
            generation_batch_size=backbone.generator_batch_size,
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

        logger.info(
            f"CHIPPOModel initialized with alpha={self.alpha_chi}, "
            f"beta={self.beta_chi}, gamma={self.gamma_chi}, r_max={self.r_max_chi}, "
            f"use_internal_clipping={self.use_internal_clipping}"
        )

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
            self.unload()

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

        logger.info("Loaded CHIPPO model")

    def unload(self) -> None:
        """Unload model from GPU memory to free up resources."""
        if self._model is None:
            logger.info("Model is already unloaded")
            return

        logger.info("Unloading CHIPPO model from GPU memory...")
        del self._model
        self._model = None
        self._device_manager.clear_cache()
        self._epoch = 0
        logger.info("CHIPPO model unloaded from GPU memory")

    def save(self) -> None:
        """Save model to Hub."""
        self.checkpoint_manager.push_model(
            model=self.model,
            model_name=self.get_name(),
            tokenizer=self.tokenizer,
            epochs=self._epoch,
        )

    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,  # Ignored - single model
        **kwargs: Any,
    ) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        clip_str = "clip" if self.use_internal_clipping else "noclip"
        repo_name = (
            f"{model_name}-a{self.alpha_chi}-b{self.beta_chi}"
            f"-g{self.gamma_chi}-r{self.r_max_chi}-{clip_str}-chippo"
        )
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

        # 1. Calculate Log Ratios: log(z) = log(pi) - log(pi_ref)
        log_ratio_chosen = lprobs_chosen - lprobs_chosen_ref
        log_ratio_rejected = lprobs_reject - lprobs_reject_ref

        # Check if we should use internal clipping (CHIPPO) or original chi2 DPO
        if self.use_internal_clipping:
            # Retrieve XPO hyperparameters
            # Defaulting to 1.0 if not present, but Table 2 suggests values
            # like 1.25, 0.1, etc.
            alpha_chi = getattr(self, "alpha_chi", 1.0)
            gamma_chi = getattr(self, "gamma_chi", 1.0)

            # 2. Define the generalized link function phi_tilde(z)
            # Formula: phi_tilde(z) = exp(clip(alpha * log_z, -88, 20)) + gamma * log_z
            def compute_phi_tilde(log_z: torch.Tensor, alpha: float, gamma: float):
                # Inner clipping for numerical stability of exp()
                # The text explicitly states clipping the upper range to 20
                # helps reduce instability and uses range [-88, 20].
                scaled_log_z = alpha * log_z
                clipped_scaled_log_z = torch.clamp(scaled_log_z, min=-88, max=20)

                term1 = torch.exp(clipped_scaled_log_z)
                term2 = gamma * log_z

                return term1 + term2

            phi_chosen = compute_phi_tilde(log_ratio_chosen, alpha_chi, gamma_chi)
            phi_rejected = compute_phi_tilde(log_ratio_rejected, alpha_chi, gamma_chi)

            # 3. Calculate preference difference
            logits = self.beta_chi * (phi_chosen - phi_rejected)

            # 4. Outer Clipping (from Algorithm 1 context)
            # While the new text focuses on inner clipping, it says
            # "utilizing the link function... in Algorithm 1".
            # Algorithm 1 includes an outer clip of 2 * R_max.
            clip_limit = 2 * getattr(self, "r_max_chi", 10.0)
            clipped_logits = torch.clamp(logits, min=-clip_limit, max=clip_limit)

            # 5. Loss Calculation
            # Minimize negative log sigmoid of the preference difference
            losses = -F.logsigmoid(clipped_logits)
        else:
            # Original chi2 DPO loss: φ(z) = z + log(z) where z = π/π_ref
            # z = exp(log_ratio), so φ(z) = exp(log_ratio) + log_ratio
            def compute_phi_chi2(log_z: torch.Tensor) -> torch.Tensor:
                z = torch.exp(log_z)
                return z + log_z

            phi_chosen = compute_phi_chi2(log_ratio_chosen)
            phi_rejected = compute_phi_chi2(log_ratio_rejected)

            # Calculate preference difference
            logits = self.beta_chi * (phi_chosen - phi_rejected)

            # Outer clipping with 2*R_max as in the algorithm
            clip_limit = 2 * self.r_max_chi
            clipped_logits = torch.clamp(logits, min=-clip_limit, max=clip_limit)

            # Loss: maximize log σ(clipped_logits), so minimize -log σ(...)
            losses = -F.logsigmoid(clipped_logits)
        loss = losses.mean()

        # Calculate metrics
        with torch.no_grad():
            accuracy = (clipped_logits > 0).float().mean()
            dpo_margins = log_ratio_chosen - log_ratio_rejected

            metrics = {
                "loss": loss.item(),
                "rewards/chosen": (self.beta_chi * log_ratio_chosen).mean().item(),
                "rewards/rejected": (self.beta_chi * log_ratio_rejected).mean().item(),
                "rewards/margins": (self.beta_chi * dpo_margins).mean().item(),
                "accuracy": accuracy.item(),
            }

        return loss, metrics
