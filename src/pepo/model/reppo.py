"""REPPO (Reward Ensemble Pessimistic Preference Optimization) Model.

This module contains all REPPO model classes:
- RewardHead: Linear projection for scalar reward
- REPPORewardModel: Ensemble of L reward models
- REPPOPolicyModel: Single policy model (placeholder loss)
- REPPOModel: Orchestrator for two-phase RLHF training
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, cast

if TYPE_CHECKING:
    from ..trainer import BaseTrainer
    from .config import BackboneConfig


import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import AutoConfig, AutoModel, PreTrainedModel

from ..data import RewardDataCollator
from ..data.annotators.reward import RewardAnnotator
from ..loader import CheckpointManager
from ..utils import get_device_manager, get_hub_manager
from ..utils.model_utils import get_log_probs
from .ensemble_base import EnsembleModel
from .single_base import SingleModel

logger = logging.getLogger(__name__)


class RewardHead(nn.Module):
    """Linear projection from hidden state to scalar reward.

    Uses the last non-padding token's hidden state as input.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar reward from hidden states.

        Args:
            hidden_states: (B, T, H) hidden states from base model
            attention_mask: (B, T) attention mask

        Returns:
            (B,) scalar rewards
        """
        # Get sequence lengths (index of last non-padding token)
        seq_lens = attention_mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        # Extract last non-padding token's hidden state
        last_hidden = hidden_states[batch_indices, seq_lens]  # (B, H)

        # Project to scalar
        return self.linear(last_hidden).squeeze(-1)


class REPPORewardModel(EnsembleModel):
    """Ensemble of L reward models: base LLM + LoRA + RewardHead.

    Each reward model shares the base LLM architecture but has:
    - Independent LoRA adapters
    - Independent RewardHead (linear layer)

    Trained using Bradley-Terry loss on preference pairs.
    """

    def __init__(
        self,
        num_networks: int,
        backbone: "BackboneConfig",
        trainer: Optional["BaseTrainer"] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize REPPO Reward Model ensemble."""
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
            train_batch_size=backbone.train_batch_size,
            eval_batch_size=backbone.eval_batch_size,
            generation_batch_size=backbone.generator_batch_size,
            trainer=trainer,
        )

        self.tokenizer_id = backbone.tokenizer_id
        self.chat_template = backbone.chat_template
        self.debug = debug
        self._models_lock = threading.Lock()

        self.lora_config = LoraConfig(
            r=backbone.lora_r,
            lora_alpha=backbone.lora_alpha,
            lora_dropout=backbone.lora_dropout,
            bias=cast(Literal["none", "all", "lora_only"], backbone.lora_bias),
            task_type="FEATURE_EXTRACTION",
            target_modules=backbone.lora_target_modules,
            modules_to_save=["reward_head"],
        )

        logger.info(
            f"REPPORewardModel initialized with L={self._num_models}, "
            f"model_id={self.model_id}"
        )

    @property
    def reward_heads(self) -> list[RewardHead]:
        """List of reward heads (extracted from PeftModels)."""
        if not self.is_loaded():
            raise RuntimeError("Models not loaded. Call load() first.")
        assert self._models is not None  # for type narrowing
        heads = []
        for model in self._models:
            if not hasattr(model, "reward_head"):
                raise AttributeError("Model missing 'reward_head' attribute")
            heads.append(model.reward_head)
        return heads

    def load(
        self, init_new: bool = False, epoch: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Load models and reward heads into memory."""
        if self._models is not None:
            logger.warning("Models already loaded. Unload first to reload.")
            return

        # We need to know hidden_size to init reward heads if init_new=True
        # For simplicity, we load the first model to get the config
        # This is a bit redundant but ensures we have the right hidden_size

        config = AutoConfig.from_pretrained(self.model_id)
        hidden_size = config.hidden_size
        model_dtype = self._device_manager.dtype
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)

        self._models = []
        for model_idx in range(self._num_models):
            # Reward head stays on CPU, moved to GPU during training
            reward_head = RewardHead(hidden_size).to(dtype=model_dtype)

            self._models.append(
                self.checkpoint_manager.load_model(
                    model_id=self.model_id,
                    model_name=self.get_name(model_idx=model_idx),
                    model_idx=model_idx,
                    lora_config=self.lora_config,
                    init_new=init_new,
                    epoch=epoch,
                    custom_modules={"reward_head": reward_head},
                    model_class=cast(type[PreTrainedModel], AutoModel),
                )
            )

        if epoch is not None:
            self.epochs_per_model = [epoch] * self._num_models

        logger.info(f"Loaded {self._num_models} reward models with heads via PEFT")

    def unload(self) -> None:
        """Unload all models and reward heads from GPU memory."""
        if not self._models:
            logger.info("Models already unloaded")
            return

        self._models = None
        self.epochs_per_model = [0] * self._num_models
        logger.info("All submodels unloaded from GPU memory")

    @property
    def models(self) -> list[PeftModel]:
        if self._models is None:
            with self._models_lock:
                # Double-check pattern to avoid race condition
                if self._models is None:
                    logger.info("Lazy loading models from current epoch state.")
                    epoch = self.get_epoch(0)
                    self.load(epoch=epoch)

        if self._models is None:
            raise RuntimeError("Failed to lazy load models.")

        return self._models

    @models.setter
    def models(self, value: list[PeftModel]) -> None:
        self._models = value

    def _compute_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: PeftModel,
        reward_head: RewardHead,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute scalar reward for a batch of sequences."""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        return reward_head(hidden_states, attention_mask)

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Bradley-Terry loss for preference learning.

        Args:
            batch: Training batch.
            model: Submodel to train.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, metrics).
        """
        model_idx = self.models.index(model)
        reward_head = self.reward_heads[model_idx]

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        chosen_rewards = self._compute_reward(
            chosen_ids, chosen_mask, model, reward_head, device
        )
        rejected_rewards = self._compute_reward(
            rejected_ids, rejected_mask, model, reward_head, device
        )

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            metrics = {
                "loss": loss.item(),
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
                "accuracy": accuracy.item(),
            }

        return loss, metrics

    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-reppo-L{self._num_models}"
        if model_idx is not None:
            repo_name = f"{repo_name}-r{model_idx}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def train(
        self,
        data_manager: Any,
        max_epochs: Optional[int] = None,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
        force_annotation: bool = False,
    ) -> None:
        """Train the reward model ensemble."""
        if self.trainer is None:
            raise ValueError("Trainer not set in REPPORewardModel.")

        self.init_trainer()
        r_epochs = (
            max_epochs if max_epochs is not None else self.trainer.training_epochs
        )

        # Skip if models exist and not forced
        trainer_force = self.trainer.force if self.trainer else False
        can_load = r_epochs is not None and self.can_load_from_epoch(r_epochs)
        if not trainer_force and can_load:
            logger.info(
                f"Reward models for epoch {r_epochs} already exist. Skipping training."
            )
            for i in range(self._num_models):
                self.set_epoch(r_epochs if r_epochs is not None else 0, model_idx=i)
        else:
            self.trainer.train(
                model=self,
                data_manager=data_manager,
                max_epochs=r_epochs,
                wandb_manager=wandb_manager,
                continue_training=continue_training,
            )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        model_indices: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Prediction is not supported for Reward Models."""
        raise NotImplementedError("Predict not implemented for Reward Model")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        greedy_sampling: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        token_callback: Optional[Callable[[str], None]] = None,
        use_cache: bool = True,
        model_indices: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generation is not supported for Reward Models."""
        raise NotImplementedError("Generate not implemented for Reward Model")


class REPPOModel(SingleModel):
    """REPPO Policy Model (Orchestrator).

    This model acts as the Policy Model for RLHF, but also manages a
    helper REPPORewardModel ensemble for reward calculation.

    Training Phases:
    1. Reward Training: Train the helper reward_model (via trainer orchestration).
    2. Policy Training: Train self (policy) using rewards from helper.
    """

    def __init__(
        self,
        backbone: "BackboneConfig",
        reward_model: "REPPORewardModel",
        trainer: Optional["BaseTrainer"] = None,
        beta: float = 0.1,
        generator: Optional[Any] = None,  # Unused - for Phase 2
        force_annotation: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize REPPO Model (Policy)."""
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

        self.beta = beta
        self.debug = debug
        self._force_annotation = force_annotation
        self.reward_model = reward_model

        self.lora_config = LoraConfig(
            r=backbone.lora_r,
            lora_alpha=backbone.lora_alpha,
            lora_dropout=backbone.lora_dropout,
            bias=cast(Literal["none", "all", "lora_only"], backbone.lora_bias),
            task_type=cast(Literal["CAUSAL_LM"], backbone.lora_task_type),
            target_modules=backbone.lora_target_modules,
        )

        logger.info(
            f"REPPOModel initialized as Policy ({self.model_id}) "
            f"with helper Reward Model (L={self.reward_model._num_models})"
        )

    def load(
        self, init_new: bool = False, epoch: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Load Policy Model (self) AND Reward Model (optionally)."""
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
        logger.info("Loaded REPPO Policy model")

    def unload(self) -> None:
        """Unload Policy Model (self) AND Reward Model (helper)."""
        self.reward_model.unload()

        if self._model is None:
            return
        del self._model
        self._model = None
        self._device_manager.clear_cache()
        self._epoch = 0

    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,  # Ignored - policy has only one model
        **kwargs: Any,
    ) -> str:
        """Get policy model name - policy for L reward models."""
        model_name = self.model_id.rsplit("/", 1)[-1]
        # L refers to the number of reward models
        repo_name = f"{model_name}-reppo-b{self.beta}-L{self.reward_model._num_models}"

        if (
            self.reward_model.trainer is not None
            and self.reward_model.trainer.training_epochs is not None
        ):
            repo_name = f"{repo_name}-re{self.reward_model.trainer.training_epochs}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def save(self) -> None:
        """Save policy model to Hub."""
        self.checkpoint_manager.push_model(
            model=self.model,
            model_name=self.get_name(),
            tokenizer=self.tokenizer,
            epochs=self._epoch,
        )

    def train(
        self,
        data_manager: Any,
        max_epochs: Optional[int] = None,
        wandb_manager: Any = None,
        continue_training: bool = False,
    ) -> None:
        self.reward_model.train(
            data_manager=data_manager,
            wandb_manager=wandb_manager,
            continue_training=continue_training,
            force_annotation=self._force_annotation,
        )
        logger.info("Starting Phase 2: Policy Training")

        r_trainer = self.reward_model.trainer
        force_annotation = self._force_annotation or (
            r_trainer.force if r_trainer else False
        )

        reward_annotator = RewardAnnotator(
            reward_model=self.reward_model,
            device_manager=self._device_manager,
            tokenizer=self.tokenizer,
            force=force_annotation,
        )

        # Dynamically add annotator to existing DataManager
        # This handles caching, processing, pushing, and updating internal datasets
        data_manager.add_annotator(reward_annotator)

        # Update DataManager for single-model policy training
        data_manager.set_num_splits(1)

        # Update collator for reward columns
        data_manager.collator = RewardDataCollator(
            tokenizer=data_manager.tokenizer,
            max_length=data_manager.max_length,
            max_prompt_length=data_manager.max_prompt_length,
        )

        self.reward_model.unload()

        self.init_trainer()

        # Force policy training if annotation was forced
        policy_continue = continue_training and not force_annotation
        if self._trainer is None:
            raise ValueError("Policy trainer (_trainer) not set.")
        self._trainer.train(
            model=self,
            data_manager=data_manager,
            max_epochs=max_epochs,
            wandb_manager=wandb_manager,
            continue_training=policy_continue,
        )

    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute RLHF loss for the policy.

        Args:
            batch: Training batch.
            model: Policy model to train.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, metrics).
        """
        c_p = batch["chosen_input_ids"].to(device)
        c_m = batch["chosen_attention_mask"].to(device)
        c_r = batch["chosen_response_mask"].to(device)
        r_p = batch["rejected_input_ids"].to(device)
        r_m = batch["rejected_attention_mask"].to(device)
        r_r = batch["rejected_response_mask"].to(device)
        lp_c = get_log_probs(cast(PreTrainedModel, model), device, c_p, c_m, c_r)
        lp_r = get_log_probs(cast(PreTrainedModel, model), device, r_p, r_m, r_r)

        if "reference_chosen_logps" in batch:
            lrf_c = batch["reference_chosen_logps"].to(device)
            lrf_r = batch["reference_rejected_logps"].to(device)
        else:
            with (
                model.disable_adapter()
                if isinstance(model, PeftModel)
                else torch.no_grad()
            ):
                lrf_c = get_log_probs(
                    cast(PreTrainedModel, model), device, c_p, c_m, c_r
                )
                lrf_r = get_log_probs(
                    cast(PreTrainedModel, model), device, r_p, r_m, r_r
                )

        rs_w = batch["rewards_chosen"].to(device)
        rs_l = batch["rewards_rejected"].to(device)
        r_ps_w = rs_w.min(-1)[0]
        r_ps_l = rs_l.min(-1)[0]

        diff = self.beta * (lp_c - lp_r - (lrf_c - lrf_r)) - (r_ps_w - r_ps_l)
        loss = (diff**2).mean()

        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "rewards/pess_chosen": r_ps_w.mean().item(),
                "rewards/pess_rejected": r_ps_l.mean().item(),
                "accuracy": (lp_c - lp_r > lrf_c - lrf_r).float().mean().item(),
            }
        return loss, metrics
