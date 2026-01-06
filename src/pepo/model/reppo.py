"""REPPO (Reward Ensemble Pessimistic Preference Optimization) Model.

This module contains all REPPO model classes:
- RewardHead: Linear projection for scalar reward
- REPPORewardModel: Ensemble of L reward models
- REPPOPolicyModel: Single policy model (placeholder loss)
- REPPOModel: Orchestrator for two-phase RLHF training
"""

import logging
import threading
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer

from ..loader import ModelLoader
from ..utils import DeviceManager, HubManager
from .base import BaseModel

# Trainers will be implemented later
# if TYPE_CHECKING:
#     from ..trainer.reppo import REPPOPolicyTrainer, REPPORewardTrainer

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
        return self.linear(last_hidden).squeeze(-1)  # type: ignore[no-any-return]


class REPPORewardModel:
    """Ensemble of L reward models: base LLM + LoRA + RewardHead.

    Each reward model shares the base LLM architecture but has:
    - Independent LoRA adapters
    - Independent RewardHead (linear layer)

    Trained using Bradley-Terry loss on preference pairs.
    """

    def __init__(
        self,
        num_networks: int,
        model_id: str,
        device_manager: DeviceManager,
        hub_manager: HubManager,
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        lora_target_modules: str = "all-linear",
        compile: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize REPPO Reward Model ensemble.

        Args:
            num_networks: Number of reward models (L).
            model_id: HuggingFace model ID for base model.
            device_manager: Device manager for GPU allocation.
            hub_manager: Hub manager for model storage.
            tokenizer_id: Optional tokenizer ID (defaults to model_id).
            chat_template: Optional chat template override.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            lora_bias: LoRA bias mode.
            lora_task_type: LoRA task type.
            lora_target_modules: LoRA target modules.
            compile: Whether to compile the model.
            trainer: Optional trainer instance.
            debug: Enable debug logging.
        """
        self._num_models = num_networks
        self.model_id = model_id
        self._device_manager = device_manager
        self._hub_manager = hub_manager
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self.debug = debug

        # Initialize Loader
        self.loader = ModelLoader(
            device_manager=device_manager,
            hub_manager=hub_manager,
            compile_model=compile,
        )

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=lora_target_modules,
        )

        self._tokenizer = self.loader.load_tokenizer(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            chat_template=chat_template,
        )
        self._models: list[PeftModel] | None = None
        self._reward_heads: list[RewardHead] | None = None
        self.epochs_per_model: list[Optional[int]] = [0] * self._num_models

        logger.info(
            f"REPPORewardModel initialized with L={self._num_models}, "
            f"model_id={self.model_id}"
        )

    @property
    def reward_heads(self) -> list[RewardHead]:
        """List of reward heads. Raises if not loaded."""
        if self._reward_heads is None:
            raise RuntimeError("Reward heads not loaded. Call load() first.")
        if self._reward_heads is None:
            raise RuntimeError("Reward heads not loaded. Call load() first.")
        return self._reward_heads

    @property
    def num_models(self) -> int:
        """Number of reward models in the ensemble."""
        return self._num_models

    @property
    def device_manager(self) -> DeviceManager:
        return self._device_manager

    @property
    def hub_manager(self) -> HubManager:
        return self._hub_manager

    def init_trainer(self) -> None:
        """Initialize the trainer if it's a partial."""
        # if isinstance(self._trainer, functools.partial):
        #     self._trainer = self._trainer()
        pass

    def load(self, init_new: bool = False, epoch: Optional[int] = None) -> None:
        """Load models and reward heads into memory."""
        if self._models is not None:
            logger.warning("Models already loaded. Unload first to reload.")
            return

        self._models = []
        for model_idx in range(self._num_models):
            self._models.append(
                self.loader.load_model(
                    model_id=self.model_id,
                    model_name=self.get_submodel_name(model_idx),
                    model_idx=model_idx,
                    lora_config=self.lora_config,
                    init_new=init_new,
                    epoch=epoch,
                )
            )

        # Initialize reward heads with same dtype as model
        hidden_size = self._models[0].config.hidden_size
        model_dtype = next(self._models[0].parameters()).dtype
        self._reward_heads = []

        for model_idx in range(self._num_models):
            device = torch.device(self._device_manager.get_device_for_model(model_idx))
            reward_head = RewardHead(hidden_size).to(device=device, dtype=model_dtype)
            self._reward_heads.append(reward_head)

        if epoch is not None:
            self.epochs_per_model = [epoch] * self._num_models

        logger.info(f"Loaded {self._num_models} reward models with heads")

    def unload(self) -> None:
        """Unload all models and reward heads from GPU memory."""
        if not self._models:
            logger.info("Models already unloaded")
            return

        for model in self._models:
            del model
        if self._reward_heads:
            for head in self._reward_heads:
                del head

        self._models = None
        self._reward_heads = None
        self._device_manager.clear_cache()
        self.epochs_per_model = [0] * self._num_models
        logger.info("All reward models unloaded")

    @property
    def models(self) -> list[PeftModel]:
        if self._models is None:
            raise RuntimeError("Models not loaded. Call load() first.")
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
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        return reward_head(hidden_states, attention_mask)  # type: ignore[no-any-return]

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bradley-Terry loss for preference learning."""
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
        return loss, chosen_rewards, rejected_rewards

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute rewards using device-resident tensors (pessimistic aggregation)."""
        if self._models is None or self._reward_heads is None:
            raise RuntimeError("Models not loaded. Call load() first.")

        rewards_ensemble: list[Optional[torch.Tensor]] = [None] * self._num_models
        thread_exceptions: list[Optional[BaseException]] = [None] * self._num_models

        def predict_reward(
            model_idx: int,
            model: PeftModel,
            reward_head: RewardHead,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> None:
            try:
                device = input_ids.device
                with torch.no_grad():
                    model.eval()
                    rewards = self._compute_reward(
                        input_ids, attention_mask, model, reward_head, device
                    )
                    rewards_ensemble[model_idx] = rewards
            except BaseException as e:
                thread_exceptions[model_idx] = e

        threads = []
        for model_idx in range(self._num_models):
            thread = threading.Thread(
                target=predict_reward,
                args=(
                    model_idx,
                    self.models[model_idx],
                    self.reward_heads[model_idx],
                    device_input_ids[model_idx],
                    device_attention_masks[model_idx],
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for model_idx, exc in enumerate(thread_exceptions):
            if exc is not None:
                raise RuntimeError(f"Exception in reward model {model_idx}") from exc

        rewards_filtered = [r.cpu() for r in rewards_ensemble if r is not None]
        if len(rewards_filtered) == 1:
            return rewards_filtered[0]

        rewards_tensor = torch.stack(rewards_filtered, dim=0)
        return rewards_tensor.min(dim=0).values

    def get_name(self, epoch: Optional[int] = None) -> str:
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-reppo-a0.0-b0.0-L{self._num_models}"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def get_submodel_name(self, model_idx: int) -> str:
        return f"{self.get_name()}-r{model_idx}"

    def can_load_from_epoch(self, epoch: int) -> bool:
        for model_idx in range(self._num_models):
            submodel_name = self.get_submodel_name(model_idx)
            if not self._hub_manager.model_exists(submodel_name, epoch):
                return False
        return True

    def load_from_epoch(self, epoch: int) -> None:
        if self._models is not None:
            self.unload()
        self.load(init_new=False, epoch=epoch)

    def train(
        self,
        data_manager: Any,
        max_epochs: int,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """Train the reward model ensemble."""
        raise NotImplementedError("REPPORewardModel training is disabled for now.")


class REPPOModel(BaseModel):
    """REPPO Policy Model (Orchestrator).

    This model acts as the Policy Model for RLHF, but also manages a
    helper REPPORewardModel ensemble for reward calculation.

    Training Phases:
    1. Reward Training: Train the helper reward_model (via trainer orchestration).
    2. Policy Training: Train self (policy) using rewards from helper.
    """

    def __init__(
        self,
        model_id: str,
        reward_model: DictConfig,
        device_manager: DeviceManager,
        hub_manager: HubManager,
        kl_coef: float = 0.1,
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        lora_target_modules: str = "all-linear",
        compile: bool = False,
        generator: Optional[Any] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize REPPO Model (Policy)."""
        self.model_id = model_id
        self._device_manager = device_manager
        self._hub_manager = hub_manager
        self.kl_coef = kl_coef
        self.tokenizer_id = tokenizer_id
        self.chat_template = chat_template
        self.generator = generator
        self.debug = debug
        self._num_models = 1

        # Instantiate Helper Reward Model
        logger.info("Instantiating REPPO reward model helper...")
        self.reward_model = instantiate(
            reward_model,
            device_manager=device_manager,
            hub_manager=hub_manager,
        )

        # Initialize Loader for Policy
        self.loader = ModelLoader(
            device_manager=device_manager,
            hub_manager=hub_manager,
            compile_model=compile,
        )

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=lora_target_modules,
        )

        self._tokenizer = self.loader.load_tokenizer(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            chat_template=chat_template,
        )
        self._models: list[PeftModel] | None = None
        self._ref_policy: PeftModel | None = None
        self.epochs_per_model: list[Optional[int]] = [0]

        logger.info(
            f"REPPOModel initialized as Policy ({self.model_id}) "
            f"with helper Reward Model (L={self.reward_model._num_models})"
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def device_manager(self) -> DeviceManager:
        return self._device_manager

    @property
    def hub_manager(self) -> HubManager:
        return self._hub_manager

    @property
    def policy(self) -> PeftModel:
        """Alias for the single policy model."""
        if self._models is None or len(self._models) == 0:
            raise RuntimeError("Policy not loaded. Call load() first.")
        return self._models[0]

    def load(self, init_new: bool = False, epoch: Optional[int] = None) -> None:
        """Load Policy Model (self) AND Reward Model (helper)."""
        # Load Reward Model Helper
        self.reward_model.load(init_new=init_new, epoch=epoch)

        # Load Policy (Self)
        if self._models is not None:
            logger.warning("Policy already loaded. Unload first to reload.")
            # We allow proceeding, similar to reload
        else:
            model = self.loader.load_model(
                model_id=self.model_id,
                model_name=self.get_submodel_name(0),
                model_idx=0,
                lora_config=self.lora_config,
                init_new=init_new,
                epoch=epoch,
            )
            self._models = [model]
            if epoch is not None:
                self.epochs_per_model = [epoch]
            logger.info("Loaded REPPO Policy model")

    def unload(self) -> None:
        """Unload Policy Model (self) AND Reward Model (helper)."""
        self.reward_model.unload()

        if not self._models:
            return
        for model in self._models:
            del model
        self._models = None
        self._ref_policy = None
        self._device_manager.clear_cache()
        self.epochs_per_model = [0]
        logger.info("REPPO Policy model unloaded")

    def get_name(self, epoch: Optional[int] = None) -> str:
        """Get policy model name (formerly REPPOPolicyModel.get_name)."""
        model_name = self.model_id.rsplit("/", 1)[-1]
        repo_name = f"{model_name}-reppo-a0.0-b0.0-L1"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def get_submodel_name(self, model_idx: int) -> str:
        return f"{self.get_name()}-policy"

    def can_load_from_epoch(self, epoch: int) -> bool:
        return self._hub_manager.model_exists(self.get_submodel_name(0), epoch)

    def load_from_epoch(self, epoch: int) -> None:
        if self._models is not None:
            # Partial unload if needed, or just rely on load handling it
            pass
        self.load(init_new=False, epoch=epoch)

    def train(
        self,
        data_manager: Any,
        max_epochs: int,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """Execute orchestrated training via configured trainer."""
        raise NotImplementedError("REPPOModel training is disabled for now.")

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Compute Policy Loss (PPO/RLOO/etc)."""
        raise NotImplementedError("Policy loss implementation pending.")

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Policy prediction (generation or log probs)."""
        raise NotImplementedError("Policy prediction implementation pending.")
