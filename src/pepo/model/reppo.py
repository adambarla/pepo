"""REPPO (Reward Ensemble Pessimistic Preference Optimization) Model.

This module contains all REPPO model classes:
- RewardHead: Linear projection for scalar reward
- REPPORewardModel: Ensemble of L reward models
- REPPOPolicyModel: Single policy model (placeholder loss)
- REPPOModel: Orchestrator for two-phase RLHF training
"""

import functools
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..trainer import BaseTrainer

import copy
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ..loader import CheckpointManager
from ..utils import DeviceManager, HubManager
from ..utils.data import DataCollator
from .base import BaseModel

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


class REPPORewardModel(BaseModel):
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

        self._checkpoint_manager = CheckpointManager(
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
            modules_to_save=["reward_head"],
        )

        self._tokenizer = self.checkpoint_manager.load_tokenizer(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            chat_template=chat_template,
        )
        self._models: list[PeftModel] | None = None
        self.epochs_per_model: list[Optional[int]] = [0] * self._num_models

        logger.info(
            f"REPPORewardModel initialized with L={self._num_models}, "
            f"model_id={self.model_id}"
        )

    @property
    def reward_heads(self) -> list[RewardHead]:
        """List of reward heads (extracted from PeftModels)."""
        if self._models is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        heads = []
        for model in self._models:
            if not hasattr(model, "reward_head"):
                raise AttributeError("Model missing 'reward_head' attribute")
            heads.append(model.reward_head)
        return heads

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

        # We need to know hidden_size to init reward heads if init_new=True
        # For simplicity, we load the first model to get the config
        # This is a bit redundant but ensures we have the right hidden_size
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model_id)
        hidden_size = config.hidden_size
        model_dtype = self._device_manager.dtype
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)

        self._models = []
        for model_idx in range(self._num_models):
            device = torch.device(self._device_manager.get_device_for_model(model_idx))
            # Always create a head; if loading from hub, PEFT will overwrite its weights
            reward_head = RewardHead(hidden_size).to(device=device, dtype=model_dtype)

            self._models.append(
                self.checkpoint_manager.load_model(
                    model_id=self.model_id,
                    model_name=self.get_name(model_idx=model_idx),
                    model_idx=model_idx,
                    lora_config=self.lora_config,
                    init_new=init_new,
                    epoch=epoch,
                    custom_modules={"reward_head": reward_head},
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

        for model in self._models:
            del model

        self._models = None
        self._device_manager.clear_cache()
        self.epochs_per_model = [0] * self._num_models
        logger.info("All submodels unloaded from GPU memory")

    def save(self) -> None:
        """Save all ensemble models to Hub."""
        for i in range(self._num_models):
            self._push_model(i)

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch for a model."""
        if model_idx is not None:
            self.epochs_per_model[model_idx] = epoch
        else:
            self.epochs_per_model = [epoch] * self._num_models

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch for a model."""
        return self.epochs_per_model[model_idx] or 0

    def _push_model(self, model_idx: int, epochs: Optional[int] = None) -> None:
        """Push single model to hub."""
        self.checkpoint_manager.push_model(
            model=self.models[model_idx],
            model_name=self.get_name(model_idx=model_idx),
            tokenizer=self.tokenizer,
            epochs=epochs,
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
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

    def _get_base_model_name(self) -> str:
        return self.model_id.rsplit("/", 1)[-1]

    def can_load_from_epoch(self, epoch: int) -> bool:
        for model_idx in range(self._num_models):
            submodel_name = self.get_name(model_idx=model_idx)
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

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Inference prediction (unused for reward model training)."""
        raise NotImplementedError("Predict not implemented for Reward Model")

    def annotate_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        semaphore_limit: Optional[int] = None,
        batch_size: int = 8,
    ) -> Any:
        """
        Annotate dataset with rewards from all ensemble models.

        Uses parallel threads (up to semaphore_limit or num_gpus) to run inference.
        Updates the dataset gradually as models finish.

        Args:
            dataset: The dataset to annotate (HuggingFace Dataset).
            dataset_name: Name for pushing to Hub.
            semaphore_limit: Max concurrent threads (defaults to num_gpus).
            batch_size: Batch size for inference.

        Returns:
            The annotated dataset.
        """
        if self._hub_manager.dataset_exists(dataset_name):
            logger.info(
                f"Dataset {dataset_name} already exists on Hub. Skipping annotation."
            )
            # We return the original dataset; robust training might fetch from Hub
            # independently or we could return None to signal "done".
            return dataset

        if self._models is None:
            raise RuntimeError("Models not loaded. Call load() first.")

        limit = semaphore_limit or self._device_manager.num_available_gpus
        gpu_semaphore = threading.Semaphore(value=limit)
        dataset_lock = threading.Lock()

        logger.info(f"Starting dataset annotation with {limit} concurrent threads...")

        threads = []
        # We need a shared reference to the dataset that threads can update
        # Using a container list to allow 'pass-by-reference' update of the
        # dataset object safely within the lock if the library returns a new
        # object on add_column
        shared_dataset = [dataset]

        def _annotate_single_model(model_idx: int) -> None:
            gpu_semaphore.acquire()
            try:
                # 1. Load Model
                device = torch.device(
                    self._device_manager.get_device_for_model(model_idx)
                )
                # Ensure model is on correct device (it should be if loaded via load())
                # Explicit check for mypy
                if self._models is None:
                    raise RuntimeError("Models not loaded.")
                model = self._models[model_idx]
                reward_head = self.reward_heads[model_idx]

                # 2. Prepare DataLoader
                tokenizer_copy = copy.deepcopy(self.tokenizer)
                collator = DataCollator(
                    tokenizer=tokenizer_copy,
                    # We assume max_length etc are handled by the dataset's
                    # preprocessing or passed in logic.
                    max_length=None,
                    max_prompt_length=None,
                )

                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    pin_memory=True,
                )

                rewards_chosen = []
                rewards_rejected = []

                desc = f"Model {model_idx} Inference"
                for batch in tqdm(
                    dataloader, desc=desc, position=model_idx, leave=False
                ):
                    with torch.no_grad():
                        chosen_ids = batch["chosen_input_ids"].to(device)
                        chosen_mask = batch["chosen_attention_mask"].to(device)
                        rejected_ids = batch["rejected_input_ids"].to(device)
                        rejected_mask = batch["rejected_attention_mask"].to(device)

                        r_c = self._compute_reward(
                            chosen_ids, chosen_mask, model, reward_head, device
                        )
                        r_r = self._compute_reward(
                            rejected_ids, rejected_mask, model, reward_head, device
                        )

                        rewards_chosen.extend(r_c.cpu().tolist())
                        rewards_rejected.extend(r_r.cpu().tolist())

                # 3. Critical Section: Update Dataset
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

            except Exception as e:
                logger.error(f"Error in annotation thread for model {model_idx}: {e}")
            finally:
                gpu_semaphore.release()

        # Launch threads
        for i in range(self.num_models):
            t = threading.Thread(target=_annotate_single_model, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        final_dataset = shared_dataset[0]

        # Persistence
        logger.info(f"Pushing annotated dataset to {dataset_name}...")
        self._hub_manager.push_dataset(final_dataset, dataset_name)

        return final_dataset


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
        kl_coef: float = 0.1,  # Unused - for Phase 2
        tokenizer_id: Optional[str] = None,
        chat_template: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        lora_target_modules: str = "all-linear",
        compile: bool = False,
        generator: Optional[Any] = None,  # Unused - for Phase 2
        reward_trainer: Optional[DictConfig] = None,
        policy_trainer: Optional[DictConfig] = None,
        trainer: Optional[DictConfig] = None,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize REPPO Model (Policy)."""
        self.model_id = model_id
        self._device_manager = device_manager
        self._hub_manager = hub_manager
        # Unused Phase 2 attributes (kept for config compatibility)
        self.generator = generator
        self.kl_coef = kl_coef
        self.debug = debug
        self._num_models = 1
        self._reward_trainer_cfg = reward_trainer
        # Support both policy_trainer and trainer (preferred) key
        self._policy_trainer_cfg = policy_trainer or trainer
        self._reward_trainer: Optional[BaseTrainer] = None
        self._policy_trainer: Optional[BaseTrainer] = None
        # We set self._trainer to policy config to satisfy generic checks relying
        # on model.trainer
        self._trainer: Any = self._policy_trainer_cfg

        self.reward_model = instantiate(
            reward_model,
            device_manager=device_manager,
            hub_manager=hub_manager,
        )

        self._checkpoint_manager = CheckpointManager(
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

        self._tokenizer = self.checkpoint_manager.load_tokenizer(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            chat_template=chat_template,
        )
        self._models: list[PeftModel] | None = None
        self._epoch: int = 0

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
            model = self.checkpoint_manager.load_model(
                model_id=self.model_id,
                model_name=self.get_name(),
                model_idx=0,
                lora_config=self.lora_config,
                init_new=init_new,
                epoch=epoch,
            )
            self._models = [model]
            if epoch is not None:
                self._epoch = epoch
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
        # L refers to the number of reward models, not policy models
        repo_name = f"{model_name}-reppo-L{self.reward_model._num_models}-policy"
        if epoch is not None:
            repo_name = f"{repo_name}-e{epoch}"
        return repo_name

    def _get_base_model_name(self) -> str:
        return self.model_id.rsplit("/", 1)[-1]

    def can_load_from_epoch(self, epoch: int) -> bool:
        return self._hub_manager.model_exists(self.get_name(), epoch)

    def load_from_epoch(self, epoch: int) -> None:
        if self._models is not None:
            # Partial unload if needed, or just rely on load handling it
            pass
        self.load(init_new=False, epoch=epoch)

    def save(self) -> None:
        """Save policy model to Hub."""
        self.checkpoint_manager.push_model(
            model=self.policy,
            model_name=self.get_name(),
            tokenizer=self.tokenizer,
            epochs=self._epoch,
        )

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch for policy (model_idx ignored)."""
        self._epoch = epoch

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch for policy (model_idx ignored)."""
        return self._epoch

    def train(
        self,
        data_manager: Any,
        max_epochs: int,
        wandb_manager: Optional[Any] = None,
        continue_training: bool = False,
    ) -> None:
        """Execute orchestrated training: Reward Ensemble -> Policy."""
        if self._reward_trainer_cfg is None or self._policy_trainer_cfg is None:
            raise ValueError("Trainers not configured in model config.")

        # Phase 1: Reward Training
        logger.info("--- Phase 1: Training Reward Ensemble ---")
        # Determine reward epochs first
        reward_epochs = self._reward_trainer_cfg.get("training_epochs", max_epochs)

        if self.reward_model.can_load_from_epoch(reward_epochs):
            logger.info(
                f"Reward models for epoch {reward_epochs} already exist. "
                "Skipping training and loading from Hub."
            )
            self.reward_model.load_from_epoch(reward_epochs)
        else:
            if self._reward_trainer is None:
                # Pop epochs from config to avoid passing it to trainer init
                cfg = self._reward_trainer_cfg.copy()
                if "training_epochs" in cfg:
                    del cfg["training_epochs"]

                trainer_instance = instantiate(cfg)
                if isinstance(trainer_instance, functools.partial):
                    trainer_instance = trainer_instance()
                self._reward_trainer = trainer_instance

            if self._reward_trainer is None:
                raise ValueError("Failed to initialize reward trainer")

            self._reward_trainer.train(
                model=self.reward_model,
                data_manager=data_manager,
                max_epochs=reward_epochs,
                wandb_manager=wandb_manager,
                continue_training=continue_training,
            )

        # Inter-Phase: Annotate Dataset
        logger.info("--- Phase 1 Complete. Starting Annotation ---")

        # We annotate the dataset corresponding to the trained state
        # We assume all models are at the same epoch after training
        annotated_epoch = self.reward_model.get_epoch(0)
        annotation_dataset_name = self.reward_model.get_name(epoch=annotated_epoch)

        # For now, we annotate the evaluation dataset provided by DataManager
        # TODO: Allow configuring which dataset split to annotate
        # (e.g. 'train' or 'unlabeled')
        target_dataset = data_manager.eval_dataset

        if target_dataset is not None:
            self.reward_model.annotate_dataset(
                dataset=target_dataset, dataset_name=annotation_dataset_name
            )
        else:
            logger.warning("No evaluation dataset found in DataManager to annotate.")

        # Phase 2: Policy Training (Not Implemented)
        logger.info("--- Phase 2: Training Policy Model ---")
        raise NotImplementedError(
            "Phase 2 (Policy Training) is not implemented yet. "
            "Only Phase 1 (Reward Ensemble Training) is currently supported."
        )

    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss (delegated to phase-specific models)."""
        raise NotImplementedError("REPPOModel loss_fn should not be called directly.")

    def predict(
        self,
        device_input_ids: list[torch.Tensor],
        device_attention_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Inference prediction."""
        raise NotImplementedError("REPPOModel predict not implemented yet.")
