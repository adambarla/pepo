"""Abstract base class for policy models."""

from __future__ import annotations

import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ..utils.model_utils import get_next_token_log_probs, top_p_sample

if TYPE_CHECKING:
    from ..generator import Generator
    from ..loader import CheckpointManager
    from ..trainer import BaseTrainer
    from ..utils import DeviceManager, HubManager

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for policy models (DEPPO, REPPO)."""

    def __init__(
        self,
        model_id: str,
        device_manager: "DeviceManager",
        hub_manager: "HubManager",
        checkpoint_manager: "CheckpointManager",
        tokenizer: PreTrainedTokenizerBase,
        trainer: Optional["BaseTrainer"] = None,
        generator: Optional["Generator"] = None,
    ) -> None:
        """Initialize base model attributes.

        Args:
            model_id: HuggingFace model ID.
            device_manager: Device manager for GPU allocation.
            hub_manager: Hub manager for model storage.
            checkpoint_manager: Checkpoint manager for saving/loading.
            tokenizer: Tokenizer for the model.
            trainer: Optional trainer instance.
            generator: Optional generator instance.
        """
        self.model_id = model_id
        self._device_manager = device_manager
        self._hub_manager = hub_manager
        self._checkpoint_manager = checkpoint_manager
        self._tokenizer = tokenizer
        self._trainer = trainer
        self.generator = generator

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer for the model."""
        return self._tokenizer

    @property
    def device_manager(self) -> "DeviceManager":
        """Device manager for GPU allocation."""
        return self._device_manager

    @property
    def hub_manager(self) -> "HubManager":
        """Hub manager for model storage."""
        return self._hub_manager

    @property
    def checkpoint_manager(self) -> "CheckpointManager":
        """Checkpoint manager for saving/loading models."""
        return self._checkpoint_manager

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if models are loaded."""

    @abstractmethod
    def loss_fn(
        self,
        batch: dict[str, torch.Tensor],
        model: PeftModel,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss for training.

        Args:
            batch: Training batch with input tensors.
            model: The submodel to compute loss for.
            device: Device to run computation on.

        Returns:
            Tuple of (loss, metrics_dict).
        """

    @abstractmethod
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[Any]]:
        """Single-token prediction.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            device: Device the model is on.
            past_key_values: Optional key values for caching.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (Log probs for next token (B, V) on CPU, past_key_values).
        """

    @abstractmethod
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            max_new_tokens: Maximum tokens to generate.
            greedy_sampling: If True, use argmax. If False, use top-p.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling threshold.
            token_callback: Optional callback for streaming tokens.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (output_ids, output_mask) on CPU.
        """

    @abstractmethod
    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if model can be loaded from epoch."""

    @abstractmethod
    def save(self) -> None:
        """Save all models to storage."""

    @abstractmethod
    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch. For ensembles, model_idx specifies which model."""

    @abstractmethod
    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch. For ensembles, model_idx specifies which model."""

    def find_latest_epoch(self, max_epoch: int) -> Optional[int]:
        """
        Find the latest epoch where all submodels have checkpoints.

        Args:
            max_epoch: Maximum epoch to check from (checks backwards).

        Returns:
            The latest epoch where all submodels exist, or None if none found.
        """
        for epoch in range(max_epoch, -1, -1):
            if self.can_load_from_epoch(epoch):
                logger.info(f"Found latest common epoch {epoch} for model")
                return epoch

        logger.warning(
            f"No common epoch found (checked from epoch {max_epoch} down to 0)"
        )
        return None

    @abstractmethod
    def load(
        self, init_new: bool = False, epoch: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Load models into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload models from memory."""

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer."""
        return self.tokenizer

    @abstractmethod
    def get_name(
        self,
        *,
        epoch: Optional[int] = None,
        model_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Get model name for identification.

        Args:
            epoch: Optional epoch number to include in name.
            model_idx: Optional model index for ensemble submodels.
            **kwargs: Additional optional arguments.
        """

    def _get_base_model_name(self) -> str:
        """Get the base model name (e.g. from model_id)."""
        return self.model_id.rsplit("/", 1)[-1]

    def init_trainer(self) -> None:
        """Initialize the trainer if it's a partial."""
        if isinstance(self._trainer, functools.partial):
            self._trainer = self._trainer()

    @property
    def trainer(self) -> Optional["BaseTrainer"]:
        """Get the trainer instance."""
        if isinstance(self._trainer, functools.partial):
            self.init_trainer()
        return self._trainer

    @trainer.setter
    def trainer(self, value: Optional["BaseTrainer"]) -> None:
        """Set the trainer instance."""
        self._trainer = value

    @abstractmethod
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
                Defaults to trainer config.
            wandb_manager: Optional WandbManager instance for logging.
            continue_training: Whether to continue from checkpoint.
        """

    def generate_responses(
        self,
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> list[dict[str, Any]]:
        """Generate responses for a list of prompts.

        Args:
            prompts: List of prompt strings or histories.
            apply_chat_template: Whether to apply chat template.
            token_callback: Optional callback for streaming tokens.

        Returns:
            List of dicts with 'prompt' and 'output' keys.
        """
        if self.generator is None:
            raise ValueError(
                "Generator not set on model. Set model.generator before "
                "calling generate_responses()."
            )
        return self.generator.generate_responses(
            model=self,
            prompts=prompts,
            apply_chat_template=apply_chat_template,
            token_callback=token_callback,
        )


class EnsembleModel(BaseModel):
    """Base class for ensemble models with multiple submodels (DEPPO, REPPOReward)."""

    def __init__(
        self,
        num_models: int,
        **kwargs: Any,
    ) -> None:
        """Initialize ensemble model.

        Args:
            num_models: Number of models in the ensemble.
            **kwargs: Arguments passed to BaseModel.__init__.
        """
        super().__init__(**kwargs)
        self._num_models = num_models
        self._models: Optional[list[PeftModel]] = None
        self.epochs_per_model: list[Optional[int]] = [0] * num_models

    @property
    def num_models(self) -> int:
        """Number of models in the ensemble."""
        return self._num_models

    @property
    def models(self) -> list[PeftModel]:
        """List of loaded models. Raises if not loaded."""
        if not self.is_loaded():
            raise RuntimeError(
                "Models not loaded. Call load() before accessing models."
            )
        assert self._models is not None  # for type narrowing
        return self._models

    @models.setter
    def models(self, value: list[PeftModel]) -> None:
        """Set the models list."""
        self._models = value

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models is not None

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch for a model."""
        if model_idx is not None:
            self.epochs_per_model[model_idx] = epoch
        else:
            self.epochs_per_model = [epoch] * self._num_models

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch for a model."""
        return self.epochs_per_model[model_idx] or 0

    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if all submodels have checkpoints at the specified epoch."""
        for model_idx in range(self._num_models):
            submodel_name = self.get_name(model_idx=model_idx)
            if not self.hub_manager.model_exists(submodel_name, epoch):
                return False
        return True

    def save(self) -> None:
        """Save all ensemble models to Hub."""
        for i in range(len(self.models)):
            self.checkpoint_manager.push_model(
                model=self.models[i],
                model_name=self.get_name(model_idx=i),
                tokenizer=self.tokenizer,
            )

    @property
    def shared_backbone(self) -> bool:
        """Whether ensemble uses shared backbone. Override in subclass."""
        return False

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Single-token prediction for ensemble using shared backbone.

        Uses adapter switching to get predictions from each submodel sequentially,
        then returns the minimum log probabilities across all models.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            device: Device the model is on.
            past_key_values: List of past_key_values for each model.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (Min log probs (B, V) on CPU, updated past_key_values_list).
        """
        if not self.shared_backbone:
            raise NotImplementedError(
                "EnsembleModel.predict() is only supported for shared backbone models. "
                "For parallel execution, implement manual parallelization."
            )

        past_key_values_list = cast(Optional[list[Any]], past_key_values)
        model = self.models[0]
        new_past_key_values_list: list[Any] = []

        with torch.no_grad():
            log_probs_list = []
            inp = input_ids.to(device)
            mask = attention_mask.to(device)

            for model_idx in range(self._num_models):
                adapter_name = "default" if model_idx == 0 else f"adapter_{model_idx}"
                model.set_adapter(adapter_name)

                past_kv = (
                    past_key_values_list[model_idx]
                    if past_key_values_list is not None
                    else None
                )

                log_probs, new_past = get_next_token_log_probs(
                    model,
                    inp,
                    mask,
                    past_key_values=past_kv,
                    use_cache=use_cache,
                )
                log_probs_list.append(log_probs.cpu())
                new_past_key_values_list.append(new_past)

        log_probs_tensor = torch.stack(log_probs_list, dim=0)  # (L, B, V)
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        return min_log_probs, new_past_key_values_list

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens using ensemble prediction.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            max_new_tokens: Maximum tokens to generate.
            greedy_sampling: If True, use argmax. If False, use top-p.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling threshold.
            token_callback: Optional callback for streaming tokens.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (output_ids, output_mask) on CPU.
        """

        batch_size = input_ids.shape[0]
        stop_signal = torch.zeros(batch_size, dtype=torch.bool)
        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        past_key_values_list: list[Any] = [None] * self._num_models

        if self.shared_backbone:
            # Shared backbone: hold GPU for entire generation
            with self._device_manager.request_gpu() as device:
                try:
                    self.models[0].to(device)

                    pbar = tqdm(range(max_new_tokens), disable=disable_tqdm)
                    for i in pbar:
                        if i > 0 and i % 100 == 0:
                            self._device_manager.clear_cache()

                        log_probs, past_key_values_list = self.predict(
                            input_ids,
                            attention_mask,
                            device,
                            past_key_values_list,
                            use_cache=use_cache,
                        )
                        input_ids, attention_mask, stop_signal = self._generation_step(
                            log_probs,
                            input_ids,
                            attention_mask,
                            stop_signal,
                            greedy_sampling,
                            temperature,
                            top_p,
                            token_callback,
                        )
                        pbar.set_postfix(
                            {"stopped": f"{stop_signal.sum().item()}/{batch_size}"}
                        )
                        if torch.all(stop_signal):
                            break
                finally:
                    self.models[0].cpu()
                    self._device_manager.clear_cache()
        else:
            raise NotImplementedError(
                "Ensemble execution without shared backbone is not supported."
            )

        return input_ids, attention_mask

    def _generation_step(
        self,
        log_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stop_signal: torch.Tensor,
        greedy_sampling: bool,
        temperature: float,
        top_p: float,
        token_callback: Optional[Callable[[str], None]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single generation step: sample and update tensors."""
        if greedy_sampling:
            sampled_token_ids = torch.argmax(log_probs, dim=-1)
        else:
            sampled_token_ids = top_p_sample(log_probs, temperature, top_p)

        stop_signal = stop_signal | (sampled_token_ids == self._tokenizer.eos_token_id)

        input_ids = torch.cat([input_ids, sampled_token_ids.unsqueeze(-1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, (~stop_signal).unsqueeze(-1).float()], dim=1
        )

        if token_callback is not None:
            new_token_id = int(sampled_token_ids[0].item())
            if new_token_id not in [
                self._tokenizer.eos_token_id,
                self._tokenizer.pad_token_id,
            ]:
                token_text = self._tokenizer.decode([new_token_id])
                token_callback(token_text)

        return input_ids, attention_mask, stop_signal


class SingleModel(BaseModel):
    """Base class for single-model architectures (REPPOModel policy)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize single model.

        Args:
            **kwargs: Arguments passed to BaseModel.__init__.
        """
        super().__init__(**kwargs)
        self._model: Optional[PeftModel] = None
        self._epoch: int = 0

    @property
    def model(self) -> PeftModel:
        """The loaded model. Raises if not loaded."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        assert self._model is not None  # for type narrowing
        return self._model

    @model.setter
    def model(self, value: PeftModel) -> None:
        """Set the model."""
        self._model = value

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch (model_idx ignored for single model)."""
        self._epoch = epoch

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch (model_idx ignored for single model)."""
        return self._epoch

    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if model has checkpoint at the specified epoch."""
        return self.hub_manager.model_exists(self.get_name(), epoch)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        past_key_values: Optional[list[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """Single-token prediction. Assumes model already on device.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            device: Device the model is on.

        Returns:
            Log probs for next token (B, V) on CPU.
        """
        with torch.no_grad():
            self.model.eval()
            log_probs, past_key_values = get_next_token_log_probs(
                self.model,
                input_ids.to(device),
                attention_mask.to(device),
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return log_probs.cpu(), past_key_values

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens with GPU held for entire sequence.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            max_new_tokens: Maximum tokens to generate.
            greedy_sampling: If True, use argmax. If False, use top-p.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling threshold.
            token_callback: Optional callback for streaming tokens.
            use_cache: Whether to use KV caching.

        Returns:
            Tuple of (output_ids, output_mask) on CPU.
        """
        batch_size = input_ids.shape[0]
        stop_signal = torch.zeros(batch_size, dtype=torch.bool)
        past_key_values = None

        with self._device_manager.request_gpu() as device:
            try:
                self.model.to(device)

                disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
                pbar = tqdm(range(max_new_tokens), disable=disable_tqdm)
                for i in pbar:
                    if i > 0 and i % 100 == 0:
                        self._device_manager.clear_cache()

                    log_probs, past_key_values = self.predict(
                        input_ids,
                        attention_mask,
                        device,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                    if greedy_sampling:
                        sampled_token_ids = torch.argmax(log_probs, dim=-1)
                    else:
                        sampled_token_ids = top_p_sample(log_probs, temperature, top_p)

                    stop_signal = stop_signal | (
                        sampled_token_ids == self._tokenizer.eos_token_id
                    )

                    input_ids = torch.cat(
                        [input_ids, sampled_token_ids.unsqueeze(-1)], dim=1
                    )
                    attention_mask = torch.cat(
                        [attention_mask, (~stop_signal).unsqueeze(-1).float()], dim=1
                    )

                    pbar.set_postfix(
                        {"stopped": f"{stop_signal.sum().item()}/{batch_size}"}
                    )

                    if token_callback is not None:
                        new_token_id = int(sampled_token_ids[0].item())
                        if new_token_id not in [
                            self._tokenizer.eos_token_id,
                            self._tokenizer.pad_token_id,
                        ]:
                            token_text = self._tokenizer.decode([new_token_id])
                            token_callback(token_text)

                    if torch.all(stop_signal):
                        break
            finally:
                # Ensure model is moved off GPU to prevent memory accumulation
                # if it wasn't already on GPU (request_gpu implies temp slot)
                # But to be safe and fix the user issue, we move to cpu.
                self.model.cpu()
                self._device_manager.clear_cache()

        return input_ids, attention_mask
