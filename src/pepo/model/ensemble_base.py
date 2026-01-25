"""Base class for ensemble models with multiple submodels."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import torch
from peft import PeftModel
from tqdm import tqdm

from ..utils.model_utils import get_next_token_log_probs, top_p_sample
from .base import BaseModel

if TYPE_CHECKING:
    pass

__all__ = ["EnsembleModel"]


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

    def to(self, device: torch.device) -> None:
        """Move all models to device (only iff shared backbone)."""
        if not self.shared_backbone:
            # If not shared, we'd need to move N huge models. Dangerous default.
            # User should manage manually if they really want that.
            raise NotImplementedError(
                "EnsembleModel.to() is only supported for shared backbone models. "
                "For independent models, manage devices manually."
            )

        # Move shared backbone (model 0 holds it)
        self.models[0].to(device)

    def cpu(self) -> None:
        """Move all models to CPU."""
        if self.is_loaded():
            for m in self.models:
                m.cpu()

    def clone(self) -> "EnsembleModel":
        """Create a deep copy of the ensemble model."""
        import copy

        # Shallow copy self to preserve managers
        new_model = copy.copy(self)

        # Deep copy the list of models (PeftModels)
        if self.is_loaded():
            new_model.models = [copy.deepcopy(m) for m in self.models]

        return new_model

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
        model_indices: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Single-token prediction for ensemble using shared backbone.

        Uses adapter switching to get predictions from each submodel sequentially,
        then returns the minimum log probabilities across specified models.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            device: Device the model is on.
            past_key_values: List of past_key_values for each model.
            use_cache: Whether to use KV caching.
            model_indices: Which models to use. None = all models (pessimistic).
                [0] = first model only (proposal mode).

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

        # Determine which models to use
        indices = (
            model_indices
            if model_indices is not None
            else list(range(self._num_models))
        )

        with torch.no_grad():
            log_probs_list = []
            inp = input_ids.to(device)
            mask = attention_mask.to(device)

            for model_idx in indices:
                adapter_name = "default" if model_idx == 0 else f"adapter_{model_idx}"
                if model.active_adapter != adapter_name:
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
                log_probs_list.append(log_probs)
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
        model_indices: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens using ensemble prediction. Assumes model is on device.

        Caller is responsible for moving the model to GPU before calling
        and moving it back to CPU after.

        Args:
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T) on CPU.
            max_new_tokens: Maximum tokens to generate.
            greedy_sampling: If True, use argmax. If False, use top-p.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling threshold.
            token_callback: Optional callback for streaming tokens.
            use_cache: Whether to use KV caching.
            model_indices: Which models to use. None = all models (pessimistic).
                [0] = first model only (proposal mode).

        Returns:
            Tuple of (output_ids, output_mask) on CPU.
        """

        batch_size = input_ids.shape[0]
        stop_signal = torch.zeros(batch_size, dtype=torch.bool)
        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        past_key_values_list: list[Any] = [None] * self._num_models

        if self.shared_backbone:
            # Get device from where the model currently lives
            device = next(self.models[0].parameters()).device

            pbar = tqdm(range(max_new_tokens), disable=disable_tqdm, leave=False)
            for i in pbar:
                if i > 0 and i % 100 == 0:
                    self._device_manager.clear_cache()

                log_probs, past_key_values_list = self.predict(
                    input_ids,
                    attention_mask,
                    device,
                    past_key_values_list,
                    use_cache=use_cache,
                    model_indices=model_indices,
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

        sampled_token_ids = sampled_token_ids.cpu()

        # Replace sampled tokens with pad_token_id for already-stopped sequences
        sampled_token_ids = torch.where(
            stop_signal,
            torch.full_like(sampled_token_ids, self._tokenizer.pad_token_id),
            sampled_token_ids,
        )

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
