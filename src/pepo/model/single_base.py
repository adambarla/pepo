"""Base class for single-model architectures."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from peft import PeftModel
from tqdm import tqdm

from ..utils.model_utils import get_next_token_log_probs, top_p_sample
from .base import BaseModel

if TYPE_CHECKING:
    pass

__all__ = ["SingleModel"]


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

    @property
    def num_models(self) -> int:
        """Single model always has 1 model."""
        return 1

    def set_epoch(self, epoch: int, model_idx: Optional[int] = None) -> None:
        """Set trained epoch (model_idx ignored for single model)."""
        self._epoch = epoch

    def get_epoch(self, model_idx: int = 0) -> int:
        """Get trained epoch (model_idx ignored for single model)."""
        return self._epoch

    def can_load_from_epoch(self, epoch: int) -> bool:
        """Check if model has checkpoint at the specified epoch."""
        return self.hub_manager.model_exists(self.get_name(), epoch)

    def to(self, device: torch.device) -> None:
        """Move model to device."""
        self.model.to(device)

    def cpu(self) -> None:
        """Move model to CPU."""
        if self.is_loaded():
            self.model.cpu()

    def clone(self) -> "SingleModel":
        """Create a deep copy of the model."""
        import copy

        new_model = copy.copy(self)
        if self.is_loaded():
            new_model.model = copy.deepcopy(self.model)
        return new_model

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
        """Generate tokens. Assumes model is already on the target device.

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

        Returns:
            Tuple of (output_ids, output_mask) on CPU.
        """
        batch_size = input_ids.shape[0]
        stop_signal = torch.zeros(batch_size, dtype=torch.bool)
        past_key_values = None

        # Get device from where the model currently lives
        device = next(self.model.parameters()).device

        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        pbar = tqdm(range(max_new_tokens), disable=disable_tqdm, leave=False)
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

            # Replace tokens with pad_token_id for stopped sequences
            sampled_token_ids = torch.where(
                stop_signal,
                torch.full_like(sampled_token_ids, self._tokenizer.pad_token_id),
                sampled_token_ids,
            )

            stop_signal = stop_signal | (
                sampled_token_ids == self._tokenizer.eos_token_id
            )

            input_ids = torch.cat([input_ids, sampled_token_ids.unsqueeze(-1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, (~stop_signal).unsqueeze(-1).float()], dim=1
            )

            pbar.set_postfix({"stopped": f"{stop_signal.sum().item()}/{batch_size}"})

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

        return input_ids, attention_mask
