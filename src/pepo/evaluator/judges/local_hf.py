from __future__ import annotations

import gc
import logging
from typing import Any, Optional, Sequence, cast

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseJudge, JudgePrompt

logger = logging.getLogger(__name__)


class LocalHFJudge(BaseJudge):
    """Local HuggingFace causal LM judge.

    The model is loaded lazily so evaluators can finish policy generation and
    unload the policy before a large judge model is brought into memory.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        model_kwargs: Optional[dict[str, Any] | DictConfig] = None,
        tokenizer_kwargs: Optional[dict[str, Any] | DictConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_kwargs = self._normalize_model_kwargs(model_kwargs)
        self.tokenizer_kwargs = self._to_plain_dict(tokenizer_kwargs)
        self._model: Any = None
        self._tokenizer: Any = None

    def generate(self, prompts: Sequence[JudgePrompt]) -> list[str]:
        if not prompts:
            return []

        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model

        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            formatted_prompts = [self._format_prompt(prompt) for prompt in batch]
            tokenizer.padding_side = "left"
            tokenized = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            input_length = input_ids.shape[1]
            input_device = self._get_input_device(model)
            tokenized = {
                "input_ids": input_ids.to(input_device),
                "attention_mask": attention_mask.to(input_device),
            }

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = self.top_p

            with torch.inference_mode():
                generated = model.generate(**tokenized, **generation_kwargs)

            generated_suffix = generated[:, input_length:]
            outputs.extend(
                tokenizer.batch_decode(generated_suffix, skip_special_tokens=True)
            )

        return [output.strip() for output in outputs]

    def unload(self) -> None:
        if self._model is not None:
            logger.info("Unloading local judge model: %s", self.model_name)
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading local judge model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            **self.tokenizer_kwargs,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.model_kwargs,
        ).eval()

    def _format_prompt(self, prompt: JudgePrompt) -> str:
        tokenizer = self._tokenizer
        messages = []
        if prompt.system_prompt:
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.user_prompt})

        try:
            return cast(
                str,
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
        except (ValueError, AttributeError):
            return self._fallback_prompt(prompt)

    @staticmethod
    def _fallback_prompt(prompt: JudgePrompt) -> str:
        if prompt.system_prompt:
            return (
                f"{prompt.system_prompt}\n\n"
                f"User: {prompt.user_prompt}\n"
                "Assistant:"
            )
        return f"User: {prompt.user_prompt}\nAssistant:"

    @staticmethod
    def _get_input_device(model: Any) -> torch.device:
        model_device = getattr(model, "device", None)
        if isinstance(model_device, torch.device):
            return model_device
        return next(model.parameters()).device

    @classmethod
    def _normalize_model_kwargs(
        cls,
        model_kwargs: Optional[dict[str, Any] | DictConfig],
    ) -> dict[str, Any]:
        normalized = cls._to_plain_dict(model_kwargs)
        for key in ("dtype", "torch_dtype"):
            if key in normalized and isinstance(normalized[key], str):
                normalized[key] = getattr(torch, normalized[key])
        return normalized

    @staticmethod
    def _to_plain_dict(
        value: Optional[dict[str, Any] | DictConfig],
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, DictConfig):
            converted = OmegaConf.to_container(value, resolve=True)
            if not isinstance(converted, dict):
                raise ValueError("Judge config must be a mapping")
            return dict(converted)
        return dict(value)
