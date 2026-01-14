"""Abstract base class for generators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from ..model import BaseModel

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class defining the generator interface."""

    def __init__(
        self,
        max_prompt_length: int = 512,
        max_new_tokens: int = 1024,
        greedy_sampling: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """
        Initialize generator.

        Args:
            max_prompt_length: Maximum length for input prompts (truncation).
            max_new_tokens: Maximum number of new tokens to generate.
            greedy_sampling: If True, use greedy (argmax). If False, use top-p.
            temperature: Sampling temperature (only used if greedy_sampling=False).
            top_p: Top-p nucleus sampling threshold
                (only used if greedy_sampling=False).
        """
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.greedy_sampling = greedy_sampling
        self.temperature = temperature
        self.top_p = top_p

    def _is_formatted(self, prompt: list[Any]) -> bool:
        for message in prompt:
            if not isinstance(message, dict):
                return False
            if "role" not in message or not isinstance(message["role"], str):
                return False
            if "content" not in message or not isinstance(message["content"], str):
                return False
        return True

    def _process_prompts(
        self,
        prompts: list[Any],
        tokenizer: "PreTrainedTokenizerBase",
        apply_chat_template: bool = True,
    ) -> tuple[list[Any], list[str]]:
        """
        Apply chat template, filter long prompts, and sort by length.

        Returns:
            Tuple of (original_prompts, formatted_prompts) sorted by length descending.
        """
        processed: list[tuple[str, str, int]] = []  # (original, formatted, length)

        for prompt in prompts:
            if apply_chat_template:
                if isinstance(prompt, list) and self._is_formatted(prompt):
                    messages = prompt
                elif isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    raise ValueError("Invalid prompt format")
                formatted = cast(
                    str,
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                )
            else:
                formatted = prompt
            # logger.debug(f"Formatted prompt:\n{formatted}")
            tokens = tokenizer(formatted, truncation=False, add_special_tokens=False)
            length = len(tokens["input_ids"])

            if length <= self.max_prompt_length:
                processed.append((prompt, formatted, length))

        if len(processed) < len(prompts):
            logger.warning(
                f"Filtered {len(prompts) - len(processed)}/{len(prompts)} prompts "
                f"exceeding {self.max_prompt_length} tokens."
            )

        processed.sort(key=lambda x: x[2], reverse=True)

        original_prompts = [p[0] for p in processed]
        formatted_prompts = [p[1] for p in processed]

        return original_prompts, formatted_prompts

    @abstractmethod
    def generate_responses(
        self,
        model: "BaseModel",
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate responses for a list of prompts.

        Args:
            model: BaseModel instance.
            prompts: List of prompts.
            apply_chat_template: Whether to apply chat template.
            token_callback: Optional callback for streaming tokens.

        Returns:
            Tuple of (List of dicts with 'prompt' and 'output' keys, metrics dict).
        """
        pass

    def get_name(self) -> str:
        """Get generator name for file naming."""
        parts = [f"mt{self.max_new_tokens}"]
        if not self.greedy_sampling:
            parts.append("top-p")
            parts.append(f"t{self.temperature}")
            parts.append(f"p{self.top_p}")
        return "-".join(parts)

    def get_short_name(self) -> str:
        parts = [f"mt{self.max_new_tokens}"]
        return "-".join(parts)
