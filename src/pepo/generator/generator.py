from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import torch

from .base import BaseGenerator

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from ..model import BaseModel

logger = logging.getLogger(__name__)


class Generator(BaseGenerator):
    """Simple generator class for producing model responses from instructions."""

    def __init__(
        self,
        max_prompt_length: int = 512,
        max_new_tokens: int = 1024,
        batch_size: int = 10,
        greedy_sampling: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """
        Initialize generator.

        Args:
            max_prompt_length: Maximum length for input prompts (truncation).
            max_new_tokens: Maximum number of new tokens to generate.
            batch_size: Batch size for generation.
            greedy_sampling: If True, use greedy (argmax). If False, use top-p.
            temperature: Sampling temperature (only used if greedy_sampling=False).
            top_p: Top-p nucleus sampling threshold
                (only used if greedy_sampling=False).
        """
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
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
            logger.debug(f"Formatted prompt:\n{formatted}")
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

    def generate(
        self,
        model: "BaseModel",
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens from input_ids using the model.

        Args:
            model: BaseModel instance.
            input_ids: Input token IDs (B, T) on CPU.
            attention_mask: Attention mask (B, T). If None, inferred from pad tokens.
            token_callback: Optional callback for streaming tokens.

        Returns:
            Tuple of (output_ids, output_mask) tensors on CPU.
        """
        logger.debug(f"Generate input size: {input_ids.shape}")
        if attention_mask is None:
            attention_mask = (input_ids != model.tokenizer.pad_token_id).float()

        output_ids, output_mask = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            greedy_sampling=self.greedy_sampling,
            temperature=self.temperature,
            top_p=self.top_p,
            token_callback=token_callback,
        )

        decoded = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.debug(f"Generated sequence idx=0:\n{decoded}")

        return output_ids, output_mask

    def get_name(self) -> str:
        """Get generator name for file naming."""
        parts = [f"mt{self.max_new_tokens}"]
        if not self.greedy_sampling:
            parts.append("top-p")
            parts.append(f"t{self.temperature}")
            parts.append(f"p{self.top_p}")
        return "-".join(parts)

    def generate_responses(
        self,
        model: "BaseModel",
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for a list of prompts.

        Args:
            model: BaseModel instance.
            prompts: List of prompt strings.
            apply_chat_template: Whether to apply chat template to prompts.

        Returns:
            List of dictionaries with 'prompt' and 'output' keys.
        """
        tokenizer = model.get_tokenizer()
        prompts, formatted_prompts = self._process_prompts(
            prompts, tokenizer, apply_chat_template
        )
        outputs = []

        logger.info(f"Generating responses for {len(prompts)} prompts")
        logger.debug("Generation parameters:")
        logger.debug(f"  max_prompt_length: {self.max_prompt_length}")
        logger.debug(f"  max_new_tokens: {self.max_new_tokens}")
        logger.debug(f"  batch_size: {self.batch_size}")
        logger.debug(f"  greedy_sampling: {self.greedy_sampling}")
        if not self.greedy_sampling:
            logger.debug(f"  temperature: {self.temperature}")
            logger.debug(f"  top_p: {self.top_p}")

        for i in range(0, len(prompts), self.batch_size):
            batch_num = i // self.batch_size + 1
            total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
            logger.info(f"Generating batch {batch_num}/{total_batches}")

            batch_prompts = prompts[i : i + self.batch_size]
            batch_formatted = formatted_prompts[i : i + self.batch_size]

            tokenizer.padding_side = "left"
            tokenized = tokenizer(
                batch_formatted,
                return_tensors="pt",
                padding=True,
                truncation=False,  # already filtered out the long prompts
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            output_ids, output_mask = self.generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_callback=token_callback if i == 0 else None,
            )

            starting_idx = input_ids.shape[1]
            output_mask[:, :starting_idx] = False
            output_ids = output_ids.where(output_mask.bool(), tokenizer.pad_token_id)

            for j, prompt in enumerate(batch_prompts):
                response = tokenizer.decode(
                    output_ids[j, starting_idx:], skip_special_tokens=True
                )
                outputs.append({"prompt": prompt, "output": response})

        logger.info(f"Successfully generated {len(outputs)} responses")
        return outputs
