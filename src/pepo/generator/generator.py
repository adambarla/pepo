from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from .base import BaseGenerator

if TYPE_CHECKING:
    from ..model import BaseModel

logger = logging.getLogger(__name__)


class Generator(BaseGenerator):
    """Simple generator class for producing model responses from instructions."""

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

        # decoded = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # logger.debug(f"Generated sequence idx=0:\n{decoded}")

        return output_ids, output_mask

    def generate_responses(
        self,
        model: "BaseModel",
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        batch_size = model.generation_batch_size
        logger.debug(f"  batch_size: {batch_size}")
        logger.debug(f"  greedy_sampling: {self.greedy_sampling}")
        if not self.greedy_sampling:
            logger.debug(f"  temperature: {self.temperature}")
            logger.debug(f"  top_p: {self.top_p}")

        for i in range(0, len(prompts), batch_size):
            batch_num = i // batch_size + 1
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            logger.info(f"Generating batch {batch_num}/{total_batches}")

            batch_prompts = prompts[i : i + batch_size]
            batch_formatted = formatted_prompts[i : i + batch_size]

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
        return outputs, {}
