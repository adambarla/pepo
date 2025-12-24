import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from .model import BaseModel

logger = logging.getLogger(__name__)


class Generator:
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

    def _process_prompts(
        self,
        prompts: list[str],
        tokenizer: Any,
        apply_chat_template: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Apply chat template, filter long prompts, and sort by length.

        Returns:
            Tuple of (original_prompts, formatted_prompts) sorted by length descending.
        """
        processed: list[tuple[str, str, int]] = []  # (original, formatted, length)

        for prompt in prompts:
            if apply_chat_template:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = prompt

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

    def _top_p_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from logits using top-p (nucleus) sampling."""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        filtered_probs = F.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        return sampled_indices

    def generate(
        self,
        model: "BaseModel",
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens from input_ids using the model.

        Args:
            model: BaseModel instance.
            input_ids: Input token IDs (B, T).
            attention_mask: Attention mask (B, T). If None, inferred from pad tokens.

        Returns:
            Tuple of (output_ids, output_mask) tensors.
        """
        logger.debug(f"Generate input size: {input_ids.shape}")
        if attention_mask is None:
            attention_mask = (input_ids != model.tokenizer.pad_token_id).float()

        batch_size = input_ids.shape[0]
        max_length = input_ids.shape[1] + self.max_new_tokens

        # Prepare inputs on each device
        device_input_ids = []
        device_attention_masks = []
        for model_idx in range(model.num_models):
            device = torch.device(model.device_manager.get_device_for_model(model_idx))
            device_input_ids.append(input_ids.to(device))
            device_attention_masks.append(attention_mask.to(device))

        stop_signal = torch.zeros(batch_size, dtype=torch.bool).cpu()

        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        pbar = tqdm(range(max_length - input_ids.shape[1]), disable=disable_tqdm)
        for i in pbar:
            if i > 0 and i % 100 == 0:
                model.device_manager.clear_cache()

            log_probs = model.predict(device_input_ids, device_attention_masks)

            if self.greedy_sampling:
                # argmax is equivalent to resampling until not missing token
                # missing token has prob 1-sum(exp(logprobs))
                sampled_token_ids = torch.argmax(log_probs, dim=-1)
            else:
                sampled_token_ids = self._top_p_sample(log_probs)

            stop_signal = stop_signal.to(device=sampled_token_ids.device) | (
                sampled_token_ids == model.tokenizer.eos_token_id
            )

            for model_idx in range(model.num_models):
                device = torch.device(
                    model.device_manager.get_device_for_model(model_idx)
                )
                new_token_tensor = sampled_token_ids.to(device).unsqueeze(-1)
                device_input_ids[model_idx] = torch.cat(
                    [device_input_ids[model_idx], new_token_tensor], dim=1
                )
                device_attention_masks[model_idx] = torch.cat(
                    [
                        device_attention_masks[model_idx],
                        ~stop_signal.unsqueeze(-1).to(device),
                    ],
                    dim=1,
                )

            pbar.set_postfix({"stopped": f"{stop_signal.sum().item()}/{batch_size}"})
            if torch.all(stop_signal):
                break

        decoded = model.tokenizer.decode(
            device_input_ids[0].cpu()[0], skip_special_tokens=True
        )
        logger.debug(f"Generated sequence idx=0:\n{decoded}")

        return device_input_ids[0].cpu(), device_attention_masks[0].cpu()

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
        prompts: list[str],
        apply_chat_template: bool = True,
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
        logger.info("Generation parameters:")
        logger.info(f"  max_prompt_length: {self.max_prompt_length}")
        logger.info(f"  max_new_tokens: {self.max_new_tokens}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  greedy_sampling: {self.greedy_sampling}")
        if not self.greedy_sampling:
            logger.info(f"  temperature: {self.temperature}")
            logger.info(f"  top_p: {self.top_p}")

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
            )

            starting_idx = input_ids.shape[1]
            output_mask[:, :starting_idx] = False
            output_ids = output_ids.where(output_mask.bool(), tokenizer.pad_token_id)

            for j, prompt in enumerate(batch_prompts):
                response = tokenizer.decode(output_ids[j], skip_special_tokens=True)
                outputs.append({"prompt": prompt, "output": response})

        logger.info(f"Successfully generated {len(outputs)} responses")
        return outputs
