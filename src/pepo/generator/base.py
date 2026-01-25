"""Abstract base class for generators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from ..model import BaseModel

from concurrent.futures import ThreadPoolExecutor

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

    def _run_parallel_workers(
        self,
        model: "BaseModel",
        prompts: list[Any],
        formatted_prompts: list[str],
        worker_fn: Callable[..., Any],
        **worker_kwargs: Any,
    ) -> list[Any]:
        """
        Run generic worker function in parallel across available GPUs.

        Args:
            model: The model to use.
            prompts: List of original prompts.
            formatted_prompts: List of formatted prompts.
            worker_fn: Function to run in each worker thread.
                       Signature: (worker_id, model, queue, results, lock, **kwargs)
            **worker_kwargs: Additional kwargs to pass to worker_fn.

        Returns:
            List of aggregated results from all workers.
        """
        num_gpus = model.device_manager.num_available_gpus
        logger.info(
            f"Parallel execution using {num_gpus} GPUs for {len(prompts)} items"
        )

        # Setup standard batch queue
        # For simple generation, batches are chunks of prompts.
        # For BestOfN, queue is initialized differently (individual slots).
        # To make this truly generic, we might need the caller to provide the queue.
        # Let's standardize on the caller providing the queue or queue init logic?
        # Actually, let's keep it simple: this method spins up threads.
        # The worker_fn and queue management is caller specific mostly,
        # BUT the model cloning and thread pool is what we want to share.

        if num_gpus > 1:
            worker_models = [model] + [model.clone() for _ in range(1, num_gpus)]

            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [
                    executor.submit(worker_fn, i, worker_models[i], **worker_kwargs)
                    for i in range(num_gpus)
                ]
                # We can return results if workers return something
                # Or caller manages results via a shared list/lock passed in kwargs
                results = []
                for f in futures:
                    res = f.result()
                    if res:
                        results.append(res)
                return results
        else:
            # Single GPU
            return [worker_fn(0, model, **worker_kwargs)]
