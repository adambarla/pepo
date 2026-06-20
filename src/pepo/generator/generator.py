import logging
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from .base import BaseGenerator

if TYPE_CHECKING:
    from ..model import BaseModel

logger = logging.getLogger(__name__)


class Generator(BaseGenerator):
    """Simple generator class for producing model responses from instructions."""

    def _worker_loop(
        self,
        worker_id: int,
        model: "BaseModel",
        batch_queue: queue.Queue,
        results: list[dict[str, Any]],
        results_lock: threading.Lock,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Worker loop for processing batches of prompts."""
        total_processed = 0
        tokenizer = model.get_tokenizer()

        # Move model to device (using new model API)
        # Note: BaseGenerator doesn't enforce expected device handling
        with model.device_manager.request_gpu() as device:
            model.to(device)
            try:
                while True:
                    try:
                        item = batch_queue.get_nowait()
                    except queue.Empty:
                        break

                    start_idx, batch_prompts, batch_formatted = item
                    logger.info(
                        f"Worker {worker_id} processing batch starting at {start_idx}"
                    )

                    tokenizer.padding_side = "left"
                    tokenized = tokenizer(
                        batch_formatted,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    )
                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized["attention_mask"]

                    # Generate
                    output_ids, output_mask = self.generate(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_callback=token_callback
                        if worker_id == 0 and start_idx == 0
                        else None,
                    )

                    starting_idx = input_ids.shape[1]
                    output_mask[:, :starting_idx] = False
                    output_ids = output_ids.where(
                        output_mask.bool(), tokenizer.pad_token_id
                    )

                    batch_outputs = []
                    for j, prompt in enumerate(batch_prompts):
                        response = tokenizer.decode(
                            output_ids[j, starting_idx:], skip_special_tokens=True
                        )
                        batch_outputs.append(
                            {
                                "index": start_idx + j,
                                "prompt": prompt,
                                "output": response,
                            }
                        )

                    with results_lock:
                        results.extend(batch_outputs)

                    total_processed += len(batch_prompts)
                    batch_queue.task_done()

            finally:
                model.cpu()

        return total_processed

    def generate(
        self,
        model: "BaseModel",
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens from input_ids using the model.

        Assumes model is already on the target device.

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
        num_gpus = model.device_manager.num_available_gpus
        logger.info(
            f"Generating responses for {len(prompts)} prompts using {num_gpus} GPUs"
        )

        batch_size = model.generation_batch_size

        # Dynamically adjust batch size to utilize all available GPUs evenly
        if num_gpus > 1 and len(prompts) < batch_size * num_gpus:
            import math

            adjusted_batch_size = max(1, math.ceil(len(prompts) / num_gpus))
            logger.info(
                f"Dynamically adjusting batch size from {batch_size} to {adjusted_batch_size} "
                f"to distribute workload evenly across all {num_gpus} GPUs."
            )
            batch_size = adjusted_batch_size

        batch_queue = queue.Queue()

        # Fill queue
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_formatted = formatted_prompts[i : i + batch_size]
            batch_queue.put((i, batch_prompts, batch_formatted))

        results = []
        results_lock = threading.Lock()

        # Use shared parallel worker runner from BaseGenerator
        # We pass queue and shared results list as kwargs to the worker function
        # worker_fn signature: (worker_id, model, **kwargs)
        self._run_parallel_workers(
            model=model,
            prompts=prompts,  # Passed mainly for logging
            formatted_prompts=formatted_prompts,  # Same
            worker_fn=self._worker_loop,
            batch_queue=batch_queue,
            results=results,
            results_lock=results_lock,
            token_callback=token_callback,
        )

        # Re-sort results by index to match original prompt order
        results.sort(key=lambda x: x["index"])
        final_outputs = [
            {"prompt": r["prompt"], "output": r["output"]} for r in results
        ]

        logger.info(f"Successfully generated {len(final_outputs)} responses")
        return final_outputs, {}
