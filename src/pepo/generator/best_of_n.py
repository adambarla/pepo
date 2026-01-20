"""Best of N Generator using rejection sampling with slot-based batching."""

from __future__ import annotations

import copy
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import torch
from tqdm import tqdm

from .base import BaseGenerator

if TYPE_CHECKING:
    from ..model import BaseModel, EnsembleModel

logger = logging.getLogger(__name__)


@dataclass
class Slot:
    """Active generation slot."""

    prompt_idx: int
    prompt: Any
    formatted: str
    output_ids: Optional[torch.Tensor] = None
    output_mask: Optional[torch.Tensor] = None
    prompt_length: int = 0
    trial_idx: int = 0


class BestOfNGenerator(BaseGenerator):
    """Best of N Generator using rejection sampling with slot-based batching.

    Uses the model's generate method with model_indices=[0] for proposal-only
    generation, then scores with all ensemble members for rejection sampling.
    Processes multiple prompts in parallel using a slot-based approach.

    Assumes shared_backbone=True (all adapters on single model).
    """

    def __init__(
        self,
        max_trials: int = 16,
        **kwargs: Any,
    ):
        """Initialize Best of N Generator.

        Args:
            max_trials: Maximum attempts per prompt before accepting best candidate.
            **kwargs: Arguments passed to parent Generator.
        """
        super().__init__(**kwargs)
        self.max_trials = max_trials
        # Force non-greedy sampling for candidate generation
        self.greedy_sampling = False

    def _extract_response(
        self,
        output_ids: torch.Tensor,
        output_mask: torch.Tensor,
        prompt_length: int,
        tokenizer: Any,
    ) -> str:
        """Extract response text, using output_mask to exclude post-EOS tokens.

        Args:
            output_ids: Full sequence IDs (T,) for a single sequence.
            output_mask: Attention mask (T,) for the sequence.
            prompt_length: Length of the prompt prefix.
            tokenizer: Tokenizer for decoding.

        Returns:
            Decoded response string.
        """
        response_ids = output_ids[prompt_length:]
        response_mask = output_mask[prompt_length:]

        # Find valid length based on mask (mask=1 means valid token)
        valid_length = int(response_mask.sum().item())
        valid_ids = response_ids[:valid_length]

        return tokenizer.decode(valid_ids, skip_special_tokens=True)

    def _compute_sequence_log_probs(
        self,
        model: "EnsembleModel",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probabilities for sequences across all ensemble members.

        Assumes shared_backbone mode (adapter switching on single model).
        Uses batched processing with model.eval_batch_size.

        Args:
            model: EnsembleModel instance (ensemble with shared backbone).
            input_ids: Full sequences (B, T).
            attention_mask: Attention mask (B, T).
            prompt_lengths: List of prompt lengths for each sequence.

        Returns:
            Tuple of (proposal_log_probs, min_log_probs, all_log_probs).
        """
        from ..utils.model_utils import get_log_probs

        num_seqs = input_ids.shape[0]
        eval_batch_size = model.eval_batch_size

        # Create response mask for each sequence
        response_mask = torch.zeros_like(input_ids, dtype=torch.float)
        for i, pl in enumerate(prompt_lengths):
            response_mask[i, pl:] = attention_mask[i, pl:].float()

        # log_probs_per_model[model_idx] = tensor of shape (B,)
        log_probs_per_model: list[torch.Tensor] = []
        shared_model = model.models[0]

        # Use architecture's DeviceManager to request GPU
        with model.device_manager.request_gpu() as device:
            shared_model.to(device)
            shared_model.eval()

            try:
                with torch.no_grad():
                    for model_idx in range(model.num_models):
                        adapter_name = (
                            "default" if model_idx == 0 else f"adapter_{model_idx}"
                        )
                        shared_model.set_adapter(adapter_name)

                        # Process in batches to avoid OOM
                        batch_log_probs = []
                        for start in range(0, num_seqs, eval_batch_size):
                            end = min(start + eval_batch_size, num_seqs)
                            batch_ids = input_ids[start:end].to(device)
                            batch_mask = attention_mask[start:end].to(device)
                            batch_response_mask = response_mask[start:end].to(device)

                            lp = get_log_probs(
                                shared_model,
                                device,
                                batch_ids,
                                batch_mask,
                                batch_response_mask,
                            )
                            batch_log_probs.append(lp.cpu())

                        # Concatenate all batches for this model
                        log_probs_per_model.append(torch.cat(batch_log_probs, dim=0))
            finally:
                # Ensure model returns to CPU to release resources
                shared_model.cpu()
                model.device_manager.clear_cache()

        # Stack: (L, B) and compute min over ensemble
        log_probs_tensor = torch.stack(log_probs_per_model, dim=0)
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        proposal_log_probs = log_probs_per_model[0]

        return proposal_log_probs, min_log_probs, log_probs_tensor

    def _fill_slots(
        self,
        slots: list[Optional[Slot]],
        pending: queue.Queue,
        tokenizer: Any,
        gen_batch_size: int,
        results: dict[int, str],
        state_lock: threading.Lock,
    ) -> int:
        """Fill empty slots with new prompts from pending queue."""
        filled_count = 0
        for i in range(gen_batch_size):
            if slots[i] is None:
                try:
                    # Non-blocking get
                    # Loop until we find a prompt that isn't solved yet
                    while True:
                        item = pending.get_nowait()
                        prompt_idx, prompt, formatted, trial_idx = item

                        # Check if already solved (optimistic check)
                        # Use lock for safety though results dict reads are atomic-ish
                        with state_lock:
                            is_solved = prompt_idx in results

                        if is_solved:
                            continue

                        # If not solved, enqueue next trial if valid
                        if trial_idx + 1 < self.max_trials:
                            pending.put((prompt_idx, prompt, formatted, trial_idx + 1))

                        # Use this item
                        tokenizer.padding_side = "left"
                        tokenized = tokenizer(
                            [formatted],
                            return_tensors="pt",
                            padding=True,
                            truncation=False,
                        )
                        slots[i] = Slot(
                            prompt_idx=prompt_idx,
                            prompt=prompt,
                            formatted=formatted,
                            prompt_length=tokenized["input_ids"].shape[1],
                            trial_idx=trial_idx,
                        )
                        filled_count += 1
                        break

                except queue.Empty:
                    break
        return filled_count

    def _generate_candidates(
        self,
        model: "EnsembleModel",
        active_slots: list[Slot],
        tokenizer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Generate candidates for active slots using proposal model."""
        batch_formatted = [s.formatted for s in active_slots]
        tokenizer.padding_side = "left"
        tokenized = tokenizer(
            batch_formatted,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Cast to Any to allow model_indices argument
        output_ids, output_mask = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            greedy_sampling=False,
            temperature=self.temperature,
            top_p=self.top_p,
            model_indices=[0],  # Proposal only
        )
        prompt_end_idx = input_ids.shape[1]
        return output_ids, output_mask, prompt_end_idx

    def _score_candidates(
        self,
        model: "EnsembleModel",
        output_ids: torch.Tensor,
        output_mask: torch.Tensor,
        active_slots: list[Slot],
        prompt_end_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score generated candidates with all ensemble members."""
        prompt_lengths = []
        for i, slot in enumerate(active_slots):
            # Update slot with generation data
            slot.output_ids = output_ids[i : i + 1]
            slot.output_mask = output_mask[i : i + 1]
            slot.prompt_length = prompt_end_idx
            prompt_lengths.append(slot.prompt_length)

        return self._compute_sequence_log_probs(
            model, output_ids, output_mask, prompt_lengths
        )

    def _process_acceptance(
        self,
        active_indices: list[int],
        active_slots: list[Slot],
        alphas: torch.Tensor,
        output_ids: torch.Tensor,
        output_mask: torch.Tensor,
        tokenizer: Any,
        slots: list[Optional[Slot]],
        results: dict[int, str],
        prompt_states: dict[int, dict[str, Any]],
        state_lock: threading.Lock,
    ) -> tuple[int, int, int]:
        """Process acceptance/rejection logic for candidates."""
        u = torch.rand(len(active_slots))
        accepted_this_iter = 0
        accepted_trials_sum_delta = 0
        total_accepted_delta = 0

        # Pre-decode all responses to save time inside lock (if possible)
        # However, extracting response needs prompt_length
        pass

        for i, (slot_idx, slot) in enumerate(zip(active_indices, active_slots)):
            alpha = alphas[i].item()

            # Response extraction
            response_text = self._extract_response(
                output_ids[i], output_mask[i], slot.prompt_length, tokenizer
            )

            accepted = u[i].item() <= alpha

            with state_lock:
                # Check if already solved by another worker while we were generating
                if slot.prompt_idx in results:
                    slots[slot_idx] = None  # Clear slot
                    continue

                state = prompt_states[slot.prompt_idx]
                state["finished_trials"] += 1

                # Update best seen
                if alpha > state["best_alpha"]:
                    state["best_alpha"] = alpha
                    state["best_response"] = response_text

                # Check acceptance
                if accepted:
                    results[slot.prompt_idx] = response_text
                    slots[slot_idx] = None
                    accepted_this_iter += 1
                    total_accepted_delta += 1
                    # Approximate trials sum: this trial was the lucky one
                    # We don't track exact cumulative trials perfectly in async mode
                    # Use finished_trials as proxy
                    accepted_trials_sum_delta += state["finished_trials"]

                elif state["finished_trials"] >= self.max_trials:
                    # Force accept best seen
                    # Check if best_response is set (should be populated)
                    best_resp = state["best_response"]
                    if best_resp is None:
                        # Should not happen if alpha > -1 initialized and we updated it
                        # Fallback to current
                        best_resp = response_text

                    results[slot.prompt_idx] = best_resp
                    slots[slot_idx] = None
                    # Count as accepted for metrics? Or separate?
                    # Original logic counted force accept as accepted
                    accepted_this_iter += 1
                    total_accepted_delta += 1
                    accepted_trials_sum_delta += state["finished_trials"]
                else:
                    # Rejected and not max trials vs result
                    # Slot is cleared to make room for *next* item from queue
                    # (which might be next trial of this prompt or another prompt)
                    slots[slot_idx] = None

        return accepted_this_iter, total_accepted_delta, accepted_trials_sum_delta

    def _duplicate_model(self, model: "EnsembleModel") -> "EnsembleModel":
        """Create a deep copy of the model for a worker thread.

        Shallow copy wrapper, deep copy inner models for independent weights.
        """
        # Shallow copy the wrapper to preserve references to managers
        new_model = copy.copy(model)
        new_model.models = [copy.deepcopy(m) for m in model.models]

        return new_model

    def _worker_loop(
        self,
        worker_id: int,
        model: "EnsembleModel",
        pending_queue: queue.Queue,
        results: dict[int, str],
        pbar: tqdm,
        pbar_lock: threading.Lock,
        prompt_states: dict[int, dict[str, Any]],
        state_lock: threading.Lock,
    ) -> dict[str, float]:
        """Worker loop for processing prompts on a dedicated GPU/thread."""
        tokenizer = model.get_tokenizer()
        gen_batch_size = model.generation_batch_size

        # Local slots for this worker
        slots: list[Optional[Slot]] = [None] * gen_batch_size

        logger.info(f"Worker {worker_id} started with batch size {gen_batch_size}")

        iteration = 0
        total_accepted = 0
        accepted_trials_sum = 0
        total_alpha_sum = 0.0
        total_samples_scored = 0

        while True:
            # Check if we should stop: empty queue AND all slots empty
            if pending_queue.empty() and all(s is None for s in slots):
                break

            iteration += 1

            # Fill slots
            _ = self._fill_slots(
                slots, pending_queue, tokenizer, gen_batch_size, results, state_lock
            )

            active_indices = [i for i, s in enumerate(slots) if s is not None]
            if not active_indices:
                # Could happen if queue became empty just before fill_slots
                # small sleep to avoid tight loop if waiting?
                # But we used get_nowait, so if empty, we just loop or break.
                # If queue is empty and we have no active slots, we are done.
                if pending_queue.empty():
                    break
                continue

            active_slots = [slots[i] for i in active_indices]
            # Type checker assertion
            active_slots = cast(list[Slot], active_slots)
            n_active = len(active_slots)

            # Generate proposals
            # Note: This acquires GPU via RequestGPU context logic
            try:
                output_ids, output_mask, prompt_end_idx = self._generate_candidates(
                    model, active_slots, tokenizer
                )

                # Score proposals
                proposal_lps, min_lps, _ = self._score_candidates(
                    model, output_ids, output_mask, active_slots, prompt_end_idx
                )
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}")
                # Clean up slots to avoid getting stuck?
                # For now re-raise or skip
                raise e

            log_alpha = min_lps - proposal_lps
            alphas = torch.exp(log_alpha).clamp(0.0, 1.0)

            accepted_iter, accepted_delta, trials_delta = self._process_acceptance(
                active_indices,
                active_slots,
                alphas,
                output_ids,
                output_mask,
                tokenizer,
                slots,
                results,
                prompt_states,
                state_lock,
            )

            with pbar_lock:
                pbar.update(accepted_delta)

            total_accepted += accepted_delta
            accepted_trials_sum += trials_delta

            total_alpha_sum += alphas.sum().item()
            total_samples_scored += n_active

            # Update pbar description/postfix occasionally?
            if accepted_delta > 0:
                avg_alpha = alphas.mean().item()
                avg_try = (
                    accepted_trials_sum / total_accepted if total_accepted > 0 else 0.0
                )
                with pbar_lock:
                    pbar.set_postfix(
                        {
                            "wk": worker_id,
                            "alpha": f"{avg_alpha:.2f}",
                            "avg_try": f"{avg_try:.1f}",
                        }
                    )

        logger.info(f"Worker {worker_id} finished. Processed {total_accepted} samples.")

        return {
            "total_accepted": total_accepted,
            "accepted_trials_sum": accepted_trials_sum,
            "total_alpha_sum": total_alpha_sum,
            "total_samples_scored": total_samples_scored,
        }

    def generate_responses(
        self,
        model: "BaseModel",
        prompts: list[Any],
        apply_chat_template: bool = True,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate responses using slot-based Best of N rejection sampling.

        Processes batch_size prompts in parallel. When a prompt is accepted,
        its slot is filled with a new prompt from the queue.
        Supports multi-GPU execution via threading.

        Args:
            model: BaseModel instance (should be an ensemble model).
            prompts: List of prompts or histories.
            apply_chat_template: Whether to use chat template.
            token_callback: Callback for streaming (not used for Best-of-N).

        Returns:
            List of dicts with 'prompt' and 'output'.
        """
        model = cast("EnsembleModel", model)
        tokenizer = model.get_tokenizer()
        prompts, formatted_prompts = self._process_prompts(
            prompts, tokenizer, apply_chat_template
        )

        # Results dict: prompt_idx -> response
        results: dict[int, str] = {}

        num_gpus = model.device_manager.num_available_gpus
        logger.info(
            f"Best-of-N generation (max_trials={self.max_trials}) "
            f"for {len(prompts)} prompts using {num_gpus} GPUs."
        )

        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        pbar = tqdm(
            total=len(prompts),
            desc="Best-of-N",
            disable=disable_tqdm,
            leave=False,
        )
        pbar_lock = threading.Lock()

        # Metrics aggregation
        total_accepted = 0
        accepted_trials_sum = 0
        total_alpha_sum = 0.0
        total_samples_scored = 0

        if num_gpus > 1:
            # Multi-GPU: Threaded execution
            logger.info("Initializing multi-threaded generation...")

            # Queue for pending prompts. Items: (prompt_idx, prompt, formatted, trial)
            # Initialize with trial 0 for all prompts
            pending_queue: queue.Queue = queue.Queue()
            for idx, (p, f) in enumerate(zip(prompts, formatted_prompts)):
                pending_queue.put((idx, p, f, 0))

            # Shared state for prompts
            state_lock = threading.Lock()
            prompt_states = {
                idx: {"best_alpha": -1.0, "best_response": None, "finished_trials": 0}
                for idx in range(len(prompts))
            }
            futures = []
            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                # Launch workers. Duplicate models serially to avoid concurrency issues.
                # Worker 0 uses the main model to save memory/time.

                worker_models = []
                for i in range(num_gpus):
                    if i == 0:
                        worker_models.append(model)
                    else:
                        logger.info(f"Cloning model for worker {i}...")
                        worker_models.append(self._duplicate_model(model))

                logger.info("Starting workers...")
                for i in range(num_gpus):
                    futures.append(
                        executor.submit(
                            self._worker_loop,
                            i,
                            worker_models[i],
                            pending_queue,
                            results,
                            pbar,
                            pbar_lock,
                            prompt_states,
                            state_lock,
                        )
                    )

                # Wait for completion and aggregate metrics
                for future in futures:
                    metrics = future.result()
                    total_accepted += metrics["total_accepted"]
                    accepted_trials_sum += metrics["accepted_trials_sum"]
                    total_alpha_sum += metrics["total_alpha_sum"]
                    total_samples_scored += metrics["total_samples_scored"]

        else:
            # Single GPU: Reuse worker logic with main model
            # Use queue.Queue for consistency

            # Reset queue for single GPU case
            pending_queue = queue.Queue()
            for idx, (p, f) in enumerate(zip(prompts, formatted_prompts)):
                pending_queue.put((idx, p, f, 0))

            state_lock = threading.Lock()
            prompt_states = {
                idx: {"best_alpha": -1.0, "best_response": None, "finished_trials": 0}
                for idx in range(len(prompts))
            }

            metrics = self._worker_loop(
                0,
                model,
                pending_queue,
                results,
                pbar,
                pbar_lock,
                prompt_states,
                state_lock,
            )
            total_accepted += metrics["total_accepted"]
            accepted_trials_sum += metrics["accepted_trials_sum"]
            total_alpha_sum += metrics["total_alpha_sum"]
            total_samples_scored += metrics["total_samples_scored"]

        pbar.close()

        # Build output list
        outputs = [
            {"prompt": prompts[i], "output": results.get(i, "")}
            for i in range(len(prompts))
        ]

        logger.info(f"Successfully generated {len(outputs)} responses.")

        metrics = {
            "avg_try": accepted_trials_sum / total_accepted
            if total_accepted > 0
            else 0.0,
            "avg_alpha": total_alpha_sum / total_samples_scored
            if total_samples_scored > 0
            else 0.0,
        }

        return outputs, metrics

    def get_name(self) -> str:
        """Get generator name for file naming."""
        base_name = super().get_name()
        return f"bon-n{self.max_trials}-{base_name}"
