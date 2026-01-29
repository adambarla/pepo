"""Best of N Generator using rejection sampling with slot-based batching."""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, cast

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
        sampling_mode: Literal["min", "mean_std"] = "min",
        eta: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Best of N Generator.

        Args:
            max_trials: Maximum attempts per prompt before accepting best candidate.
            sampling_mode: How to compute f_out (both apply exp(-α/β) penalty):
                - "min": f_out = min_π * exp(-α/β)
                - "mean_std": f_out = (mean_π - η*std_π) * exp(-α/β)
            eta: Standard deviation coefficient for mean_std mode.
            **kwargs: Arguments passed to parent Generator.
        """
        super().__init__(**kwargs)
        self.max_trials = max_trials
        self.sampling_mode = sampling_mode
        self.eta = eta
        self.greedy_sampling = False  # Force non-greedy for candidate generation

    def _extract_response(
        self,
        output_ids: torch.Tensor,
        output_mask: torch.Tensor,
        prompt_length: int,
        tokenizer: Any,
    ) -> str:
        """Extract response text, excluding post-EOS tokens via output_mask."""
        response_ids = output_ids[prompt_length:]
        response_mask = output_mask[prompt_length:]
        valid_length = int(response_mask.sum().item())
        return tokenizer.decode(response_ids[:valid_length], skip_special_tokens=True)

    def _compute_sequence_log_probs(
        self,
        model: "EnsembleModel",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probs for sequences across all ensemble members.

        Args:
            model: EnsembleModel with shared backbone.
            input_ids: Full sequences (B, T).
            attention_mask: Attention mask (B, T).
            prompt_lengths: List of prompt lengths for each sequence.
            device: Device the model is on.

        Returns:
            Tuple of (proposal_log_probs, min_log_probs, mean_log_probs, std_log_probs).
        """
        from ..utils.model_utils import get_log_probs

        num_seqs = input_ids.shape[0]
        eval_batch_size = model.eval_batch_size

        response_mask = torch.zeros_like(input_ids, dtype=torch.float)
        for i, pl in enumerate(prompt_lengths):
            response_mask[i, pl:] = attention_mask[i, pl:].float()

        log_probs_per_model: list[torch.Tensor] = []
        shared_model = model.models[0]
        shared_model.eval()

        with torch.no_grad():
            for model_idx in range(model.num_models):
                adapter_name = "default" if model_idx == 0 else f"adapter_{model_idx}"
                shared_model.set_adapter(adapter_name)

                batch_log_probs = []
                for start in range(0, num_seqs, eval_batch_size):
                    end = min(start + eval_batch_size, num_seqs)
                    lp = get_log_probs(
                        shared_model,
                        device,
                        input_ids[start:end].to(device),
                        attention_mask[start:end].to(device),
                        response_mask[start:end].to(device),
                    )
                    batch_log_probs.append(lp.cpu())
                log_probs_per_model.append(torch.cat(batch_log_probs, dim=0))

        log_probs_tensor = torch.stack(log_probs_per_model, dim=0)  # (L, B)
        min_log_probs, _ = torch.min(log_probs_tensor, dim=0)
        mean_log_probs = log_probs_tensor.mean(dim=0)
        std_log_probs = log_probs_tensor.std(dim=0)
        return log_probs_per_model[0], min_log_probs, mean_log_probs, std_log_probs

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
            if slots[i] is not None:
                continue
            try:
                while True:
                    prompt_idx, prompt, formatted, trial_idx = pending.get_nowait()
                    with state_lock:
                        if prompt_idx in results:
                            continue  # Already solved
                    if trial_idx + 1 < self.max_trials:
                        pending.put((prompt_idx, prompt, formatted, trial_idx + 1))

                    tokenizer.padding_side = "left"
                    tokenized = tokenizer(
                        [formatted], return_tensors="pt", padding=True, truncation=False
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
        """Generate candidates for active slots using proposal model (model_idx=0)."""
        tokenizer.padding_side = "left"
        tokenized = tokenizer(
            [s.formatted for s in active_slots],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        output_ids, output_mask = model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            greedy_sampling=False,
            temperature=self.temperature,
            top_p=self.top_p,
            model_indices=[0],
        )
        return output_ids, output_mask, tokenized["input_ids"].shape[1]

    def _score_candidates(
        self,
        model: "EnsembleModel",
        output_ids: torch.Tensor,
        output_mask: torch.Tensor,
        active_slots: list[Slot],
        prompt_end_idx: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score generated candidates with all ensemble members.

        Returns:
            Tuple of (proposal_lps, min_lps, mean_lps, std_lps).
        """
        prompt_lengths = []
        for i, slot in enumerate(active_slots):
            slot.output_ids = output_ids[i : i + 1]
            slot.output_mask = output_mask[i : i + 1]
            slot.prompt_length = prompt_end_idx
            prompt_lengths.append(prompt_end_idx)
        return self._compute_sequence_log_probs(
            model, output_ids, output_mask, prompt_lengths, device
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
        """Process acceptance/rejection for each candidate."""
        u = torch.rand(len(active_slots))
        accepted_this_iter = 0
        accepted_trials_sum_delta = 0
        total_accepted_delta = 0

        for i, (slot_idx, slot) in enumerate(zip(active_indices, active_slots)):
            alpha = alphas[i].item()
            response = self._extract_response(
                output_ids[i], output_mask[i], slot.prompt_length, tokenizer
            )
            accepted = u[i].item() <= alpha

            with state_lock:
                if slot.prompt_idx in results:
                    slots[slot_idx] = None
                    continue

                state = prompt_states[slot.prompt_idx]
                state["finished_trials"] += 1
                if alpha > state["best_alpha"]:
                    state["best_alpha"] = alpha
                    state["best_response"] = response

                if accepted:
                    results[slot.prompt_idx] = response
                    slots[slot_idx] = None
                    accepted_this_iter += 1
                    total_accepted_delta += 1
                    accepted_trials_sum_delta += state["finished_trials"]
                elif state["finished_trials"] >= self.max_trials:
                    # Force accept best seen after max trials
                    results[slot.prompt_idx] = state["best_response"] or response
                    slots[slot_idx] = None
                    accepted_this_iter += 1
                    total_accepted_delta += 1
                    accepted_trials_sum_delta += state["finished_trials"]
                else:
                    slots[slot_idx] = None

        return accepted_this_iter, total_accepted_delta, accepted_trials_sum_delta

    def _init_prompt_states(self, n_prompts: int) -> dict[int, dict[str, Any]]:
        """Initialize per-prompt tracking state."""
        return {
            idx: {"best_alpha": -1.0, "best_response": None, "finished_trials": 0}
            for idx in range(n_prompts)
        }

    def _init_pending_queue(
        self, prompts: list[Any], formatted_prompts: list[str]
    ) -> queue.Queue:
        """Initialize pending queue with all prompts at trial 0."""
        q: queue.Queue = queue.Queue()
        for idx, (p, f) in enumerate(zip(prompts, formatted_prompts)):
            q.put((idx, p, f, 0))
        return q

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
        """Worker loop for processing prompts. Holds GPU for entire duration."""
        tokenizer = model.get_tokenizer()
        gen_batch_size = model.generation_batch_size
        slots: list[Optional[Slot]] = [None] * gen_batch_size

        logger.info(f"Worker {worker_id} started with batch size {gen_batch_size}")

        total_accepted = 0
        accepted_trials_sum = 0
        total_alpha_sum = 0.0
        total_samples_scored = 0

        with model.device_manager.request_gpu() as device:
            model.to(device)
            try:
                while True:
                    if pending_queue.empty() and all(s is None for s in slots):
                        break

                    self._fill_slots(
                        slots,
                        pending_queue,
                        tokenizer,
                        gen_batch_size,
                        results,
                        state_lock,
                    )

                    active_indices = [i for i, s in enumerate(slots) if s is not None]
                    if not active_indices:
                        if pending_queue.empty():
                            break
                        continue

                    active_slots = cast(list[Slot], [slots[i] for i in active_indices])

                    try:
                        output_ids, output_mask, prompt_end_idx = (
                            self._generate_candidates(model, active_slots, tokenizer)
                        )
                        proposal_lps, min_lps, mean_lps, std_lps = (
                            self._score_candidates(
                                model,
                                output_ids,
                                output_mask,
                                active_slots,
                                prompt_end_idx,
                                device,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error: {e}")
                        raise

                    # Calculate penalty term: exp(-α/β) from model params
                    model_alpha = getattr(model, "alpha", 0.0)
                    model_beta = getattr(model, "beta", 0.1)
                    log_penalty = -model_alpha / model_beta

                    # Compute f_out based on sampling mode
                    if self.sampling_mode == "min":
                        # f_out = min_π * exp(-α/β)
                        f_out_log = min_lps + log_penalty
                    else:  # mean_std
                        # f_out = (mean_π - η*std_π) * exp(-α/β) in log space
                        # We have log probs, so: mean_p = exp(mean_lp), std_p approx
                        # For numerical stability, work in probability space briefly
                        mean_p = torch.exp(mean_lps)
                        std_p = torch.exp(mean_lps) * std_lps.clamp(min=1e-8)
                        f_out_p = (mean_p - self.eta * std_p).clamp(min=1e-10)
                        f_out_log = torch.log(f_out_p) + log_penalty

                    alphas = torch.exp(f_out_log - proposal_lps).clamp(0.0, 1.0)
                    _, accepted_delta, trials_delta = self._process_acceptance(
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
                    total_samples_scored += len(active_slots)

                    if accepted_delta > 0:
                        avg_alpha = alphas.mean().item()
                        avg_try = (
                            accepted_trials_sum / total_accepted
                            if total_accepted
                            else 0
                        )
                        with pbar_lock:
                            pbar.set_postfix(
                                {
                                    "wk": worker_id,
                                    "alpha": f"{avg_alpha:.2f}",
                                    "avg_try": f"{avg_try:.1f}",
                                }
                            )
            finally:
                model.cpu()
                # Removed empty_cache to prevent illegal memory access
                pass

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

        Args:
            model: BaseModel instance (should be an EnsembleModel).
            prompts: List of prompts or histories.
            apply_chat_template: Whether to use chat template.
            token_callback: Callback for streaming (unused for Best-of-N).

        Returns:
            Tuple of (list of {prompt, output} dicts, metrics dict).
        """
        model = cast("EnsembleModel", model)
        tokenizer = model.get_tokenizer()
        prompts, formatted_prompts = self._process_prompts(
            prompts, tokenizer, apply_chat_template
        )
        results: dict[int, str] = {}

        num_gpus = model.device_manager.num_available_gpus
        logger.info(
            f"Best-of-N (max_trials={self.max_trials}) "
            f"for {len(prompts)} prompts using {num_gpus} GPUs."
        )

        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        pbar = tqdm(
            total=len(prompts), desc="Best-of-N", disable=disable_tqdm, leave=False
        )
        pbar_lock = threading.Lock()
        state_lock = threading.Lock()
        prompt_states = self._init_prompt_states(len(prompts))
        pending_queue = self._init_pending_queue(prompts, formatted_prompts)

        total_accepted = 0
        accepted_trials_sum = 0
        total_alpha_sum = 0.0
        total_samples_scored = 0

        if num_gpus > 1:
            logger.info("Initializing multi-threaded generation...")
            # Use BaseGenerator's shared parallelism
            # worker_fn signature mismatch? Base expects (worker_id, model, **kwargs)
            # here we have many args: pending_queue, results, pbar, etc.
            # We pass them as kwargs.
            worker_stats = self._run_parallel_workers(
                model=model,
                prompts=prompts,
                formatted_prompts=formatted_prompts,
                worker_fn=self._worker_loop,
                pending_queue=pending_queue,
                results=results,
                pbar=pbar,
                pbar_lock=pbar_lock,
                prompt_states=prompt_states,
                state_lock=state_lock,
            )

            # Aggregate stats
            for m in worker_stats:
                total_accepted += m["total_accepted"]
                accepted_trials_sum += m["accepted_trials_sum"]
                total_alpha_sum += m["total_alpha_sum"]
                total_samples_scored += m["total_samples_scored"]
        else:
            m = self._worker_loop(
                0,
                model,
                pending_queue,
                results,
                pbar,
                pbar_lock,
                prompt_states,
                state_lock,
            )
            total_accepted += m["total_accepted"]
            accepted_trials_sum += m["accepted_trials_sum"]
            total_alpha_sum += m["total_alpha_sum"]
            total_samples_scored += m["total_samples_scored"]

        pbar.close()

        outputs = [
            {"prompt": prompts[i], "output": results.get(i, "")}
            for i in range(len(prompts))
        ]
        logger.info(f"Successfully generated {len(outputs)} responses.")

        return outputs, {
            "avg_try": accepted_trials_sum / total_accepted if total_accepted else 0.0,
            "avg_alpha": total_alpha_sum / total_samples_scored
            if total_samples_scored
            else 0.0,
        }

    def get_name(self) -> str:
        """Get generator name for file naming."""
        parts = [f"bon-{self.sampling_mode}"]
        if self.sampling_mode == "mean_std":
            parts.append(f"eta{self.eta}")
        parts.append(f"n{self.max_trials}")
        parts.append(super().get_name())
        return "-".join(parts)
