"""Best of N Generator using rejection sampling with slot-based batching."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import queue
import threading
import time
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


def acceptance_log_probs(
    log_probs: torch.Tensor,
    sampling_mode: Literal["min", "mean_std"],
    eta: float,
) -> torch.Tensor:
    """Log acceptance probability for proposals drawn from the ensemble mixture.

    Proposals are drawn from q(a) = (1/L) sum_l pi_l(a|x). Accepting with
    probability f_out(a) / q(a) then yields exact samples from f_out / sum f_out:
      - "min":      f_out = min_l pi_l,               acceptance = min / mean
      - "mean_std": f_out = max(0, mean - eta * std), acceptance = 1 - eta * CV
    Both ratios lie in [0, 1], so no envelope constant or clamping is needed.
    ``std`` is the population standard deviation over members; by Samuelson's
    inequality, eta = sqrt(L - 1) gives f_out <= min_l pi_l.

    Args:
        log_probs: (L, B) sequence log-probabilities under each member.
        sampling_mode: Target numerator, see above.
        eta: Standard deviation coefficient for "mean_std".

    Returns:
        (B,) log acceptance probabilities (-inf where f_out = 0).
    """
    num_models = log_probs.shape[0]
    log_mean = torch.logsumexp(log_probs, dim=0) - math.log(num_models)
    if sampling_mode == "min":
        return log_probs.min(dim=0).values - log_mean
    # CV^2 = E[p^2] / E[p]^2 - 1, computed in log space since sequence
    # probabilities underflow in float32.
    log_second_moment = torch.logsumexp(2 * log_probs, dim=0) - math.log(num_models)
    cv = torch.expm1(log_second_moment - 2 * log_mean).clamp(min=0.0).sqrt()
    remaining = 1.0 - eta * cv
    return torch.where(
        remaining > 0,
        torch.log(remaining.clamp(min=torch.finfo(remaining.dtype).tiny)),
        torch.full_like(remaining, -torch.inf),
    )


class BestOfNGenerator(BaseGenerator):
    """Best of N Generator using rejection sampling with slot-based batching.

    Each round draws one ensemble member uniformly at random and generates
    proposals with it, so every proposal is a sample from the uniform mixture
    of members. All members then score the proposals and each one is accepted
    with probability ``acceptance_log_probs``. With top_p=1 and temperature=1
    the accepted samples are exact draws from the target; with top-p the
    proposal is the mixture of truncated members instead.
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
            sampling_mode: Target numerator f_out:
                - "min": f_out = min_l pi_l
                - "mean_std": f_out = max(0, mean_l pi_l - eta * std_l pi_l)
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
    ) -> torch.Tensor:
        """Compute log probs for sequences across all ensemble members.

        Args:
            model: EnsembleModel with shared backbone.
            input_ids: Full sequences (B, T).
            attention_mask: Attention mask (B, T).
            prompt_lengths: List of prompt lengths for each sequence.
            device: Device the model is on.

        Returns:
            (L, B) per-model sequence log-probabilities.
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

        return torch.stack(log_probs_per_model, dim=0)  # (L, B)

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
        """Generate candidates from one uniformly drawn ensemble member.

        Drawing the member independently each round makes every proposal a
        sample from the uniform mixture of members.
        """
        proposal_idx = int(torch.randint(model.num_models, (1,)).item())
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
            model_indices=[proposal_idx],
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
    ) -> torch.Tensor:
        """Score generated candidates with all ensemble members.

        Returns:
            (L, B) per-model sequence log-probabilities.
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
            response_tokens = int(output_mask[i, slot.prompt_length :].sum().item())
            accepted = u[i].item() <= alpha

            with state_lock:
                state = prompt_states[slot.prompt_idx]
                if slot.prompt_idx in results:
                    slots[slot_idx] = None
                    continue

                state["proposal_count"] += 1
                response_digest = hashlib.sha256(response.encode("utf-8")).digest()
                if response_digest in state["seen_responses"]:
                    state["duplicate_proposals"] += 1
                else:
                    state["seen_responses"].add(response_digest)

                state["finished_trials"] += 1
                if alpha > state["best_alpha"]:
                    state["best_alpha"] = alpha
                    state["best_response"] = response
                    state["best_response_tokens"] = response_tokens

                if accepted:
                    results[slot.prompt_idx] = response
                    state["accepted_response_tokens"] = response_tokens
                    slots[slot_idx] = None
                    state["accepted_by_rejection"] = True
                    accepted_this_iter += 1
                    total_accepted_delta += 1
                    accepted_trials_sum_delta += state["finished_trials"]
                elif state["finished_trials"] >= self.max_trials:
                    # Force accept best seen after max trials
                    results[slot.prompt_idx] = state["best_response"] or response
                    state["accepted_response_tokens"] = state["best_response_tokens"]
                    slots[slot_idx] = None
                    state["cap_hit"] = True
                    accepted_this_iter += 1
                    total_accepted_delta += 1
                    accepted_trials_sum_delta += state["finished_trials"]
                else:
                    slots[slot_idx] = None

        return accepted_this_iter, total_accepted_delta, accepted_trials_sum_delta

    def _init_prompt_states(self, n_prompts: int) -> dict[int, dict[str, Any]]:
        """Initialize per-prompt tracking state."""
        return {
            idx: {
                "best_alpha": -1.0,
                "best_response": None,
                "best_response_tokens": 0,
                "accepted_response_tokens": 0,
                "finished_trials": 0,
                "proposal_count": 0,
                "duplicate_proposals": 0,
                "seen_responses": set(),
                "accepted_by_rejection": False,
                "cap_hit": False,
            }
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
        generation_barrier: threading.Barrier,
        worker_timings: list[tuple[float, float]],
        timing_lock: threading.Lock,
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
        total_raw_alpha_sum = 0.0
        raw_alpha_above_one = 0
        max_raw_alpha = 0.0
        nonfinite_proposal_count = 0
        nonfinite_fout_count = 0
        min_proposal_lp = float("inf")
        max_proposal_lp = float("-inf")
        min_fout_lp = float("inf")
        max_fout_lp = float("-inf")

        with model.device_manager.request_gpu() as device:
            model.to(device)
            self._synchronize_device(device)
            generation_barrier.wait()
            worker_start = time.perf_counter()
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
                        log_probs_tensor = self._score_candidates(
                            model,
                            output_ids,
                            output_mask,
                            active_slots,
                            prompt_end_idx,
                            device,
                        )
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error: {e}")
                        raise

                    # Mixture proposal log-prob and target numerator, for
                    # diagnostics only; acceptance never needs their ratio.
                    proposal_lps = torch.logsumexp(log_probs_tensor, dim=0) - math.log(
                        log_probs_tensor.shape[0]
                    )
                    log_alphas = acceptance_log_probs(
                        log_probs_tensor, self.sampling_mode, self.eta
                    )
                    f_out_log = proposal_lps + log_alphas

                    proposal_finite = torch.isfinite(proposal_lps)
                    fout_finite = torch.isfinite(f_out_log)
                    nonfinite_proposal_count += int((~proposal_finite).sum().item())
                    nonfinite_fout_count += int((~fout_finite).sum().item())
                    if proposal_finite.any():
                        finite_proposals = proposal_lps[proposal_finite]
                        min_proposal_lp = min(
                            min_proposal_lp, finite_proposals.min().item()
                        )
                        max_proposal_lp = max(
                            max_proposal_lp, finite_proposals.max().item()
                        )
                    if fout_finite.any():
                        finite_fout = f_out_log[fout_finite]
                        min_fout_lp = min(min_fout_lp, finite_fout.min().item())
                        max_fout_lp = max(max_fout_lp, finite_fout.max().item())
                    # Exact ratios are <= 1; the clamp only absorbs rounding.
                    raw_alphas = torch.exp(log_alphas)
                    alphas = raw_alphas.clamp(0.0, 1.0)
                    total_raw_alpha_sum += raw_alphas.sum().item()
                    raw_alpha_above_one += int((raw_alphas > 1.0).sum().item())
                    max_raw_alpha = max(max_raw_alpha, raw_alphas.max().item())
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
                self._synchronize_device(device)
                worker_end = time.perf_counter()
                with timing_lock:
                    worker_timings.append((worker_start, worker_end))
                model.cpu()
                # Removed empty_cache to prevent illegal memory access
                pass

        logger.info(f"Worker {worker_id} finished. Processed {total_accepted} samples.")
        return {
            "total_accepted": total_accepted,
            "accepted_trials_sum": accepted_trials_sum,
            "total_alpha_sum": total_alpha_sum,
            "total_samples_scored": total_samples_scored,
            "total_raw_alpha_sum": total_raw_alpha_sum,
            "raw_alpha_above_one": raw_alpha_above_one,
            "max_raw_alpha": max_raw_alpha,
            "nonfinite_proposal_count": nonfinite_proposal_count,
            "nonfinite_fout_count": nonfinite_fout_count,
            "min_proposal_lp": min_proposal_lp,
            "max_proposal_lp": max_proposal_lp,
            "min_fout_lp": min_fout_lp,
            "max_fout_lp": max_fout_lp,
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
        generation_barrier = threading.Barrier(num_gpus)
        worker_timings: list[tuple[float, float]] = []
        timing_lock = threading.Lock()
        prompt_states = self._init_prompt_states(len(prompts))
        pending_queue = self._init_pending_queue(prompts, formatted_prompts)

        total_accepted = 0
        accepted_trials_sum = 0
        total_alpha_sum = 0.0
        total_samples_scored = 0
        total_raw_alpha_sum = 0.0
        raw_alpha_above_one = 0
        max_raw_alpha = 0.0
        nonfinite_proposal_count = 0
        nonfinite_fout_count = 0
        min_proposal_lp = float("inf")
        max_proposal_lp = float("-inf")
        min_fout_lp = float("inf")
        max_fout_lp = float("-inf")

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
                generation_barrier=generation_barrier,
                worker_timings=worker_timings,
                timing_lock=timing_lock,
            )

            # Aggregate stats
            for m in worker_stats:
                total_accepted += m["total_accepted"]
                accepted_trials_sum += m["accepted_trials_sum"]
                total_alpha_sum += m["total_alpha_sum"]
                total_samples_scored += m["total_samples_scored"]
                total_raw_alpha_sum += m["total_raw_alpha_sum"]
                raw_alpha_above_one += m["raw_alpha_above_one"]
                max_raw_alpha = max(max_raw_alpha, m["max_raw_alpha"])
                nonfinite_proposal_count += m["nonfinite_proposal_count"]
                nonfinite_fout_count += m["nonfinite_fout_count"]
                min_proposal_lp = min(min_proposal_lp, m["min_proposal_lp"])
                max_proposal_lp = max(max_proposal_lp, m["max_proposal_lp"])
                min_fout_lp = min(min_fout_lp, m["min_fout_lp"])
                max_fout_lp = max(max_fout_lp, m["max_fout_lp"])
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
                generation_barrier,
                worker_timings,
                timing_lock,
            )
            total_accepted += m["total_accepted"]
            accepted_trials_sum += m["accepted_trials_sum"]
            total_alpha_sum += m["total_alpha_sum"]
            total_samples_scored += m["total_samples_scored"]
            total_raw_alpha_sum += m["total_raw_alpha_sum"]
            raw_alpha_above_one += m["raw_alpha_above_one"]
            max_raw_alpha = max(max_raw_alpha, m["max_raw_alpha"])
            nonfinite_proposal_count += m["nonfinite_proposal_count"]
            nonfinite_fout_count += m["nonfinite_fout_count"]
            min_proposal_lp = min(min_proposal_lp, m["min_proposal_lp"])
            max_proposal_lp = max(max_proposal_lp, m["max_proposal_lp"])
            min_fout_lp = min(min_fout_lp, m["min_fout_lp"])
            max_fout_lp = max(max_fout_lp, m["max_fout_lp"])

        pbar.close()

        outputs = [
            {"prompt": prompts[i], "output": results.get(i, "")}
            for i in range(len(prompts))
        ]
        response_tokens = sum(
            int(state["accepted_response_tokens"]) for state in prompt_states.values()
        )
        metrics = self._generation_metrics(worker_timings, response_tokens)
        logger.info(
            "Generation timing: %.3f s, %d response tokens, %.3f ms/response token",
            metrics["generation_wall_time_s"],
            response_tokens,
            metrics["generation_ms_per_response_token"],
        )
        total_proposals = sum(
            state["proposal_count"] for state in prompt_states.values()
        )
        total_unique_proposals = sum(
            len(state["seen_responses"]) for state in prompt_states.values()
        )
        total_duplicate_proposals = sum(
            state["duplicate_proposals"] for state in prompt_states.values()
        )
        cap_hits = sum(state["cap_hit"] for state in prompt_states.values())
        rejection_accepts = sum(
            state["accepted_by_rejection"] for state in prompt_states.values()
        )
        num_prompts = len(prompts)
        duplicate_rate = (
            total_duplicate_proposals / total_proposals if total_proposals else 0.0
        )
        logger.info(
            "Proposal diagnostics: avg_proposals=%.2f, avg_unique=%.2f, "
            "duplicate_rate=%.3f, cap_hit_rate=%.3f, rejection_accept_rate=%.3f",
            total_proposals / num_prompts if num_prompts else 0.0,
            total_unique_proposals / num_prompts if num_prompts else 0.0,
            duplicate_rate,
            cap_hits / num_prompts if num_prompts else 0.0,
            rejection_accepts / num_prompts if num_prompts else 0.0,
        )
        logger.info(
            "Raw alpha diagnostics: avg_raw_alpha=%.3f, raw_alpha_gt1_rate=%.3f, "
            "max_raw_alpha=%.3f",
            total_raw_alpha_sum / total_samples_scored if total_samples_scored else 0.0,
            raw_alpha_above_one / total_samples_scored if total_samples_scored else 0.0,
            max_raw_alpha,
        )
        logger.info(
            "Log-probability diagnostics: nonfinite_proposal=%d, "
            "nonfinite_fout=%d, proposal_range=[%.1f, %.1f], "
            "fout_range=[%.1f, %.1f]",
            nonfinite_proposal_count,
            nonfinite_fout_count,
            min_proposal_lp,
            max_proposal_lp,
            min_fout_lp,
            max_fout_lp,
        )
        logger.info(f"Successfully generated {len(outputs)} responses.")

        return outputs, {
            **metrics,
            "avg_try": accepted_trials_sum / total_accepted if total_accepted else 0.0,
            "avg_alpha": total_alpha_sum / total_samples_scored
            if total_samples_scored
            else 0.0,
            "avg_proposals": total_proposals / num_prompts if num_prompts else 0.0,
            "avg_unique_proposals": (
                total_unique_proposals / num_prompts if num_prompts else 0.0
            ),
            "proposal_duplicate_rate": duplicate_rate,
            "cap_hit_rate": cap_hits / num_prompts if num_prompts else 0.0,
            "rejection_accept_rate": (
                rejection_accepts / num_prompts if num_prompts else 0.0
            ),
            "avg_raw_alpha": (
                total_raw_alpha_sum / total_samples_scored
                if total_samples_scored
                else 0.0
            ),
            "raw_alpha_gt1_rate": (
                raw_alpha_above_one / total_samples_scored
                if total_samples_scored
                else 0.0
            ),
            "max_raw_alpha": max_raw_alpha,
            "nonfinite_proposal_count": nonfinite_proposal_count,
            "nonfinite_fout_count": nonfinite_fout_count,
            "min_proposal_lp": min_proposal_lp,
            "max_proposal_lp": max_proposal_lp,
            "min_fout_lp": min_fout_lp,
            "max_fout_lp": max_fout_lp,
        }

    def get_name(self) -> str:
        """Get generator name for file naming."""
        parts: list[str] = [f"bon-{self.sampling_mode}"]
        if self.sampling_mode == "mean_std":
            parts.append(f"eta{self.eta}")
        # "mix" marks the exact mixture-proposal sampler; earlier response
        # files (member-0 proposal, alpha/beta penalty) lack it, so
        # overwrite=false never reuses them.
        parts.append("mix")
        parts.append(f"n{self.max_trials}")
        parts.append(super().get_name())
        return "-".join(parts)
