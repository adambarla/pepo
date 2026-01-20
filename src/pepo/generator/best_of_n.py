"""Best of N Generator using rejection sampling with slot-based batching."""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import torch
from tqdm import tqdm

from .base import BaseGenerator

if TYPE_CHECKING:
    from ..model import BaseModel

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
    trials: int = 0
    best_response: Optional[str] = None
    best_alpha: float = -1.0


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
        model: "BaseModel",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probabilities for sequences across all ensemble members.

        Assumes shared_backbone mode (adapter switching on single model).
        Uses batched processing with model.eval_batch_size.

        Args:
            model: BaseModel instance (ensemble with shared backbone).
            input_ids: Full sequences (B, T).
            attention_mask: Attention mask (B, T).
            prompt_lengths: List of prompt lengths for each sequence.

        Returns:
            Tuple of (proposal_log_probs, min_log_probs, all_log_probs).
        """
        from ..utils.model_utils import get_log_probs

        # Cast to Any to access EnsembleModel attributes
        model_any = cast(Any, model)
        num_seqs = input_ids.shape[0]
        eval_batch_size = model.eval_batch_size

        # Create response mask for each sequence
        response_mask = torch.zeros_like(input_ids, dtype=torch.float)
        for i, pl in enumerate(prompt_lengths):
            response_mask[i, pl:] = attention_mask[i, pl:].float()

        # log_probs_per_model[model_idx] = tensor of shape (B,)
        log_probs_per_model: list[torch.Tensor] = []
        shared_model = model_any.models[0]

        # Use architecture's DeviceManager to request GPU
        with model.device_manager.request_gpu() as device:
            shared_model.to(device)
            shared_model.eval()

            try:
                with torch.no_grad():
                    for model_idx in range(model_any.num_models):
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
        pending: deque[tuple[int, Any, str]],
        tokenizer: Any,
        gen_batch_size: int,
    ) -> int:
        """Fill empty slots with new prompts from pending queue."""
        filled_count = 0
        for i in range(gen_batch_size):
            if slots[i] is None and pending:
                prompt_idx, prompt, formatted = pending.popleft()
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
                )
                filled_count += 1
        return filled_count

    def _generate_candidates(
        self,
        model: "BaseModel",
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
        output_ids, output_mask = cast(Any, model).generate(
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
        model: "BaseModel",
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
    ) -> tuple[int, int, int]:
        """Process acceptance/rejection logic for candidates."""
        u = torch.rand(len(active_slots))
        accepted_this_iter = 0
        accepted_trials_sum_delta = 0
        total_accepted_delta = 0

        for i, (slot_idx, slot) in enumerate(zip(active_indices, active_slots)):
            alpha = alphas[i].item()
            slot.trials += 1

            if alpha > slot.best_alpha:
                slot.best_alpha = alpha
                slot.best_response = self._extract_response(
                    output_ids[i], output_mask[i], slot.prompt_length, tokenizer
                )

            accepted = u[i].item() <= alpha
            force_accept = slot.trials >= self.max_trials

            if accepted:
                results[slot.prompt_idx] = self._extract_response(
                    output_ids[i], output_mask[i], slot.prompt_length, tokenizer
                )
            elif force_accept:
                if slot.best_response is not None:
                    results[slot.prompt_idx] = slot.best_response
                else:
                    results[slot.prompt_idx] = self._extract_response(
                        output_ids[i], output_mask[i], slot.prompt_length, tokenizer
                    )

            if accepted or force_accept:
                slots[slot_idx] = None
                accepted_this_iter += 1
                total_accepted_delta += 1
                accepted_trials_sum_delta += slot.trials

        return accepted_this_iter, total_accepted_delta, accepted_trials_sum_delta

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

        Args:
            model: BaseModel instance (should be an ensemble model).
            prompts: List of prompts or histories.
            apply_chat_template: Whether to use chat template.
            token_callback: Callback for streaming (not used for Best-of-N).

        Returns:
            List of dicts with 'prompt' and 'output'.
        """
        tokenizer = model.get_tokenizer()
        prompts, formatted_prompts = self._process_prompts(
            prompts, tokenizer, apply_chat_template
        )

        # Results dict: prompt_idx -> response
        results: dict[int, str] = {}

        # Pending queue: (prompt_idx, prompt, formatted)
        pending: deque[tuple[int, Any, str]] = deque()
        for idx, (p, f) in enumerate(zip(prompts, formatted_prompts)):
            pending.append((idx, p, f))

        # Active slots
        gen_batch_size = model.generation_batch_size
        slots: list[Optional[Slot]] = [None] * gen_batch_size

        logger.info(
            f"Best-of-N generation (max_trials={self.max_trials}, "
            f"batch_size={gen_batch_size}) for {len(prompts)} prompts"
        )

        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        pbar = tqdm(
            total=len(prompts),
            desc="Best-of-N",
            disable=disable_tqdm,
            leave=False,
        )

        iteration = 0
        total_accepted = 0
        accepted_trials_sum = 0
        total_alpha_sum = 0.0
        total_samples_scored = 0

        while pending or any(s is not None for s in slots):
            iteration += 1

            filled_count = self._fill_slots(slots, pending, tokenizer, gen_batch_size)

            active_indices = [i for i, s in enumerate(slots) if s is not None]
            if not active_indices:
                break
            active_slots = [slots[i] for i in active_indices]
            # Type checker assertion
            active_slots = cast(list[Slot], active_slots)
            n_active = len(active_slots)

            logger.info(
                f"Iteration {iteration}: {n_active} active slots, "
                f"{len(pending)} pending, {filled_count} newly filled"
            )

            logger.info(f"Generating {n_active} candidates with proposal model...")
            output_ids, output_mask, prompt_end_idx = self._generate_candidates(
                model, active_slots, tokenizer
            )

            num_models = getattr(model, "num_models", 1)
            logger.info(f"Scoring {n_active} candidates with {num_models} adapters...")
            proposal_lps, min_lps, _ = self._score_candidates(
                model, output_ids, output_mask, active_slots, prompt_end_idx
            )

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
            )
            pbar.update(accepted_delta)

            total_accepted += accepted_delta
            accepted_trials_sum += trials_delta

            avg_alpha = alphas.mean().item()

            total_alpha_sum += alphas.sum().item()
            total_samples_scored += n_active

            avg_try = (
                accepted_trials_sum / total_accepted if total_accepted > 0 else 0.0
            )

            logger.info(
                f"Iteration {iteration}: {accepted_iter}/{n_active} "
                f"accepted, alpha={avg_alpha:.3f}, "
                f"total={total_accepted}/{len(prompts)}, "
                f"avg_try={avg_try:.1f}"
            )

            pbar.set_postfix(
                {
                    "iter": iteration,
                    "active": n_active - accepted_iter,
                    "alpha": f"{avg_alpha:.3f}",
                    "avg_try": f"{avg_try:.1f}",
                }
            )

        pbar.close()

        # Build output list
        outputs = [
            {"prompt": prompts[i], "output": results.get(i, "")}
            for i in range(len(prompts))
        ]

        logger.info(
            f"Successfully generated {len(outputs)} responses in {iteration} iterations"
        )

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
