import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def get_log_probs(
    model: PreTrainedModel | PeftModel,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    debug: bool = False,
) -> torch.Tensor:
    """
    Get the log probabilities of the response tokens using
    memory-efficient CrossEntropyLoss.

    Args:
        model: The causal LM model.
        device: Device to run computation on.
        input_ids: Input token IDs (B, T).
        attention_mask: Attention mask (B, T).
        response_mask: Mask indicating response tokens to sum over (B, T).
        debug: If True, compute loss both ways and compare (for verification).

    Returns:
        torch.Tensor: Sum of log probabilities for the response tokens (B,).
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    response_mask = response_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    nll = nll.view(shift_labels.shape)

    log_probs = nll * -1.0

    log_probs = log_probs * shift_mask.float()
    log_probs_sum = log_probs.sum(dim=-1)

    if debug:
        log_probs_old = F.log_softmax(shift_logits, dim=-1)
        log_probs_old = log_probs_old.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        log_probs_old = log_probs_old * shift_mask.float()
        log_probs_sum_old = log_probs_old.sum(dim=-1)

        # check allclose
        if not torch.allclose(log_probs_sum, log_probs_sum_old, atol=1e-5):
            max_diff = (log_probs_sum - log_probs_sum_old).abs().max().item()
            mean_diff = (log_probs_sum - log_probs_sum_old).abs().mean().item()
            logger.error(
                f"DEBUG: CrossEntropyLoss vs log_softmax comparison - "
                f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
            )
    return log_probs_sum


def get_next_token_log_probs(
    model: PreTrainedModel | PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Optional[list[torch.Tensor]] = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    """Compute log probabilities for the next token (the very last position).

    Args:
        model: The model to use for prediction.
        input_ids: (B, T) input IDs.
        attention_mask: (B, T) attention mask.
        past_key_values: Optional past key values for caching.
        use_cache: Whether to use KV caching.

    Returns:
        Tuple of (
            (B, V) log probabilities for the last token,
            Optional past_key_values
        ).
    """
    if past_key_values is not None:
        # If we have cache, we only need to pass the last token
        # But we still need the full attention mask for position embeddings
        model_kwargs = {
            "use_cache": use_cache,
            "past_key_values": past_key_values,
        }
        # Only pass the last token
        input_ids_step = input_ids[:, -1:]
    else:
        model_kwargs = {"use_cache": use_cache}
        input_ids_step = input_ids

    outputs = model(
        input_ids=input_ids_step,
        attention_mask=attention_mask,
        **model_kwargs,
    )
    logits = outputs.logits  # (B, T_step, V)
    last_logits = logits[:, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)

    return log_probs, outputs.past_key_values


def top_p_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample from logits using top-p (nucleus) sampling.

    Args:
        logits: (B, V) logits for each token in vocabulary.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        (B,) sampled token indices.
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.clone()
    logits[indices_to_remove] = float("-inf")
    filtered_probs = F.softmax(logits, dim=-1)
    sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    return sampled_indices
