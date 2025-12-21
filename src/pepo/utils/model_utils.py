import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def get_log_probs(
    model: AutoModelForCausalLM,
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
    return log_probs_sum  # type: ignore[no-any-return]
