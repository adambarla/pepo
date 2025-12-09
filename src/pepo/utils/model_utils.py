import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def get_log_probs(
    model: AutoModelForCausalLM,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Get the log probabilities of the response tokens.

    Args:
        model: The causal LM model.
        device: Device to run computation on.
        input_ids: Input token IDs (B, T).
        attention_mask: Attention mask (B, T).
        response_mask: Mask indicating response tokens to sum over (B, T).

    Returns:
        torch.Tensor: Sum of log probabilities for the response tokens (B,).
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore[operator]
    logits = outputs.logits

    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    response_mask = response_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs = log_probs * response_mask.float().to(device)
    log_probs_sum = log_probs.sum(dim=-1)

    return log_probs_sum
