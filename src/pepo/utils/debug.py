from typing import Optional

import torch
from transformers import AutoTokenizer

from .logger import Logger


def debug_tokens(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    attn_mask_shifted: torch.Tensor,
    response_mask: torch.Tensor,
    log_probs: torch.Tensor,
    logger: Optional[Logger] = None,
) -> None:
    """
    Debug function to decode and display tokens with their attention masks, response masks, and log probabilities.

    Displays tokens in rows of 10 tokens, followed by a row showing the attention mask, response mask, and log prob values.

    Args:
        input_ids: Input token IDs tensor of shape (B, T)
        tokenizer: Tokenizer to decode tokens
        attn_mask_shifted: Attention mask tensor of shape (B, T-1)
        response_mask: Response mask tensor of shape (B, T-1)
        log_probs: Log probabilities tensor of shape (B, T-1)
        logger: Optional logger to output debug information
    """
    B = input_ids.shape[0]

    for b in range(B):
        token_debug_str = ""
        decoded_tokens_chosen = [tokenizer.decode(token) for token in input_ids[b][1:]]
        for i in range(len(decoded_tokens_chosen)):
            if input_ids[b][i + 1] == tokenizer.eos_token_id:
                decoded_tokens_chosen[i] = "<eos>"
            elif input_ids[b][i + 1] == tokenizer.pad_token_id:
                decoded_tokens_chosen[i] = "<pad>"
            elif input_ids[b][i + 1] == tokenizer.bos_token_id:
                decoded_tokens_chosen[i] = "<bos>"
        decoded_tokens_chosen = [
            token.replace("\n", "<nline>") for token in decoded_tokens_chosen
        ]
        longest_token = max(len(token) for token in decoded_tokens_chosen)
        K = 10
        for i in range(0, len(decoded_tokens_chosen), K):
            tokens = decoded_tokens_chosen[i : i + K]
            a_mask = attn_mask_shifted[b][i : i + K]
            r_mask = response_mask[b][i : i + K]
            l_prob = log_probs[b][i : i + K]
            k = min(K, len(tokens))
            for j in range(k):
                token_debug_str += f"{tokens[j]:<{longest_token}}|"
            token_debug_str += "\n"
            for j in range(k):
                s = f"{int(a_mask[j])}, {int(r_mask[j])}, {l_prob[j]:.1f}"
                token_debug_str += f"{s:<{longest_token}}|"
            token_debug_str += "\n"
            token_debug_str += "-" * (longest_token * k + k) + "\n"
            if "<eot>" in tokens:
                break
        if logger:
            logger.debug(f"Token debug for prompt {b}: \n{token_debug_str}")


def initialize_lora_for_testing(
    model: torch.nn.Module,
    std: float = 0.1,
    logger: Optional[Logger] = None,
) -> int:
    """
    Initialize LoRA weights for testing purposes.

    Note: LoRA modification is W + (B @ A) * (alpha / r), so both A and B need to be
    non-zero to see an effect. This function initializes both A and B matrices with
    small random values.

    Args:
        model: Model with LoRA adapters (PeftModel)
        std: Standard deviation for random initialization (default: 0.1)
        logger: Optional logger to output debug information

    Returns:
        Number of LoRA parameter groups initialized
    """
    if logger:
        logger.debug(
            f"Initializing LoRA weights for testing (model: {model.__class__.__name__})"
        )

    lora_param_count = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_param_count += 1
            with torch.no_grad():
                if "lora_A" in name:
                    param.normal_(mean=0.0, std=std)
                    if logger:
                        logger.debug(
                            f"Initialized {name} shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}"
                        )
                elif "lora_B" in name:
                    param.normal_(mean=0.0, std=std)
                    if logger:
                        logger.debug(
                            f"Initialized {name} shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}"
                        )

    if logger:
        logger.debug(f"Initialized {lora_param_count} LoRA parameter groups")

    return lora_param_count
