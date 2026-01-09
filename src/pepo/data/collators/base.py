import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DataCollator:
    """Tokenizes and pads sequences on-the-fly using fast tokenizer optimization."""

    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
    ):
        self._tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        return self._tokenizer

    def set_tokenizer(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts = [f["prompt_text"] for f in features]
        chosen_texts = [f["chosen_text"] for f in features]
        reject_texts = [f["rejected_text"] for f in features]

        prompt_encoded = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=self.max_prompt_length is not None,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        chosen_encoded = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        reject_encoded = self.tokenizer(
            reject_texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )

        prompt_mask = prompt_encoded["attention_mask"]
        chosen_att_mask = chosen_encoded["attention_mask"]
        reject_att_mask = reject_encoded["attention_mask"]
        T_p = prompt_mask.shape[1]

        chosen_resp_mask = (
            torch.cat([prompt_mask, torch.zeros_like(chosen_att_mask[:, T_p:])], dim=-1)
            ^ chosen_att_mask
        )
        reject_resp_mask = (
            torch.cat([prompt_mask, torch.zeros_like(reject_att_mask[:, T_p:])], dim=-1)
            ^ reject_att_mask
        )

        batch = {
            "prompt_input_ids": prompt_encoded["input_ids"],
            "chosen_input_ids": chosen_encoded["input_ids"],
            "rejected_input_ids": reject_encoded["input_ids"],
            "prompt_attention_mask": prompt_encoded["attention_mask"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_attention_mask": reject_encoded["attention_mask"],
            "chosen_response_mask": chosen_resp_mask,
            "rejected_response_mask": reject_resp_mask,
        }

        if "reference_chosen_logps" in features[0]:
            batch["reference_chosen_logps"] = torch.tensor(
                [f["reference_chosen_logps"] for f in features], dtype=torch.float
            )
        if "reference_rejected_logps" in features[0]:
            batch["reference_rejected_logps"] = torch.tensor(
                [f["reference_rejected_logps"] for f in features], dtype=torch.float
            )

        return batch
