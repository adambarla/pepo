import logging
from typing import Any, Optional, cast

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class UltraFeedbackProcessor:
    """Processor for UltraFeedback dataset. Applies chat templates and filters."""

    def __init__(
        self, max_length: Optional[int] = None, max_prompt_length: Optional[int] = None
    ):
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def process(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        original_size = len(dataset)
        processed = dataset.map(
            lambda ex: self._process_batch(ex, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing",
        )
        logger.info(
            f"Dataset: {original_size} -> {len(processed)} examples "
            f"(filtered {original_size - len(processed)})"
        )
        return processed

    def _process_batch(
        self, examples: dict[str, list[Any]], tokenizer: PreTrainedTokenizerBase
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {
            "prompt_text": [],
            "chosen_text": [],
            "rejected_text": [],
        }
        for i in range(len(examples["prompt"])):
            prompt = self._ensure_message_list(examples["prompt"][i], is_prompt=True)
            chosen = self._ensure_message_list(examples["chosen"][i])
            rejected = self._ensure_message_list(examples["rejected"][i])
            if prompt is None or chosen is None or rejected is None:
                continue
            prompt_str = cast(
                str,
                tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                ),
            )
            chosen_str = (
                cast(str, tokenizer.apply_chat_template(chosen, tokenize=False))
                + tokenizer.eos_token
            )
            rejected_str = (
                cast(str, tokenizer.apply_chat_template(rejected, tokenize=False))
                + tokenizer.eos_token
            )
            if not self._is_valid_length(
                prompt_str, chosen_str, rejected_str, tokenizer
            ):
                continue
            result["prompt_text"].append(prompt_str)
            result["chosen_text"].append(chosen_str)
            result["rejected_text"].append(rejected_str)
        return result

    def _ensure_message_list(
        self, messages: Any, is_prompt: bool = False
    ) -> list[dict[str, str]] | None:
        if isinstance(messages, list) and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in messages
        ):
            return messages
        if isinstance(messages, str) and is_prompt:
            return [{"role": "user", "content": messages}]
        return None

    def _is_valid_length(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> bool:
        if self.max_prompt_length is not None:
            if (
                len(tokenizer(prompt, truncation=False)["input_ids"])
                > self.max_prompt_length
            ):
                return False
        if self.max_length is not None:
            if len(tokenizer(chosen, truncation=False)["input_ids"]) > self.max_length:
                return False
            if (
                len(tokenizer(rejected, truncation=False)["input_ids"])
                > self.max_length
            ):
                return False
        return True
