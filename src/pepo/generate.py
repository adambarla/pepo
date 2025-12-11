import logging
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


class Generator:
    """Simple generator class for producing model responses from instructions."""

    def __init__(
        self,
        max_new_tokens: int = 1000,
        use_ensamble: bool = True,
        batch_size: int = 10,
        greedy_sampling: bool = True,
        top_p_sampling: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """
        Initialize generator.

        Args:
            max_new_tokens: Maximum number of new tokens to generate.
            use_ensamble: Whether to use ensemble generation.
            batch_size: Batch size for generation.
            greedy_sampling: Whether to use greedy sampling.
            top_p_sampling: Whether to use top-p sampling.
            temperature: Sampling temperature for top-p sampling.
            top_p: Top-p sampling parameter (nucleus sampling).
        """
        if greedy_sampling and top_p_sampling:
            raise ValueError(
                "Greedy sampling and top-p sampling cannot be used together"
            )
        if not greedy_sampling and not top_p_sampling:
            raise ValueError("Either greedy sampling or top-p sampling must be used")
        self.max_new_tokens = max_new_tokens
        self.use_ensamble = use_ensamble
        self.batch_size = batch_size
        self.greedy_sampling = greedy_sampling
        self.top_p_sampling = top_p_sampling
        self.temperature = temperature
        self.top_p = top_p

    def _top_p_sampling(
        self, logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0
    ) -> torch.Tensor:
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
        logits[indices_to_remove] = float("-inf")
        filtered_probs = F.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        return sampled_indices

    def generate(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 1024,
        use_ensamble: bool = True,
        sample_missing_token: bool = False,
        greedy_sampling: bool = False,
        top_p_sampling: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens from input_ids using the model.
        """
        if greedy_sampling and top_p_sampling:
            raise ValueError(
                "Greedy sampling and top-p sampling cannot be used together"
            )
        if attention_mask is None:
            attention_mask = (input_ids != model.tokenizer.pad_token_id).float()

        batch_size = input_ids.shape[0]

        device_input_ids = []
        device_attention_masks = []
        for model_idx in range(model.num_networks):
            device = torch.device(model.device_manager.get_device_for_model(model_idx))
            device_input_ids.append(input_ids.to(device))
            device_attention_masks.append(attention_mask.to(device))

        stop_signal = torch.zeros(batch_size, dtype=torch.bool).cpu()

        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        pbar = tqdm(range(max_length - input_ids.shape[1]), disable=disable_tqdm)
        for i in pbar:
            if i > 0 and i % 100 == 0:
                model.device_manager.clear_cache()

            if use_ensamble:
                log_probs = model.predict(device_input_ids, device_attention_masks)
            else:
                submodel = model.models[0]
                with torch.no_grad():
                    with submodel.disable_adapter():
                        submodel.eval()
                        log_probs = model._predict_submodel(
                            submodel, device_input_ids[0], device_attention_masks[0]
                        )
            min_probs = torch.exp(log_probs)

            if sample_missing_token:
                missing_token_id = model.tokenizer.vocab_size
                missing_probs = torch.clamp(1 - torch.sum(min_probs, dim=-1), min=0.0)
                min_probs = torch.cat([min_probs, missing_probs.unsqueeze(-1)], dim=-1)
                missing_mask = torch.ones(
                    batch_size, dtype=torch.bool, device=min_probs.device
                )
                sampled_token_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=min_probs.device
                )
                while True:
                    new_sampled_token_ids = torch.multinomial(
                        min_probs[missing_mask], num_samples=1
                    ).squeeze(-1)
                    sampled_token_ids[missing_mask] = new_sampled_token_ids
                    missing_mask = sampled_token_ids == missing_token_id
                    if not torch.any(missing_mask):
                        break
            else:
                if greedy_sampling:
                    sampled_token_ids = torch.argmax(min_probs, dim=-1)
                elif top_p_sampling:
                    sampled_token_ids = self._top_p_sampling(
                        log_probs,
                        top_p=top_p,
                        temperature=temperature,
                    )
                else:
                    min_probs = min_probs / torch.sum(min_probs, dim=-1, keepdim=True)
                    sampled_token_ids = torch.multinomial(min_probs, num_samples=1)

            stop_signal = stop_signal.to(device=sampled_token_ids.device) | (
                sampled_token_ids == model.tokenizer.eos_token_id
            )
            for model_idx in range(model.num_networks):
                device = torch.device(
                    model.device_manager.get_device_for_model(model_idx)
                )
                new_token_tensor = sampled_token_ids.to(device).unsqueeze(-1)
                device_input_ids[model_idx] = torch.cat(
                    [device_input_ids[model_idx], new_token_tensor], dim=1
                )
                device_attention_masks[model_idx] = torch.cat(
                    [
                        device_attention_masks[model_idx],
                        ~stop_signal.unsqueeze(-1).to(device),
                    ],
                    dim=1,
                )

            pbar.set_postfix({"stopped": f"{stop_signal.sum().item()}/{batch_size}"})

            if sampled_token_ids[0] == model.tokenizer.eos_token_id:
                logger = logging.getLogger(__name__)
                logger.debug(f"Generated EOS token at step {i}")
            if torch.all(stop_signal):
                break
        logger = logging.getLogger(__name__)
        decoded = model.tokenizer.decode(
            device_input_ids[0].cpu()[0], skip_special_tokens=True
        )
        logger.debug(f"Generated sequence idx=0:\n{decoded}")

        return device_input_ids[0].cpu(), device_attention_masks[0].cpu()

    def get_name(self) -> str:
        """
        Get generator name.
        """
        parts = []
        parts.append(f"mt{self.max_new_tokens}")
        if self.greedy_sampling:
            parts.append("greedy")
        if self.top_p_sampling:
            parts.append("top-p")
            parts.append(f"t{self.temperature}")
            parts.append(f"p{self.top_p}")
        if not self.use_ensamble:
            parts.append("single")
        return "_".join(parts)

    def generate_responses(
        self,
        model: Any,
        prompts: list[str],
        apply_chat_template: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for a list of instructions.

        Args:
            model: Model instance with generate() method and get_tokenizer() method.
            prompts: List of prompt strings.
            apply_chat_template: Whether to apply chat template to prompts.

        Returns:
            List of dictionaries with 'prompt' and 'output' keys.
        """
        tokenizer = model.get_tokenizer()
        outputs = []
        logger = logging.getLogger(__name__)

        logger.info(f"Generating responses for {len(prompts)} prompts")
        logger.info("Generation parameters:")
        logger.info(f"  max_new_tokens: {self.max_new_tokens}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  use_ensamble: {self.use_ensamble}")
        logger.info(f"  apply_chat_template: {apply_chat_template}")
        if self.greedy_sampling:
            logger.info("  greedy_sampling: true")
        if self.top_p_sampling:
            logger.info("  top_p_sampling: true")
            logger.info(f"  temperature: {self.temperature}")
            logger.info(f"  top_p: {self.top_p}")

        for i in range(0, len(prompts), self.batch_size):
            batch_num = i // self.batch_size + 1
            total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
            logger.info(f"Generating batch {batch_num}/{total_batches}")

            batch_prompts = prompts[i : i + self.batch_size]

            if apply_chat_template:
                formatted_batch_prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in batch_prompts
                ]
            else:
                formatted_batch_prompts = batch_prompts

            tokenizer.padding_side = "left"
            tokenized = tokenizer(
                formatted_batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            max_total_length = input_ids.shape[1] + self.max_new_tokens
            output_ids, output_mask = self.generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_total_length,
                use_ensamble=self.use_ensamble,
                top_p_sampling=self.top_p_sampling,
                greedy_sampling=self.greedy_sampling,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            starting_idx = input_ids.shape[1]
            output_mask[:, :starting_idx] = False
            output_ids = output_ids.where(output_mask.bool(), tokenizer.pad_token_id)

            for j, prompt in enumerate(batch_prompts):
                response = tokenizer.decode(output_ids[j], skip_special_tokens=True)
                outputs.append({"prompt": prompt, "output": response})

        logger.info(f"Successfully generated {len(outputs)} responses")

        return outputs
