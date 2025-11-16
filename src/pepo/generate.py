from typing import Optional

from .utils import Logger


class Generator:
    """Simple generator class for producing model responses from instructions."""

    def __init__(
        self,
        max_new_tokens: int = 512,
        use_ensamble: bool = True,
        batch_size: int = 10,
        greedy_sampling: bool = True,
        top_p_sampling: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        logger: Optional[Logger] = None,
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
            logger: Optional logger instance.
        """
        if greedy_sampling and top_p_sampling:
            raise ValueError("Greedy sampling and top-p sampling cannot be used together")
        if not greedy_sampling and not top_p_sampling:
            raise ValueError("Either greedy sampling or top-p sampling must be used")
        self.max_new_tokens = max_new_tokens
        self.use_ensamble = use_ensamble
        self.batch_size = batch_size
        self.greedy_sampling = greedy_sampling
        self.top_p_sampling = top_p_sampling
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logger

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
        model,
        prompts: list[str],
        apply_chat_template: bool = True,
    ) -> list[dict]:
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

        if self.logger:
            self.logger.info(f"Generating responses for {len(prompts)} prompts")
            self.logger.info("Generation parameters:")
            self.logger.info(f"  max_new_tokens: {self.max_new_tokens}")
            self.logger.info(f"  batch_size: {self.batch_size}")
            self.logger.info(f"  use_ensamble: {self.use_ensamble}")
            self.logger.info(f"  apply_chat_template: {apply_chat_template}")
            if self.greedy_sampling:
                self.logger.info("  greedy_sampling: true")
            if self.top_p_sampling:
                self.logger.info("  top_p_sampling: true")
                self.logger.info(f"  temperature: {self.temperature}")
                self.logger.info(f"  top_p: {self.top_p}")

        for i in range(0, len(prompts), self.batch_size):
            if self.logger:
                batch_num = i // self.batch_size + 1
                total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
                self.logger.info(f"Generating batch {batch_num}/{total_batches}")

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
            output_ids, output_mask = model.generate(
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

        if self.logger:
            self.logger.info(f"Successfully generated {len(outputs)} responses")

        return outputs
