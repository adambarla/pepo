import json
from typing import Optional

from datasets import load_dataset

from ..generate import Generator
from ..utils import sanitize_filename
from .base import BaseEvaluator


class AlpacaEvalEvaluator(BaseEvaluator):
    """AlpacaEval evaluator implementation."""

    def __init__(
        self,
        model,
        dataset_id: str,
        dataset_name: str,
        dataset_split: str,
        instruction_key: str,
        output_dir: str,
        generator: Generator,
        num_samples: Optional[int] = None,
        logger=None,
    ):
        """
        Initialize AlpacaEval evaluator.

        Args:
            model: PEPOModel instance.
            dataset_id: HuggingFace dataset ID.
            dataset_name: Dataset configuration name (e.g., "alpaca_eval").
            dataset_split: Dataset split to use.
            instruction_key: Key to extract instruction from dataset items.
            output_dir: Directory to save outputs.
            generator: Generator instance for response generation.
            num_samples: Optional number of samples to use (None = full dataset).
            logger: Optional logger instance.
        """
        super().__init__(
            model=model,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            instruction_key=instruction_key,
            output_dir=output_dir,
            generator=generator,
            logger=logger,
        )
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        if logger is not None:
            self.generator.logger = logger
        self.dataset = self._load_dataset()
        self.responses_filename, self.results_filename = self._generate_filename()

    def responses_exist(self, **kwargs) -> bool:
        """
        Check if responses file already exists.

        Args:
            **kwargs: Additional generation parameters (ignored, generator is from config).

        Returns:
            True if responses file exists, False otherwise.
        """
        return (self.output_dir / self.responses_filename).exists()

    def _generate_filename(self, **kwargs) -> tuple[str, str]:
        """
        Generate file names based on model name and generation configuration.

        Args:
            model: PEPOModel instance.
            max_new_tokens: Maximum number of new tokens.
            use_ensamble: Whether ensemble is used.
            temperature: Sampling temperature (optional).
            top_p: Top-p sampling parameter (optional).
            **kwargs: Additional generation parameters.

        Returns:
            Tuple of (responses_filename, results_filename).
        """
        model_name = self.model._get_model_name() + f"-e{self.model.get_min_epochs()}"
        model_name = sanitize_filename(model_name)

        parts = [self.dataset_name, model_name]

        parts.append(f"ns{self.num_samples}")
        parts.append(self.generator.get_name())

        base_name = "_".join(parts)
        responses_filename = f"{base_name}_responses.json"
        results_filename = f"{base_name}_results.json"

        return responses_filename, results_filename

    def _load_dataset(self):
        """
        Load dataset from HuggingFace.
        """
        if self.logger:
            self.logger.info(f"Loading dataset: {self.dataset_id}")
        dataset = load_dataset(
            self.dataset_id,
            split=self.dataset_split,
            trust_remote_code=True,
        )
        # Limit number of samples if specified
        if self.num_samples is not None and self.num_samples > 0:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
            if self.logger:
                self.logger.info(f"Using {self.num_samples} samples from dataset")
        else:
            if self.logger:
                self.logger.info(f"Using full dataset with {len(dataset)} samples")
        return dataset

    def generate_responses(self, **kwargs):
        """
        Generate responses for AlpacaEval dataset.

        Args:
            **kwargs: Additional generation parameters (ignored, generator is from config).

        Returns:
            Path to the responses file.
        """
        responses_filename, results_filename = self._generate_filename(
            model=self.model,
            max_new_tokens=self.generator.max_new_tokens,
            use_ensamble=self.generator.use_ensamble,
            **kwargs,
        )

        self.responses_file = self.output_dir / responses_filename
        self.results_file = self.output_dir / results_filename

        if self.logger:
            self.logger.info("Starting AlpacaEval response generation")
            self.logger.info("Generated file names:")
            self.logger.info(f"  Responses: {self.responses_file}")
            self.logger.info(f"  Results: {self.results_file}")

        instructions = [item[self.instruction_key] for item in self.dataset]
        if self.logger:
            self.logger.info(f"Loaded {len(instructions)} instructions")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        outputs = self.generator.generate_responses(
            model=self.model,
            prompts=instructions,
            apply_chat_template=True,
        )

        if self.logger:
            self.logger.info(f"Saving results to {self.responses_file}")
        with open(self.responses_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

        if self.logger:
            self.logger.info(f"Successfully generated {len(outputs)} responses")
            self.logger.info(f"Output saved to: {self.responses_file}")

        return self.responses_file

    def evaluate(self):
        """
        Evaluate responses using AlpacaEval.
        This is a placeholder for future implementation.

        Args:
            responses_file: Path to the responses file.

        Returns:
            Path to the results file.
        """
        if self.logger:
            self.logger.info("Evaluation functionality will be implemented later")
            self.logger.info(f"Responses file: {self.responses_file}")
            self.logger.info(f"Results file: {self.results_file}")

        # TODO: Implement evaluation logic
        return self.results_file
