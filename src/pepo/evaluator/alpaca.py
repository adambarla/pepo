import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from alpaca_eval.constants import EVALUATORS_CONFIG_DIR
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from alpaca_eval import evaluate as alpaca_evaluate  # type: ignore[attr-defined]

from ..generate import Generator
from ..utils import sanitize_filename
from .base import BaseEvaluator


class AlpacaEvalEvaluator(BaseEvaluator):
    """AlpacaEval evaluator implementation."""

    def __init__(
        self,
        model: Any,
        dataset_id: str,
        dataset_name: str,
        dataset_split: str,
        instruction_key: str,
        output_dir: str,
        generator: Generator,
        annotators_config: Union[str, Dict[str, Any], DictConfig] = "alpaca_eval_gpt4",
        num_samples: Optional[int] = None,
        logger: Optional[Any] = None,
    ) -> None:
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
            annotators_config: AlpacaEval annotator configuration name or path.
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
        self.annotators_config = annotators_config
        if logger is not None:
            self.generator.logger = logger
        self.dataset = self._load_dataset()
        self.responses_filename, self.results_filename, self.leaderboard_filename = (
            self._generate_filename()
        )
        self.responses_file = self.output_dir / self.responses_filename
        self.results_file = self.output_dir / self.results_filename
        self.leaderboard_file = self.output_dir / self.leaderboard_filename

    def responses_exist(self, **kwargs: Any) -> bool:
        """
        Check if responses file already exists.

        Args:
            **kwargs: Additional generation parameters
                (ignored, generator is from config).

        Returns:
            True if responses file exists, False otherwise.
        """
        return (self.output_dir / self.responses_filename).exists()

    def _generate_filename(self, **kwargs: Any) -> tuple[str, str, str]:
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
            Tuple of (responses_filename, results_filename, leaderboard_filename).
        """
        model_name = self.model._get_model_name()
        min_epochs = self.model.get_min_epochs()
        if min_epochs is not None:
            model_name += f"-e{min_epochs}"

        model_name = sanitize_filename(model_name)

        parts = [self.dataset_name, model_name]

        parts.append(f"ns{self.num_samples}")
        parts.append(self.generator.get_name())

        base_name = "_".join(parts)
        responses_filename = f"{base_name}_responses.json"
        results_filename = f"{base_name}_results.json"
        leaderboard_filename = f"{base_name}_leaderboard.csv"

        return responses_filename, results_filename, leaderboard_filename

    def _load_dataset(self) -> Any:
        """
        Load dataset from HuggingFace.
        """
        dataset = load_dataset(
            self.dataset_id,
            split=self.dataset_split,
            trust_remote_code=True,
        )
        if self.num_samples is not None and self.num_samples > 0:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        if self.logger:
            count = self.num_samples if self.num_samples else len(dataset)
            self.logger.info(f"Loaded {count} samples from {self.dataset_id}")
        return dataset

    def generate_responses(self, **kwargs):
        """
        Generate responses for AlpacaEval dataset.

        Args:
            **kwargs: Additional generation parameters
                (ignored, generator is from config).

        Returns:
            Path to the responses file.
        """
        if self.logger:
            self.logger.info(
                f"Generating responses for {len(self.dataset)} instructions"
            )

        instructions = [item[self.instruction_key] for item in self.dataset]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        outputs = self.generator.generate_responses(
            model=self.model,
            prompts=instructions,
            apply_chat_template=True,
        )

        model_name = self.model._get_model_name()
        min_epochs = self.model.get_min_epochs()
        if min_epochs is not None:
            model_name += f"-e{min_epochs}"

        formatted_outputs = []
        for item in outputs:
            formatted_outputs.append(
                {
                    "instruction": item["prompt"],
                    "output": item["output"],
                    "generator": model_name,
                    "dataset": self.dataset_name,
                }
            )

        with open(self.responses_file, "w", encoding="utf-8") as f:
            json.dump(formatted_outputs, f, indent=2, ensure_ascii=False)

        if self.logger:
            self.logger.info(
                f"Saved {len(formatted_outputs)} responses to {self.responses_file}"
            )

        return self.responses_file

    def evaluate(self, responses_file: Optional[Path] = None) -> Path:
        """
        Evaluate responses using AlpacaEval.

        Args:
            responses_file: Path to the responses file.

        Returns:
            Path to the results file.
        """
        if responses_file is None:
            responses_file = self.responses_file
        responses_file = Path(responses_file)

        annotators_config = self.annotators_config

        if not isinstance(annotators_config, (dict, DictConfig)):
            raise ValueError("annotators_config must be a dictionary or DictConfig")

        conf_dict = OmegaConf.to_container(annotators_config, resolve=True)
        if not isinstance(conf_dict, dict):
            raise ValueError("annotators_config must resolve to a dictionary")

        for annotator in conf_dict.values():
            if "prompt_template" in annotator:
                annotator["prompt_template"] = str(
                    list(
                        (EVALUATORS_CONFIG_DIR / annotator["prompt_template"]).glob(
                            "*.txt"
                        )
                    )[0]
                )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(conf_dict, tmp)
            config_arg = tmp.name

        precomputed_leaderboard = (
            str(self.leaderboard_file) if self.leaderboard_file.exists() else None
        )

        if self.logger:
            self.logger.info(f"Evaluating responses from {responses_file}")
            self.logger.info(
                f"Using model-specific leaderboard: {self.leaderboard_file}"
            )

        try:
            df_leaderboard, all_annotations = alpaca_evaluate(
                model_outputs=str(responses_file),
                annotators_config=config_arg,
                output_path=self.output_dir,
                precomputed_leaderboard=precomputed_leaderboard,
                is_return_instead_of_print=True,
            )

            df_leaderboard.to_csv(self.leaderboard_file, index=True)
            if self.logger:
                self.logger.info(
                    f"Saved model-specific leaderboard to {self.leaderboard_file}"
                )

            if all_annotations is not None:
                annotations_df = (
                    pd.DataFrame(all_annotations)
                    if not isinstance(all_annotations, pd.DataFrame)
                    else all_annotations
                )
                annotations_df.to_json(self.results_file, orient="records", indent=2)
                if self.logger:
                    self.logger.info(f"Saved annotations to {self.results_file}")

            if self.logger:
                self.logger.info(
                    f"Evaluation completed. Leaderboard:\n{df_leaderboard}"
                )

            return self.results_file

        except Exception as e:
            if self.logger:
                self.logger.error(f"Evaluation failed: {e}")
            raise e

    @staticmethod
    def consolidate_leaderboards(
        output_dir: Union[str, Path], logger: Optional[Any] = None
    ) -> Path:
        """
        Consolidate all model-specific leaderboard CSV files into a single
        leaderboard.csv.

        Args:
            output_dir: Directory containing the *_leaderboard.csv files.
            logger: Optional logger instance.

        Returns:
            Path to the consolidated leaderboard.csv file.
        """
        output_dir = Path(output_dir)

        leaderboard_files = sorted(output_dir.glob("*_leaderboard.csv"))

        if not leaderboard_files:
            if logger:
                logger.warning(f"No *_leaderboard.csv files found in {output_dir}")
            return output_dir / "leaderboard.csv"

        if logger:
            logger.info(
                f"Found {len(leaderboard_files)} model-specific leaderboard "
                f"files to consolidate"
            )

        all_leaderboards = []
        for lb_file in leaderboard_files:
            try:
                df = pd.read_csv(lb_file, index_col=0)
                all_leaderboards.append(df)
                if logger:
                    logger.debug(f"Loaded {len(df)} entries from {lb_file.name}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load {lb_file.name}: {e}")
                continue

        if not all_leaderboards:
            if logger:
                logger.warning("No valid leaderboard files found to consolidate")
            return output_dir / "leaderboard.csv"

        consolidated_df = pd.concat(all_leaderboards, ignore_index=False)

        if consolidated_df.index.duplicated().any():
            if logger:
                logger.info("Removing duplicate model entries (keeping most recent)")
            consolidated_df = consolidated_df[
                ~consolidated_df.index.duplicated(keep="last")
            ]

        if "length_controlled_winrate" in consolidated_df.columns:
            consolidated_df = consolidated_df.sort_values(
                by="length_controlled_winrate", ascending=False
            )
        elif "win_rate" in consolidated_df.columns:
            consolidated_df = consolidated_df.sort_values(
                by="win_rate", ascending=False
            )
        else:
            consolidated_df = consolidated_df.sort_index()

        consolidated_path = output_dir / "leaderboard.csv"
        consolidated_df.to_csv(consolidated_path, index=True)

        if logger:
            logger.info(
                f"Consolidated {len(consolidated_df)} entries from "
                f"{len(all_leaderboards)} files into {consolidated_path}"
            )

        return consolidated_path
