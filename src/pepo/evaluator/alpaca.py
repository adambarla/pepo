import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from alpaca_eval.constants import EVALUATORS_CONFIG_DIR
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from alpaca_eval import evaluate as alpaca_evaluate
from alpaca_eval import metrics

from ..utils import WandbRun, sanitize_filename
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


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
        annotators_config: Union[str, Dict[str, Any], DictConfig] = "alpaca_eval_gpt4",
        num_samples: Optional[int] = None,
        wandb_run: Optional[WandbRun] = None,
    ) -> None:
        """
        Initialize AlpacaEval evaluator.

        Args:
            model: PEPOModel instance (must have generator set).
            dataset_id: HuggingFace dataset ID.
            dataset_name: Dataset configuration name (e.g., "alpaca_eval").
            dataset_split: Dataset split to use.
            instruction_key: Key to extract instruction from dataset items.
            output_dir: Directory to save outputs.
            annotators_config: AlpacaEval annotator configuration name or path.
            num_samples: Optional number of samples to use (None = full dataset).
            wandb_run: Optional wandb run handler for logging.
        """
        super().__init__(
            model=model,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            instruction_key=instruction_key,
            output_dir=output_dir,
        )
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.annotators_config = annotators_config
        self.wandb_run = wandb_run
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

        count = self.num_samples if self.num_samples else len(dataset)
        logger.info(f"Loaded {count} samples from {self.dataset_id}")
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
        logger.info(f"Generating responses for {len(self.dataset)} instructions")

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

        logger.info(
            f"Saved {len(formatted_outputs)} responses to {self.responses_file}"
        )

        return self.responses_file

    def _prepare_annotator_config(self) -> str:
        """Prepare annotator config file and return its path."""
        conf_dict = OmegaConf.to_container(self.annotators_config, resolve=True)

        if not isinstance(conf_dict, dict):
            raise ValueError("annotators_config must be a dictionary")

        for annotator in conf_dict.values():
            if "prompt_template" in annotator:
                template_path = list(
                    (EVALUATORS_CONFIG_DIR / annotator["prompt_template"]).glob("*.txt")
                )[0]
                annotator["prompt_template"] = str(template_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(conf_dict, tmp)
            return tmp.name

    def _load_existing_results(
        self,
    ) -> tuple[Optional[pd.DataFrame], Optional[list[Any]]]:
        """Load existing annotations and compute leaderboard."""
        if not self.results_file.exists():
            return None, None

        logger.info(f"Loading existing annotations from {self.results_file}")

        annotations_df = pd.read_json(self.results_file)
        all_annotations = annotations_df.to_dict(orient="records")

        model_name = self.model._get_model_name()
        min_epochs = self.model.get_min_epochs()
        if min_epochs is not None:
            model_name += f"-e{min_epochs}"
        model_name = sanitize_filename(model_name)

        fn_metric = getattr(metrics, "get_length_controlled_winrate", None) or getattr(
            metrics, "get_winrate"
        )
        metrics_dict = fn_metric(all_annotations)

        avg_length = 0
        if "output_2" in annotations_df.columns:
            avg_length = int(annotations_df["output_2"].str.len().mean())

        leaderboard_dict = {
            model_name: {
                **metrics_dict,
                "mode": "community",
                "avg_length": avg_length,
            }
        }

        df_leaderboard = pd.DataFrame.from_dict(leaderboard_dict, orient="index")

        if self.leaderboard_file.exists():
            existing_leaderboard = pd.read_csv(self.leaderboard_file, index_col=0)
            df_leaderboard = pd.concat([existing_leaderboard, df_leaderboard])
            df_leaderboard = df_leaderboard[
                ~df_leaderboard.index.duplicated(keep="last")
            ]

        sort_by = (
            "length_controlled_winrate"
            if "length_controlled_winrate" in df_leaderboard.columns
            else "win_rate"
        )
        df_leaderboard = df_leaderboard.sort_values(by=sort_by, ascending=False)

        return df_leaderboard, all_annotations

    def _run_evaluation(
        self, responses_file: Path, config_path: str
    ) -> tuple[pd.DataFrame, list[Any]]:
        """Run full evaluation with annotation."""
        logger.info(f"Evaluating responses from {responses_file}")

        precomputed_leaderboard = (
            str(self.leaderboard_file) if self.leaderboard_file.exists() else None
        )

        df_leaderboard, all_annotations = alpaca_evaluate(
            model_outputs=str(responses_file),
            annotators_config=config_path,
            output_path=self.output_dir,
            precomputed_leaderboard=precomputed_leaderboard,
            is_return_instead_of_print=True,
        )
        return df_leaderboard, all_annotations

    def _save_results(
        self,
        df_leaderboard: pd.DataFrame,
        all_annotations: Optional[list[Any]],
    ) -> None:
        """Save leaderboard and annotations to files."""
        df_leaderboard.to_csv(self.leaderboard_file, index=True)

        if all_annotations is not None:
            annotations_df = (
                pd.DataFrame(all_annotations)
                if not isinstance(all_annotations, pd.DataFrame)
                else all_annotations
            )
            annotations_df.to_json(self.results_file, orient="records", indent=2)

        logger.info(f"Saved leaderboard to {self.leaderboard_file}")
        if all_annotations is not None:
            logger.info(f"Saved annotations to {self.results_file}")

    def evaluate(self, responses_file: Optional[Path] = None) -> Path:
        """
        Evaluate responses using AlpacaEval.

        Args:
            responses_file: Path to the responses file.

        Returns:
            Path to the results file.
        """
        responses_file = Path(responses_file) if responses_file else self.responses_file

        if not responses_file.exists():
            raise FileNotFoundError(
                f"Responses file {responses_file} does not exist. "
                "Please generate responses first."
            )

        config_path = self._prepare_annotator_config()

        df_leaderboard, all_annotations = self._load_existing_results()

        if df_leaderboard is None:
            df_leaderboard, all_annotations = self._run_evaluation(
                responses_file, config_path
            )

        self._save_results(df_leaderboard, all_annotations)
        self.consolidate_leaderboards(self.output_dir)

        if self.wandb_run is not None and self.wandb_run.enabled:
            min_epochs = self.model.get_min_epochs()
            model_name = self.model._get_model_name()
            if min_epochs is not None:
                model_name += f"-e{min_epochs}"

            generator_config = self.generator.get_name()
            metric_prefix = f"eval/{self.dataset_name}/{generator_config}"

            if model_name not in df_leaderboard.index:
                return self.results_file

            model_metrics = df_leaderboard.loc[model_name]
            metrics_to_log = {}

            if "length_controlled_winrate" in model_metrics:
                metrics_to_log[f"{metric_prefix}/length_controlled_winrate"] = (
                    model_metrics["length_controlled_winrate"]
                )
            if "win_rate" in model_metrics:
                metrics_to_log[f"{metric_prefix}/win_rate"] = model_metrics["win_rate"]
            if "avg_length" in model_metrics:
                metrics_to_log[f"{metric_prefix}/avg_length"] = model_metrics[
                    "avg_length"
                ]
            if min_epochs is not None:
                metrics_to_log[f"{metric_prefix}/epoch"] = min_epochs

            if metrics_to_log:
                self.wandb_run.log(metrics_to_log)

        return self.results_file

    @staticmethod
    def consolidate_leaderboards(output_dir: Union[str, Path]) -> Path:
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
            logger.warning(f"No *_leaderboard.csv files found in {output_dir}")
            return output_dir / "leaderboard.csv"

        logger.info(
            f"Found {len(leaderboard_files)} model-specific leaderboard "
            f"files to consolidate"
        )

        all_leaderboards = []
        for lb_file in leaderboard_files:
            try:
                df = pd.read_csv(lb_file, index_col=0)
                all_leaderboards.append(df)
                logger.debug(f"Loaded {len(df)} entries from {lb_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {lb_file.name}: {e}")
                continue

        if not all_leaderboards:
            logger.warning("No valid leaderboard files found to consolidate")
            return output_dir / "leaderboard.csv"

        consolidated_df = pd.concat(all_leaderboards, ignore_index=False)

        if consolidated_df.index.duplicated().any():
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

        logger.info(
            f"Consolidated {len(consolidated_df)} entries from "
            f"{len(all_leaderboards)} files into {consolidated_path}"
        )

        return consolidated_path
