import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from alpaca_eval import evaluate as alpaca_evaluate
from alpaca_eval.constants import EVALUATORS_CONFIG_DIR
from omegaconf import DictConfig, OmegaConf

from ..base_model import BaseModel
from ..utils import WandbRun
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class AlpacaEvalEvaluator(BaseEvaluator):
    """AlpacaEval evaluator implementation."""

    def __init__(
        self,
        dataset_id: str,
        dataset_split: str,
        output_dir: str,
        annotators_config: Union[str, Dict[str, Any], DictConfig] = "alpaca_eval_gpt4",
        num_samples: Optional[int] = None,
        wandb_run: Optional[WandbRun] = None,
    ) -> None:
        """
        Initialize AlpacaEval evaluator.

        Args:
            dataset_id: HuggingFace dataset ID.
            dataset_split: Dataset split to use.
            output_dir: Directory to save outputs.
            annotators_config: AlpacaEval annotator configuration name or path.
            num_samples: Optional number of samples to use (None = full dataset).
            wandb_run: Optional wandb run handler for logging.
        """
        super().__init__(
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            output_dir=output_dir + "/alpaca_eval/",
            num_samples=num_samples,
        )
        self.num_samples = num_samples
        self.annotators_config = annotators_config
        self.wandb_run = wandb_run
        self.dataset = self._load_dataset()

    def evaluate(
        self,
        model: BaseModel,
        epoch: Optional[int] = None,
        ref_model: Optional[BaseModel] = None,
        ref_epoch: Optional[int] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Evaluate responses using AlpacaEval.

        Args:
            model: BaseModel instance (required).
            epoch: Epoch of the model (optional). Used to load a specific checkpoint.
            reference_model: BaseModel instance (optional).
            reference_epoch: Epoch of the reference model (optional).
            overwrite: Whether to overwrite existing outputs.
            **kwargs: Additional args.
        """
        self.check_generator_consistency(model, ref_model)

        model_responses_file = self._get_or_generate(
            model, epoch=epoch, overwrite=overwrite
        )
        logger.info(f"Model responses file: {model_responses_file}")
        ref_responses_file = None
        if ref_model is not None:
            ref_responses_file = self._get_or_generate(
                ref_model, epoch=ref_epoch, overwrite=overwrite
            )

            logger.info(f"Reference responses file: {ref_responses_file}")

        folder = self._get_folder(ref_model, ref_epoch)
        annotations_folder = folder / "annotations"
        annotations_folder.mkdir(parents=True, exist_ok=True)
        leaderboards_folder = folder / "leaderboards"
        leaderboards_folder.mkdir(parents=True, exist_ok=True)
        filename = self._get_filename(model, epoch)
        annotations_file = annotations_folder / f"{filename}_annotations.json"
        leaderboard_file = leaderboards_folder / f"{filename}_leaderboard.csv"

        config_path = self._prepare_annotator_config()

        df_leaderboard, all_annotations = self._run_alpaca_eval(
            model_responses_file=model_responses_file,
            ref_responses_file=ref_responses_file,
            config_path=config_path,
            leaderboard_file=leaderboard_file,
            name=filename,
            is_overwrite_leaderboard=overwrite,
        )

        self._save_and_log(
            df_leaderboard=df_leaderboard,
            all_annotations=all_annotations,
            annotations_file=annotations_file,
            leaderboard_file=leaderboard_file,
            model=model,
            epoch=epoch,
            consolidation_folder=folder,
            model_name=filename,
        )

        return annotations_file

    def _run_alpaca_eval(
        self,
        model_responses_file: Path,
        ref_responses_file: Optional[Path],
        config_path: str,
        leaderboard_file: Path,
        name: str,
        is_overwrite_leaderboard: bool,
    ) -> tuple[pd.DataFrame, list[Any]]:
        """Run Alpaca Eval evaluation."""
        logger.info(f"Evaluating responses from {model_responses_file}")

        precomputed_leaderboard = (
            str(leaderboard_file) if leaderboard_file.exists() else None
        )

        if ref_responses_file:
            df_leaderboard, all_annotations = alpaca_evaluate(
                model_outputs=str(model_responses_file),
                reference_outputs=str(ref_responses_file),
                annotators_config=config_path,
                name=name,
                precomputed_leaderboard=precomputed_leaderboard,
                is_overwrite_leaderboard=is_overwrite_leaderboard,
                is_return_instead_of_print=True,
                output_path=None,
            )
        else:
            # we don't want to override the default reference_outputs by None
            df_leaderboard, all_annotations = alpaca_evaluate(
                model_outputs=str(model_responses_file),
                annotators_config=config_path,
                name=name,
                precomputed_leaderboard=precomputed_leaderboard,
                is_overwrite_leaderboard=is_overwrite_leaderboard,
                is_return_instead_of_print=True,
                output_path=None,
            )

        return df_leaderboard, all_annotations

    def _save_and_log(
        self,
        df_leaderboard: pd.DataFrame,
        all_annotations: list[Any],
        annotations_file: Path,
        leaderboard_file: Path,
        model: BaseModel,
        epoch: Optional[int],
        consolidation_folder: Path,
        model_name: str,
    ) -> None:
        """Save results to files and log to WandB."""
        df_leaderboard.to_csv(leaderboard_file, index=True)

        if all_annotations is not None:
            annotations_df = (
                pd.DataFrame(all_annotations)
                if not isinstance(all_annotations, pd.DataFrame)
                else all_annotations
            )
            annotations_df.to_json(annotations_file, orient="records", indent=2)

        logger.info(f"Saved leaderboard to {leaderboard_file}")

        self.consolidate_leaderboards(consolidation_folder)

        # Logging if model info provided
        if self.wandb_run is not None and self.wandb_run.enabled:
            generator_config = "unknown"
            if hasattr(model, "generator") and model.generator:
                generator_config = model.generator.get_name()

            metric_prefix = f"eval/{self.dataset_id}/{generator_config}"

            target_idx = None
            if model_name in df_leaderboard.index:
                target_idx = model_name

            if target_idx:
                model_metrics = df_leaderboard.loc[target_idx]
                metrics_to_log = {}

                metric_names = [
                    "length_controlled_winrate",
                    "lc_standard_error",
                    "win_rate",
                    "standard_error",
                    "avg_length",
                    "n_wins",
                    "n_wins_base",
                    "n_draws",
                    "n_total",
                    "discrete_win_rate",
                ]
                for metric_name in metric_names:
                    if metric_name not in model_metrics:
                        logger.warning(
                            f"Metric '{metric_name}' not found in leaderboard"
                        )
                        continue
                    metrics_to_log[f"{metric_prefix}/{metric_name}"] = model_metrics[
                        metric_name
                    ]

                if epoch is not None:
                    metrics_to_log[f"{metric_prefix}/epoch"] = epoch

                if metrics_to_log:
                    self.wandb_run.log(metrics_to_log)

    def _get_or_generate(
        self,
        model: BaseModel,
        epoch: Optional[int] = None,
        overwrite: bool = False,
    ) -> Path:
        responses_folder = self.output_dir / "responses"
        responses_folder.mkdir(parents=True, exist_ok=True)
        filename = self._get_filename(model, epoch)
        path = responses_folder / f"{filename}_responses.json"
        if overwrite or not self._responses_exist(path):
            self._generate_responses(path, model, epoch=epoch)
        return path

    def _responses_exist(
        self,
        save_path: Path,
    ) -> bool:
        if save_path.exists():
            return True
        logger.info(f"File {save_path} does not exist.")
        return False

    def _generate_responses(
        self,
        save_path: Path,
        model: BaseModel,
        epoch: Optional[int] = None,
    ) -> None:
        logger.info(
            f"Generating responses for {self.dataset_id} with {model.get_name()}"
        )

        instructions = [item["instruction"] for item in self.dataset]
        save_path.parent.mkdir(exist_ok=True, parents=True)

        model.load(epoch=epoch)
        outputs = model.generate_responses(prompts=instructions)
        model.unload()

        model_name = model.get_name(epoch=epoch)
        formatted_outputs = []
        for item in outputs:
            formatted_outputs.append(
                {
                    "instruction": item["prompt"],
                    "output": item["output"],
                    "generator": model_name,
                    "dataset": f"{self.dataset_id}/{self.dataset_split}",
                }
            )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(formatted_outputs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(formatted_outputs)} responses to {save_path}")

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

        leaderboard_files = sorted(output_dir.glob("**/*_leaderboard.csv"))

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
