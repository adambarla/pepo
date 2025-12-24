import dataclasses
import logging
from typing import Any, Dict, List, Optional

import wandb
from omegaconf import OmegaConf
from wandb.util import generate_id

from ..base_model import BaseModel
from ..generator import Generator

logger = logging.getLogger(__name__)


class WandbRun:
    """
    Handler for a single Weights & Biases run.
    Created by WandbManager for each training or evaluation run.
    """

    def __init__(
        self,
        enabled: bool,
        wandb_module: Any,
        project: Optional[str],
        tags: List[str],
        notes: Optional[str],
        entity: Optional[str],
        mode: str,
        cfg: Optional[Dict[str, Any]],
        run_name: str,
        job_type: str,
        group: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize wandb handler for a single run.

        Args:
            enabled: Whether wandb logging is enabled.
            wandb_module: The wandb module.
            project: Wandb project name.
            tags: List of tags for runs.
            notes: Notes/description for runs.
            entity: Wandb entity/team name.
            mode: Wandb mode.
            cfg: Configuration dictionary to log.
            run_name: Name for this run.
            job_type: Type of job ('train' or 'evaluation').
            group: Group name for organizing related runs.
            run_id: Optional run ID to resume an existing run.
        """
        self.enabled = enabled
        self.wandb = wandb_module
        self.project = project
        self.tags = tags.copy()
        self.notes = notes
        self.entity = entity
        self.mode = mode
        self.cfg = cfg
        self.run_name = run_name
        self.job_type = job_type
        self.group = group

        self.initialized = False
        self.run: Optional[Any] = None
        self.run_id = run_id or generate_id()

    def init_run(self) -> None:
        """Initialize the wandb run."""
        if not self.enabled:
            return

        if self.initialized and self.run is not None:
            raise ValueError("Wandb run is already initialized. Call finish() first.")

        if self.wandb is None:
            raise ValueError("Wandb is not installed")

        if self.run_id is None:
            raise ValueError("Wandb run_id is None")

        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "name": self.run_name,
            "tags": self.tags,
            "notes": self.notes,
            "entity": self.entity,
            "mode": self.mode,
            "id": self.run_id,
            "job_type": self.job_type,
        }

        if self.group is not None:
            init_kwargs["group"] = self.group

        if self.cfg is not None:
            cfg_obj = self.cfg
            if OmegaConf.is_config(cfg_obj):
                init_kwargs["config"] = OmegaConf.to_container(cfg_obj, resolve=True)
            elif dataclasses.is_dataclass(cfg_obj):
                init_kwargs["config"] = dataclasses.asdict(cfg_obj)
            else:
                init_kwargs["config"] = cfg_obj

        init_kwargs["settings"] = self.wandb.Settings(console="wrap")

        self.run = self.wandb.init(**init_kwargs)
        self.initialized = True

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if self.enabled and self.initialized and self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled and self.initialized and self.run is not None:
            self.run.finish()
            self.initialized = False
            self.run = None


class WandbManager:
    """
    Manager for Weights & Biases logging.
    Singleton-like instance that manages wandb configuration and creates handlers.
    """

    def __init__(
        self,
        enabled: bool = False,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "online",
        cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize wandb manager.

        Args:
            enabled: Whether wandb logging is enabled.
            project: Wandb project name.
            tags: List of tags for runs.
            notes: Notes/description for runs.
            entity: Wandb entity/team name.
            mode: Wandb mode ("online", "offline", "disabled").
            cfg: Configuration dictionary to log to wandb.
        """
        self.enabled = enabled
        self.wandb = wandb
        self.project = project
        self.tags = tags or []
        self.notes = notes
        self.entity = entity
        self.mode = mode
        self.cfg = cfg
        self.group: Optional[str] = None

    @staticmethod
    def _generate_train_run_name(
        model: BaseModel,
        data_manager: Any,
        model_idx: int,
    ) -> str:
        """Generate run name for training runs."""
        parts = []
        if model and hasattr(model, "get_name"):
            parts.append(model.get_name())
        if data_manager and hasattr(data_manager, "get_run_identifier"):
            parts.append(data_manager.get_run_identifier())
        if parts:
            name = "-".join(parts)
            if f"-l{model_idx}" not in name:
                name = f"{name}-l{model_idx}"
            return name
        return f"train-l{model_idx}"

    @staticmethod
    def _generate_eval_run_name(
        model_name: str,
        generator_name: str,
        epoch: Optional[int],
    ) -> str:
        """Generate run name for evaluation runs."""
        parts = [model_name, generator_name]
        if epoch is not None:
            parts.append(f"e{epoch}")
        parts.append("eval")
        return "-".join(parts)

    def _get_base_tags(self, model: BaseModel) -> List[str]:
        """Get tags including base model name if available."""
        tags = self.tags.copy()
        if model and hasattr(model, "_get_base_model_name"):
            tags.append(model._get_base_model_name())
        return tags

    def get_training_wandb_handler(
        self,
        model: BaseModel,
        data_manager: Any,
        model_idx: int,
        group: Optional[str] = None,
    ) -> Optional["WandbRun"]:
        """
        Create a new wandb handler for a training run.

        Args:
            model: Model instance for this training run.
            data_manager: Data manager instance for this training run.
            model_idx: Model index for this training run.
            group: Optional group name for organizing related runs.

        Returns:
            WandbRun instance for the training run, or None if wandb is disabled.
        """
        if not self.enabled:
            return None

        if group is not None:
            self.group = group

        run_name = self._generate_train_run_name(model, data_manager, model_idx)

        handler = WandbRun(
            enabled=self.enabled,
            wandb_module=self.wandb,
            project=self.project,
            tags=self._get_base_tags(model),
            notes=self.notes,
            entity=self.entity,
            mode=self.mode,
            cfg=self.cfg,
            run_name=run_name,
            job_type="train",
            group=group or self.group,
        )
        return handler

    def find_training_run_group(
        self, model_name: str, model_idx: int = 0
    ) -> Optional[str]:
        """
        Find the group of the most recent training run for a given model name.

        Args:
            model_name: Model name to search for (base name without epoch suffix).
            model_idx: Model index (default 0 for l0).

        Returns:
            Group name of the most recent training run, or None if not found.
        """
        if not self.enabled or self.wandb is None:
            return None

        expected_name = f"{model_name}-l{model_idx}"

        try:
            api = self.wandb.Api()
            entity = self.entity or "wandb"
            project_path = f"{entity}/{self.project}"

            runs = api.runs(project_path, order="-created_at", per_page=100)
            for run in runs:
                run_name = getattr(run, "name", None) or getattr(
                    run, "display_name", None
                )
                run_job_type = getattr(run, "job_type", None)

                if run_name == expected_name:
                    if run_job_type == "train":
                        group = getattr(run, "group", None)
                        logger.debug(
                            f"Found training run {expected_name}: {run.id} "
                            f"(job_type=train, group={group})"
                        )
                        return group

            # Fallback: search with filter
            runs = api.runs(
                project_path,
                filters={"display_name": expected_name, "jobType": "train"},
                order="-created_at",
                per_page=10,
            )
            if runs:
                group = getattr(runs[0], "group", None)
                logger.debug(
                    f"Found run {expected_name} via filter: {runs[0].id}, group={group}"
                )
                return group
        except Exception as e:
            logger.error(f"Failed to find training run for {expected_name}: {e}")

        return None

    def get_evaluation_handler(
        self,
        model: BaseModel,
        generator: Generator,
        epoch: Optional[int] = None,
    ) -> Optional["WandbRun"]:
        """
        Get a wandb handler for evaluation that shares group with training run.

        Each epoch gets its own separate evaluation run to allow parallel execution.

        Args:
            model: Model instance for evaluation. Must have get_name() method.
            generator: Generator instance. Must have get_name() method.
            epoch: Optional epoch number for unique run naming.

        Returns:
            WandbRun instance for the evaluation run, or None if wandb disabled.
        """
        if not self.enabled:
            return None

        if self.wandb is None:
            raise ValueError("Wandb is not installed")

        model_name = model.get_name()
        generator_name = generator.get_name()
        group = self.find_training_run_group(model_name, model_idx=0)
        run_name = self._generate_eval_run_name(model_name, generator_name, epoch)

        handler = WandbRun(
            enabled=self.enabled,
            wandb_module=self.wandb,
            project=self.project,
            tags=self._get_base_tags(model),
            notes=self.notes,
            entity=self.entity,
            mode=self.mode,
            cfg=self.cfg,
            run_name=run_name,
            job_type="evaluation",
            group=group,
        )
        return handler
