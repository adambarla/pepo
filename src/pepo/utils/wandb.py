import dataclasses
import logging
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
from wandb.util import generate_id

import wandb

logger = logging.getLogger(__name__)


class WandbRun:
    """
    Handler for a single Weights & Biases run.
    Created by WandbManager for each training or benchmark run.
    """

    def __init__(
        self,
        manager: "WandbManager",
        model: Any,
        data_manager: Any,
        model_idx: Optional[int] = None,
        group: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize wandb handler for a single run.

        Args:
            manager: WandbManager instance that created this handler.
            model: Model instance for run name generation.
            data_manager: Data manager instance for run name generation.
            model_idx: Model index for this run.
            group: Group name for organizing related runs.
            run_id: Optional run ID to resume an existing run.
        """
        self.manager = manager
        self.enabled = manager.enabled
        self.wandb = manager.wandb
        self.initialized = False
        self.run: Optional[Any] = None
        self.model = model
        self.data_manager = data_manager
        self.model_idx = model_idx

        self.run_id = run_id or generate_id()

    def _generate_run_name(self) -> Optional[str]:
        """Generate run name from model and data manager identifiers."""
        parts = []
        if self.model:
            if hasattr(self.model, "get_run_identifier"):
                parts.append(self.model.get_run_identifier())
            elif hasattr(self.model, "get_name"):
                parts.append(self.model.get_name())
            elif hasattr(self.model, "get_submodel_name"):
                model_idx = (
                    self.model_idx
                    if self.model_idx is not None
                    else getattr(self.model, "_model_idx", 0)
                )
                parts.append(self.model.get_submodel_name(model_idx))
        if self.data_manager and hasattr(self.data_manager, "get_run_identifier"):
            parts.append(self.data_manager.get_run_identifier())
        if parts:
            name = "-".join(parts)
            if self.model_idx is not None and f"-l{self.model_idx}" not in name:
                name = f"{name}-l{self.model_idx}"
            return name
        return None

    def init_train_run(self) -> None:
        """Initialize a training run. Sets job_type='train' automatically."""
        if not self.enabled:
            return
        self._init_run(job_type="train")

    def init_bench_run(self) -> None:
        """Initialize a benchmark run that continues an existing training run."""
        if not self.enabled:
            return
        if self.run_id is None:
            raise ValueError("Cannot initialize benchmark run: run_id is None")
        self._init_run(job_type="benchmark", resume="must")

    def _init_run(
        self,
        job_type: str,
        resume: Optional[str] = None,
    ) -> None:
        """Internal method to initialize wandb run with specified job_type."""
        if not self.enabled:
            return

        if self.initialized and self.run is not None:
            raise ValueError("Wandb run is already initialized. Call finish() first.")

        if self.wandb is None:
            raise ValueError("Wandb is not installed")

        if self.run_id is None:
            raise ValueError("Wandb run_id is None")

        run_name = self._generate_run_name()
        if run_name is None:
            run_name = f"run-{self.run_id}"

        tags = self.manager.tags.copy()
        if self.model and hasattr(self.model, "_get_base_model_name"):
            tags.append(self.model._get_base_model_name())

        init_kwargs: Dict[str, Any] = {
            "project": self.manager.project,
            "name": run_name,
            "tags": tags,
            "notes": self.manager.notes,
            "entity": self.manager.entity,
            "mode": self.manager.mode,
            "id": self.run_id,
            "job_type": job_type,
        }

        if resume is not None:
            init_kwargs["resume"] = resume

        if self.manager.group is not None:
            init_kwargs["group"] = self.manager.group

        if self.manager.cfg is not None:
            cfg_obj = self.manager.cfg
            if OmegaConf.is_config(cfg_obj):
                init_kwargs["config"] = OmegaConf.to_container(cfg_obj, resolve=True)
            elif dataclasses.is_dataclass(cfg_obj):
                init_kwargs["config"] = dataclasses.asdict(cfg_obj)
            else:
                init_kwargs["config"] = cfg_obj

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

    def get_training_wandb_handler(
        self,
        model: Any,
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

        handler = WandbRun(
            manager=self,
            model=model,
            data_manager=data_manager,
            model_idx=model_idx,
            group=group,
        )
        return handler

    def find_training_run_id(
        self, model_name: str, model_idx: int = 0
    ) -> Optional[str]:
        """
        Find the most recent training run_id for a given model name with specified idx.

        Args:
            model_name: Model name to search for (base name without epoch suffix).
            model_idx: Model index (default 0 for l0).

        Returns:
            Run ID of the most recent training run matching
            {model_name}-l{model_idx}, or None if not found.
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
                        logger.debug(
                            f"Found training run {expected_name}: {run.id} "
                            "(job_type=train)"
                        )
                        return str(run.id)
                    elif (
                        run_job_type is None
                        or run_job_type == ""
                        or run_job_type == "benchmark"
                    ):
                        logger.debug(
                            f"Found run {expected_name}: {run.id} "
                            f"(job_type={run_job_type}, "
                            f"may be resumed training run)"
                        )
                        return str(run.id)

            runs = api.runs(
                project_path,
                filters={"display_name": expected_name},
                order="-created_at",
                per_page=10,
            )
            if runs:
                logger.debug(f"Found run {expected_name} via filter: {runs[0].id}")
                return str(runs[0].id)
        except Exception as e:
            logger.warning(f"Error finding training run for {expected_name}: {e}")

        return None

    def get_benchmark_handler(
        self,
        model: Any,
    ) -> Optional["WandbRun"]:
        """
        Get a wandb handler for benchmarking that continues an existing training run.
        Automatically finds the most recent training run for the model.
        The benchmark run will be in the same group as the training run.

        Args:
            model: Model instance for benchmarking. Must have get_name() method.

        Returns:
            WandbRun instance for the benchmark run, or None if wandb is disabled.

        Raises:
            ValueError: If the training run cannot be found.
        """
        if not self.enabled:
            return None

        if self.wandb is None:
            raise ValueError("Wandb is not installed")

        if not hasattr(model, "get_name"):
            raise ValueError(
                "Cannot find training run: model does not have get_name() method"
            )

        model_name = model.get_name()
        training_run_id = self.find_training_run_id(model_name, model_idx=0)

        if training_run_id is None:
            raise ValueError(
                f"Cannot find training run for model {model_name}. "
                f"Please ensure the training run completed successfully."
            )

        try:
            api = self.wandb.Api()
            run_path = f"{self.entity or 'wandb'}/{self.project}/{training_run_id}"
            run = api.run(run_path)

            if run is None:
                raise ValueError(
                    f"Wandb run {training_run_id} does not exist in project "
                    f"{self.project}"
                )

            group = None
            if hasattr(run, "group") and run.group:
                group = run.group
                self.group = group

            if hasattr(run, "tags") and run.tags:
                existing_tags = set(self.tags)
                new_tags = [tag for tag in run.tags if tag not in existing_tags]
                self.tags.extend(new_tags)
            if hasattr(run, "notes") and run.notes and not self.notes:
                self.notes = run.notes

        except Exception as e:
            error_str = str(e).lower()
            if (
                "does not exist" in error_str
                or "not found" in error_str
                or "no such" in error_str
            ):
                raise ValueError(
                    f"Cannot resume wandb run {training_run_id}: run does not "
                    f"exist in project {self.project}. "
                    f"Please ensure the training run completed successfully."
                ) from e
            raise

        handler = WandbRun(
            manager=self,
            model=model,
            data_manager=None,
            model_idx=0,
            group=group,
            run_id=training_run_id,
        )
        return handler
