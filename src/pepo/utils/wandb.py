import dataclasses
import logging
from typing import Any, Dict, List, Optional

import wandb
from omegaconf import OmegaConf
from wandb.util import generate_id

logger = logging.getLogger(__name__)


class WandbRun:
    """
    Handler for a single Weights & Biases run.
    Created by WandbManager for each training or evaluation run.
    """

    def __init__(
        self,
        manager: "WandbManager",
        model: Any,
        data_manager: Any,
        model_idx: Optional[int] = None,
        group: Optional[str] = None,
        run_id: Optional[str] = None,
        generator: Optional[Any] = None,
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
            generator: Optional generator instance for evaluation run naming.
        """
        self.manager = manager
        self.enabled = manager.enabled
        self.wandb = manager.wandb
        self.initialized = False
        self.run: Optional[Any] = None
        self.model = model
        self.data_manager = data_manager
        self.model_idx = model_idx
        self.generator = generator
        self.group = group

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

    def _generate_eval_run_name(self) -> Optional[str]:
        """Generate run name for evaluation runs: {model_name}-{generator_name}-eval."""
        parts = []
        if self.model and hasattr(self.model, "get_name"):
            parts.append(self.model.get_name())
        if self.generator and hasattr(self.generator, "get_name"):
            parts.append(self.generator.get_name())
        parts.append("eval")
        return "-".join(parts) if len(parts) > 1 else None

    def init_train_run(self) -> None:
        """Initialize a training run. Sets job_type='train' automatically."""
        if not self.enabled:
            return
        self._init_run(job_type="train")

    def init_eval_run(self) -> None:
        """Initialize an evaluation run (separate from training run but same group)."""
        if not self.enabled:
            return
        if self.run_id is None:
            raise ValueError("Cannot initialize evaluation run: run_id is None")
        self._init_run(job_type="evaluation", resume="allow")

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

        # Use appropriate name generation based on job type
        if job_type == "evaluation":
            run_name = self._generate_eval_run_name()
        else:
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

        if self.group is not None:
            init_kwargs["group"] = self.group
        elif self.manager.group is not None:
            init_kwargs["group"] = self.manager.group

        if self.manager.cfg is not None:
            cfg_obj = self.manager.cfg
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
            logger.warning(f"Error finding training run for {expected_name}: {e}")

        return None

    def find_evaluation_run_id(
        self, group: str, model_name: str, generator_name: str
    ) -> Optional[str]:
        """
        Find an existing evaluation run for the given group.

        Searches for runs with:
        - Same group
        - job_type="evaluation"
        - Name matching {model_name}-{generator_name}-eval

        Args:
            group: Group name to search within.
            model_name: Model name (base name without epoch suffix).
            generator_name: Generator name.

        Returns:
            Run ID if found, None otherwise.
        """
        if not self.enabled or self.wandb is None:
            return None

        expected_name = f"{model_name}-{generator_name}-eval"

        try:
            api = self.wandb.Api()
            entity = self.entity or "wandb"
            project_path = f"{entity}/{self.project}"

            # Search for evaluation runs in the same group
            runs = api.runs(
                project_path,
                filters={
                    "group": group,
                    "jobType": "evaluation",
                },
                order="-created_at",
                per_page=50,
            )

            for run in runs:
                run_name = getattr(run, "name", None) or getattr(
                    run, "display_name", None
                )
                if run_name == expected_name:
                    logger.debug(
                        f"Found existing evaluation run {expected_name}: {run.id}"
                    )
                    return str(run.id)

            # Fallback: search without group filter in case group wasn't indexed
            runs = api.runs(
                project_path,
                filters={"display_name": expected_name, "jobType": "evaluation"},
                order="-created_at",
                per_page=10,
            )
            for run in runs:
                run_group = getattr(run, "group", None)
                if run_group == group:
                    logger.debug(
                        f"Found evaluation run {expected_name} via fallback: {run.id}"
                    )
                    return str(run.id)

        except Exception as e:
            logger.warning(f"Error finding evaluation run for {expected_name}: {e}")

        return None

    def get_evaluation_handler(
        self,
        model: Any,
        generator: Any,
    ) -> Optional["WandbRun"]:
        """
        Get a wandb handler for evaluation that shares group with training run.

        If an evaluation run already exists for this group, it will be resumed.
        Otherwise, a new evaluation run will be created.

        Args:
            model: Model instance for evaluation. Must have get_name() method.
            generator: Generator instance. Must have get_name() method.

        Returns:
            WandbRun instance for the evaluation run, or None if wandb disabled.
        """
        if not self.enabled:
            return None

        if self.wandb is None:
            raise ValueError("Wandb is not installed")

        if not hasattr(model, "get_name"):
            raise ValueError(
                "Cannot find training run: model does not have get_name() method"
            )

        if not hasattr(generator, "get_name"):
            raise ValueError(
                "Cannot create evaluation run: generator does not have get_name() "
                "method"
            )

        model_name = model.get_name()
        generator_name = generator.get_name()

        # Find the training run's group
        group = self.find_training_run_group(model_name, model_idx=0)

        if group is None:
            logger.warning(
                f"Cannot find training run group for model {model_name}. "
                f"Creating evaluation run without group."
            )
            # Still create the evaluation run, just without a group
            group = model_name  # Use model name as fallback group

        self.group = group

        # Look for existing evaluation run in this group
        existing_run_id = self.find_evaluation_run_id(
            group=group,
            model_name=model_name,
            generator_name=generator_name,
        )

        handler = WandbRun(
            manager=self,
            model=model,
            data_manager=None,
            model_idx=0,
            group=group,
            run_id=existing_run_id,  # None if creating new, ID if resuming
            generator=generator,
        )
        return handler
