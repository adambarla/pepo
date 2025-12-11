import dataclasses
import logging
import sys
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log level and logger name
    in console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1m\033[31m",  # Bold Red
    }
    GRAY = "\033[90m"  # Gray for logger names
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        log_message = super().format(record)

        # Color the logger name in gray
        logger_name = record.name
        colored_name = f"{self.GRAY}{logger_name}{self.RESET}"
        log_message = log_message.replace(f"[{logger_name}]", f"[{colored_name}]", 1)

        # Color the levelname
        if hasattr(record, "levelname") and record.levelname in self.COLORS:
            levelname = record.levelname
            color = self.COLORS[levelname]
            colored_levelname = f"{color}{levelname}{self.RESET}"
            log_message = log_message.replace(levelname, colored_levelname, 1)

        return log_message


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    model_name: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration."""
    logger_name = f"pepo.{model_name}" if model_name else "pepo"
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Map string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent propagation to root logger to avoid duplicates
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandbHandler:
    """
    Handler for Weights & Biases logging.
    Manages wandb initialization and provides logging interface.
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
        group: Optional[str] = None,
        lazy_init: bool = False,
    ):
        """
        Initialize wandb handler.

        Args:
            enabled: Whether wandb logging is enabled.
            project: Wandb project name.
            tags: List of tags for the run.
            notes: Notes/description for the run.
            entity: Wandb entity/team name.
            mode: Wandb mode ("online", "offline", "disabled").
            cfg: Configuration dictionary to log to wandb.
            group: Group name for organizing related runs together.
            lazy_init: If True, don't initialize run until init_train_run()
                or init_bench_run() is called.
        """
        self.enabled = enabled
        self.wandb = wandb
        self.initialized = False
        self.run: Optional[Any] = None
        self.project = project
        self.tags = tags or []
        self.notes = notes
        self.entity = entity
        self.mode = mode
        self.cfg = cfg
        self.group = group
        self.lazy_init = lazy_init
        self.logger: Optional[logging.Logger] = None
        self.model: Optional[Any] = None
        self.data_manager: Optional[Any] = None
        self.model_idx: Optional[int] = None

        self.peer_run_ids: List[str] = []
        self.latest_metrics: Dict[str, Any] = {}

        if wandb is not None:
            self.run_id: Optional[str] = wandb.util.generate_id()
        else:
            self.run_id = None

        if enabled and not self.lazy_init:
            raise ValueError(
                "If enabled=True, you must use lazy_init=True and "
                "call init_train_run() or init_bench_run()"
            )

    def set_run_context(
        self, model: Any, data_manager: Any, model_idx: Optional[int] = None
    ) -> None:
        """Set model and data manager for run name generation."""
        self.model = model
        self.data_manager = data_manager
        if model_idx is not None:
            self.model_idx = model_idx

    def _generate_run_name(self) -> Optional[str]:
        """Generate run name from model and data manager identifiers."""
        parts = []
        if self.model:
            if hasattr(self.model, "get_run_identifier"):
                parts.append(self.model.get_run_identifier())
            elif hasattr(self.model, "_get_model_name"):
                parts.append(self.model._get_model_name())
            elif hasattr(self.model, "_get_submodel_name"):
                model_idx = (
                    self.model_idx
                    if self.model_idx is not None
                    else getattr(self.model, "_model_idx", 0)
                )
                parts.append(self.model._get_submodel_name(model_idx))
        if self.data_manager and hasattr(self.data_manager, "get_run_identifier"):
            parts.append(self.data_manager.get_run_identifier())
        if parts:
            name = "-".join(parts)
            if self.model_idx is not None and f"-l{self.model_idx}" not in name:
                name = f"{name}-l{self.model_idx}"
            return name
        return None

    def init_train_run(
        self, run_id: Optional[str] = None, model_idx: Optional[int] = None
    ) -> None:
        """Initialize a training run. Sets job_type='train' automatically."""
        if not self.enabled:
            return
        if model_idx is not None:
            self.model_idx = model_idx
        self._init_run(job_type="train", run_id=run_id)

    def find_training_run_id(
        self, model_name: str, model_idx: int = 0
    ) -> Optional[str]:
        """
        Search for the most recent training run_id for a given model name
        with specified idx.

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

    def init_bench_run(self, run_id: Optional[str] = None) -> None:
        """Initialize a benchmark run. Sets job_type='benchmark' automatically."""
        if not self.enabled:
            return
        if run_id is not None:
            if self.wandb is None:
                raise ValueError("Wandb is not installed")
            try:
                api = self.wandb.Api()
                run_path = f"{self.entity or 'wandb'}/{self.project}/{run_id}"
                run = api.run(run_path)
                if run is None:
                    raise ValueError(
                        f"Wandb run {run_id} does not exist in project {self.project}"
                    )
                self._preserve_run_metadata(run)
            except Exception as e:
                error_str = str(e).lower()
                if (
                    "does not exist" in error_str
                    or "not found" in error_str
                    or "no such" in error_str
                ):
                    raise ValueError(
                        f"Cannot resume wandb run {run_id}: run does not exist "
                        f"in project {self.project}. "
                        f"Please ensure the training run completed successfully."
                    ) from e
            self._init_run(job_type="benchmark", resume="must", run_id=run_id)
        else:
            training_run_id = None
            if self.model and hasattr(self.model, "_get_model_name"):
                model_name = self.model._get_model_name()
                training_run_id = self.find_training_run_id(model_name)

            if training_run_id:
                if self.wandb is not None:
                    api = self.wandb.Api()
                    run_path = (
                        f"{self.entity or 'wandb'}/{self.project}/{training_run_id}"
                    )
                    run = api.run(run_path)
                    if run:
                        self._preserve_run_metadata(run)
                self._init_run(
                    job_type="benchmark", resume="must", run_id=training_run_id
                )
            else:
                self._init_run(job_type="benchmark", resume="allow", run_id=None)

    def _init_run(
        self,
        job_type: str,
        resume: Optional[str] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Internal method to initialize wandb run with specified job_type.
        """
        if not self.enabled:
            return

        if self.initialized and self.run is not None:
            raise ValueError("Wandb run is already initialized. Call finish() first.")

        if self.wandb is None:
            raise ValueError("Wandb is not installed")

        if run_id is not None:
            self.run_id = run_id
        if self.run_id is None:
            raise ValueError("Wandb run_id is None")

        if run_name is None:
            run_name = self._generate_run_name()

        if run_name is None:
            run_name = f"run-{self.run_id}"

        tags = self.tags
        if self.model and hasattr(self.model, "_get_base_model_name"):
            tags = tags + [self.model._get_base_model_name()]

        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "name": run_name,
            "tags": tags,
            "notes": self.notes,
            "entity": self.entity,
            "mode": self.mode,
            "id": self.run_id,
            "job_type": job_type,
        }

        if resume is not None:
            init_kwargs["resume"] = resume

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

        self.run = self.wandb.init(**init_kwargs)
        self.initialized = True

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        self.latest_metrics.update(metrics)
        if self.enabled and self.initialized and self.run is not None:
            self.run.log(metrics, step=step)

    def _preserve_run_metadata(self, run: Any) -> None:
        """Preserve metadata from an existing run."""
        if hasattr(run, "group") and run.group:
            self.group = run.group
        if hasattr(run, "tags") and run.tags:
            existing_tags = set(self.tags)
            new_tags = [tag for tag in run.tags if tag not in existing_tags]
            self.tags = self.tags + new_tags
        if hasattr(run, "notes") and run.notes and not self.notes:
            self.notes = run.notes

    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled and self.initialized and self.run is not None:
            self.run.finish()
            self.initialized = False
            self.run = None
            self.peer_run_ids = []
            self.latest_metrics = {}
