import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    """
    A general-purpose logging class that provides a simple API for logging.
    Uses Python's logging library under the hood.
    """

    def __init__(
        self,
        name: str = "pepo",
        log_file: Optional[str] = None,
        log_dir: str = "logs",
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        date_format: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name (used to identify different loggers).
            log_file: Path to log file. If None, auto-generates with timestamp.
            log_dir: Directory for log files (created if doesn't exist).
            level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL).
            format_string: Custom format string for log messages.
            date_format: Custom date format string.
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = self.log_dir / f"{name}_{timestamp}.log"
        else:
            log_file_path = Path(log_file)
            if not log_file_path.is_absolute():
                log_file_path = self.log_dir / log_file_path

        self.log_file = log_file_path
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        self.logger.propagate = False

        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"

        formatter = logging.Formatter(format_string, datefmt=date_format)

        file_handler = logging.FileHandler(self.log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log a critical message."""
        self.logger.critical(message)

    def exception(self, message: str):
        """Log an exception with traceback."""
        self.logger.exception(message)

    def get_logger(self):
        """Get the underlying Python logger object for advanced usage."""
        return self.logger


class WandbHandler:
    """
    Handler for Weights & Biases logging.
    Manages wandb initialization and provides logging interface.
    """

    def __init__(
        self,
        enabled: bool = False,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "online",
        cfg: Optional[Dict[str, Any]] = None,
        reinit: Optional[str] = None,
        group: Optional[str] = None,
        _lazy_init: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize wandb handler.

        Args:
            enabled: Whether wandb logging is enabled.
            project: Wandb project name.
            name: Wandb run name (None = auto-generate).
            tags: List of tags for the run.
            notes: Notes/description for the run.
            entity: Wandb entity/team name.
            mode: Wandb mode ("online", "offline", "disabled").
            cfg: Configuration dictionary to log to wandb.
            reinit: How to handle multiple runs ("create_new", "finish_previous", "return_previous").
            group: Group name for organizing related runs together.
        """
        self.enabled = enabled
        self.wandb = wandb
        self.initialized = False
        self.run: Optional[Any] = None
        self.project = project
        self.name = name
        self.tags = tags or []
        self.notes = notes
        self.entity = entity
        self.mode = mode
        self.cfg = cfg
        self.group = group
        self._lazy_init = _lazy_init
        self.logger = logger

        if wandb is not None:
            self.run_id: Optional[str] = wandb.util.generate_id()  # type: ignore[attr-defined]
        else:
            self.run_id = None

        if enabled and not _lazy_init:
            self.init_run()

    def init_run(self):
        """
        Initialize the wandb run.
        """
        if not self.enabled:
            if self.logger is not None:
                self.logger.info("Wandb logging is disabled, skipping init_run")
            return

        if self.initialized and self.run is not None:
            if self.logger is not None:
                self.logger.warning("Wandb run is already initialized, skipping init_run")
            return

        if self.wandb is None:
            if self.logger is not None:
                self.logger.error("Wandb is not installed, skipping init_run")
            return

        if self.run_id is None:
            if self.logger is not None:
                self.logger.error("Wandb run_id is None, skipping init_run")
            return

        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "name": self.name,
            "tags": self.tags,
            "notes": self.notes,
            "entity": self.entity,
            "mode": self.mode,
            "id": self.run_id,
        }
        if self.group is not None:
            init_kwargs["group"] = self.group
        if self.cfg is not None:
            init_kwargs["config"] = self.cfg
        self.run = self.wandb.init(**init_kwargs)  # type: ignore[assignment,arg-type]
        self.initialized = True

        if self.logger is not None:
            self.logger.info(
                f"Wandb run {self.name} was initialized. Run: {self.run_id}, Group: {self.group}, Project: {self.project}, Entity: {self.entity}"
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.
        """
        if self.enabled and self.initialized and self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception:
                if self.logger is not None:
                    self.logger.exception(
                        f"Failed to log metrics to wandb. Metrics: {metrics}, Step: {step}"
                    )

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.initialized and self.run is not None:
            self.run.finish()
            self.initialized = False
            self.run = None
            if self.logger is not None:
                self.logger.info(f"Wandb run {self.name} finished.")
