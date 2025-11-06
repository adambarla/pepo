import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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
