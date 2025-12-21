import logging
import sys
from typing import Optional

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
