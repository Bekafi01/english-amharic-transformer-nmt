"""Structured logging utilities for NMT Engine."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_INITIALIZED_LOGGERS: set[str] = set()


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    rich_formatting: bool = True,
) -> None:
    """Configures root logger with optional Rich console output and file logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove and close existing handlers to avoid duplication and file lock leaks
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = []

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if rich_formatting:
        console = Console(file=sys.stdout, legacy_windows=False)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        rich_handler.setLevel(numeric_level)
        handlers.append(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(numeric_level)
        formatter = logging.Formatter(_DEFAULT_FORMAT)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(_DEFAULT_FORMAT)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str = "nmt_engine") -> logging.Logger:
    """Retrieves or creates a named logger instance."""
    logger = logging.getLogger(name)
    if name not in _INITIALIZED_LOGGERS and not logging.getLogger().handlers:
        setup_logging()
        _INITIALIZED_LOGGERS.add(name)
    return logger


def shutdown_logging() -> None:
    """Closes and removes all handlers from the root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    _INITIALIZED_LOGGERS.clear()
