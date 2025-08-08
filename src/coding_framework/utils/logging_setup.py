"""
Logging setup utilities for structured logging throughout the framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    level: str = "INFO",
    verbose: bool = False,
    log_file: Optional[str] = None,
    format_type: str = "json",
) -> None:
    """
    Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        verbose: Enable verbose output
        log_file: Optional log file path
        format_type: Log format type ("json" or "text")
    """
    # Convert level string to logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    if verbose:
        log_level = logging.DEBUG

    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="ISO")

    # Configure processors based on format type
    if format_type == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        format="%(message)s",  # structlog handles formatting
    )

    # Silence noisy third-party loggers
    _silence_noisy_loggers()


def _silence_noisy_loggers() -> None:
    """Silence overly verbose third-party loggers."""
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "docker",
        "ray",
        "wandb",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
