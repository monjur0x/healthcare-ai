"""
Centralized logger for preprocessing.

All preprocessing modules should use:

logger = get_logger(__name__)
"""

from __future__ import annotations

import logging

from logging.handlers import RotatingFileHandler

from .config import settings

settings.LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Create or return a configured logger.

    Parameters
    ----------
    name : str
        Usually __name__

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "preprocessing.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
