"""
utils/logger.py — project-wide structured logger.
"""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "forensight") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger