"""Small shared utilities used across the pipeline."""
from __future__ import annotations

import logging
import random
import sys

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent, readable format.

    Using a shared factory (instead of ad-hoc print statements) means every
    module's output is timestamped and attributable, which matters once the
    pipeline runs unattended (e.g. in CI or a scheduled job).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                               datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_global_seed(seed: int) -> None:
    """Seed every RNG we touch so that a full pipeline run is reproducible
    bit-for-bit given the same input data and config."""
    random.seed(seed)
    np.random.seed(seed)
