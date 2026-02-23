from __future__ import annotations

import logging
from pathlib import Path


def build_file_logger(log_path: Path, name: str = "fusion.train") -> logging.Logger:
    """Create a file logger for one training run."""
    logger = logging.getLogger(f"{name}.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    return logger
