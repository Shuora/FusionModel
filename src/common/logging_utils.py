from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List


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


def build_multi_file_logger(log_paths: Iterable[Path], name: str = "fusion.run") -> logging.Logger:
    """Create one logger that writes to multiple files."""
    paths: List[Path] = [Path(p) for p in log_paths]
    if not paths:
        raise ValueError("log_paths must not be empty")

    logger = logging.getLogger(f"{name}.{'|'.join(str(p) for p in paths)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for log_path in paths:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger
