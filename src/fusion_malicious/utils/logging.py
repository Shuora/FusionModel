from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence, Union


def _ensure_parent_path(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def create_logger(log_path: Union[Path, str]) -> logging.Logger:
    """Create a file logger that writes INFO messages plus timestamps."""
    output_path = Path(log_path).resolve()
    _ensure_parent_path(output_path)
    logger_name = f"fusion_malicious.logger.{output_path.as_posix()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(output_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def append_metrics_row(
    csv_path: Union[Path, str],
    row: Mapping[str, object],
) -> None:
    """Append a row of scalar metrics to CSV, keeping the header consistent."""
    output_path = Path(csv_path)
    _ensure_parent_path(output_path)
    existing_rows: list[Mapping[str, str]] = []
    existing_fieldnames: Sequence[str] = []
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
    if existing_fieldnames:
        new_keys = [
            key
            for key in row.keys()
            if key not in existing_fieldnames
        ]
        fieldnames = list(existing_fieldnames) + sorted(new_keys)
    else:
        fieldnames = list(row.keys())
    if not existing_fieldnames or fieldnames != list(existing_fieldnames):
        with output_path.open("w", newline="", encoding="utf-8") as sink:
            writer = csv.DictWriter(sink, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow(existing)
    with output_path.open("a", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writerow(row)
