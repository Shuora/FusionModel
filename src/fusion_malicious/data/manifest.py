from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_BINARY_MALICIOUS_DATASETS = {"MTA", "MFCP"}
_BINARY_BENIGN_DATASETS = {"ISCX-VPN-NonVPN-2016", "ISCX-VPN-VPN-2016"}
_MULTICLASS_TASKS = {"mta", "mfcp", "ustc"}


def infer_dataset_and_family(session_path: Path) -> tuple[str, str]:
    parts = list(session_path.parts)
    dataset = "unknown"
    family = "unknown"
    if "SourceData" in parts:
        idx = parts.index("SourceData")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
        if idx + 2 < len(parts):
            family = parts[idx + 2]
    elif len(parts) >= 2:
        dataset = parts[-2]
        family = parts[-1]
    return dataset, family


def _relative_source_path(session_path: Path) -> Path:
    parts = session_path.parts
    if "SourceData" in parts:
        idx = parts.index("SourceData")
        if idx + 1 < len(parts):
            return Path(*parts[idx + 1 :])
    return Path(session_path.name)


def sample_id_from_path(dataset: str, family: str, session_path: Path) -> str:
    relative = _relative_source_path(session_path).with_suffix("")
    identifier = relative.as_posix().replace("/", "_").strip("_")
    if not identifier:
        identifier = session_path.stem
    segments = [segment for segment in (dataset, family, identifier) if segment]
    return "_".join(segments)


def _binary_label_for_dataset(dataset: str) -> tuple[str, int]:
    if dataset in _BINARY_MALICIOUS_DATASETS:
        return "malicious", 1
    if dataset in _BINARY_BENIGN_DATASETS or dataset.lower().startswith("iscx"):
        return "benign", 0
    return "unknown", -1


def build_manifest_dataframe(
    session_paths: Iterable[Path],
    *,
    task_name: str,
) -> pd.DataFrame:
    """
    Build a manifest DataFrame for the given session PCAP paths.
    """
    rows = []
    families = set()
    resolved_paths: list[tuple[Path, str, str]] = []
    for session_path in session_paths:
        dataset, family = infer_dataset_and_family(session_path)
        label_key = family if family != "unknown" else dataset
        families.add(label_key)
        resolved_paths.append((session_path, dataset, family))

    multi_label_map: dict[str, int] = {}
    if task_name in _MULTICLASS_TASKS:
        multi_label_map = {label: index for index, label in enumerate(sorted(families))}

    for session_path, dataset, family in resolved_paths:
        sample_id = sample_id_from_path(dataset, family, session_path)
        if task_name == "binary":
            label_name, label_id = _binary_label_for_dataset(dataset)
        elif task_name in _MULTICLASS_TASKS:
            normalized_family = family if family != "unknown" else dataset
            label_name = normalized_family
            label_id = multi_label_map.get(label_name, -1)
        else:
            label_name, label_id = "unknown", -1
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "family": family,
                "source_path": str(session_path),
                "task_name": task_name,
                "label_name": label_name,
                "label_id": label_id,
            }
        )
    columns = [
        "sample_id",
        "dataset",
        "family",
        "source_path",
        "task_name",
        "label_name",
        "label_id",
    ]
    return pd.DataFrame(rows, columns=columns)
