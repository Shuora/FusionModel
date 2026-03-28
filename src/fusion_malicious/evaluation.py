from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

import pandas as pd

LabelValue = Union[int, str]


def _normalize_label_value(value: Any) -> LabelValue | None:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return str(value)
    return value


def _label_sort_key(value: LabelValue) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        return (1, value)
    return (2, str(value))


def _format_label_name(name: Any, fallback: LabelValue) -> str:
    if name is None or (isinstance(name, str) and not name.strip()):
        return str(fallback)
    return str(name)


def load_manifest_dataframe(
    manifest_path: Path,
    *,
    subset: str | None = None,
    subset_column: str = "subset",
) -> pd.DataFrame:
    """Return manifest records, filtering by subset when requested."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file {manifest_path} does not exist.")
    frame = pd.read_csv(manifest_path)
    if subset is not None:
        if subset_column not in frame.columns:
            raise KeyError(f"Subset column '{subset_column}' not found in manifest.")
        frame = frame[frame[subset_column].astype(str) == str(subset)]
    if frame.empty:
        column_info = f" subset '{subset}'" if subset is not None else ""
        raise ValueError(f"Manifest contains no entries{column_info}.")
    if "cache_path" not in frame.columns:
        raise KeyError("Manifest requires a 'cache_path' column pointing to cached samples.")
    return frame.reset_index(drop=True)


def resolve_label_names(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label_id",
    name_column: str = "label_name",
) -> tuple[list[LabelValue], list[str]]:
    """Map label ids to display names using manifest metadata."""
    if label_column not in dataframe.columns:
        raise KeyError(f"Label column '{label_column}' not found in manifest.")
    label_map: dict[LabelValue, str] = {}
    default_names: Sequence[Any] = (
        dataframe[name_column] if name_column in dataframe.columns else [None] * len(dataframe)
    )
    for raw_label, raw_name in zip(dataframe[label_column], default_names):
        label = _normalize_label_value(raw_label)
        if label is None or label in label_map:
            continue
        label_map[label] = _format_label_name(raw_name, label)
    if not label_map:
        raise ValueError("Manifest must include at least one non-null label_id.")
    ordered = sorted(label_map.keys(), key=_label_sort_key)
    return ordered, [label_map[label] for label in ordered]
