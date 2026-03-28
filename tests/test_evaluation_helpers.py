from pathlib import Path

import pandas as pd
import pytest

from fusion_malicious.evaluation import load_manifest_dataframe, resolve_label_names


def test_load_manifest_dataframe_filters_subset(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    df = pd.DataFrame(
        [
            {"cache_path": "cache/a.npz", "subset": "train", "label_id": 1},
            {"cache_path": "cache/b.npz", "subset": "test", "label_id": 0},
        ]
    )
    df.to_csv(manifest, index=False)
    loaded = load_manifest_dataframe(manifest, subset="test")
    assert len(loaded) == 1
    assert loaded.iloc[0]["subset"] == "test"


def test_load_manifest_dataframe_requires_cache_path(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_cache.csv"
    df = pd.DataFrame([{"subset": "test", "label_id": 0}])
    df.to_csv(manifest, index=False)
    with pytest.raises(KeyError):
        load_manifest_dataframe(manifest)


def test_resolve_label_names_orders_and_defaults() -> None:
    df = pd.DataFrame(
        {
            "label_id": [2.0, 1, "b", "a"],
            "label_name": ["two", None, "bee", ""],
        }
    )
    label_ids, label_names = resolve_label_names(df)
    assert label_ids == [1, 2, "a", "b"]
    assert label_names == ["1", "two", "a", "bee"]


def test_resolve_label_names_rejects_missing_labels() -> None:
    df = pd.DataFrame({"label_id": [None, float("nan")]})
    with pytest.raises(ValueError):
        resolve_label_names(df)
