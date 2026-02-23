from pathlib import Path

import pandas as pd
import pytest

from src.experiments.stage1_binary import build_stage1_manifest


def _write_manifest(root: Path, dataset: str, policy: str, rows: list[dict]) -> None:
    manifest_dir = root / dataset / policy / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_dir / "session_manifest.csv", index=False)


def test_stage1_requires_all_datasets(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [
            {
                "session_id": "mfcp_1",
                "dataset": "MFCP",
                "family": "F1",
                "capture_id": "c1.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    with pytest.raises(FileNotFoundError):
        build_stage1_manifest(processed_root=processed_root, policy=policy)


def test_stage1_label_mapping(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    datasets = ["ISCX", "MFCP", "MTA", "USTC-TFC2016"]
    for dataset in datasets:
        _write_manifest(
            processed_root,
            dataset,
            policy,
            [
                {
                    "session_id": f"{dataset}_1",
                    "dataset": dataset,
                    "family": "F1",
                    "capture_id": "c1.pcap",
                    "split": "train",
                    "policy": policy,
                }
            ],
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy)
    assert set(manifest["label_binary"].unique().tolist()) == {0, 1}
    assert (manifest[manifest["dataset"] == "ISCX"]["label_binary"] == 0).all()
    assert (manifest[manifest["dataset"] != "ISCX"]["label_binary"] == 1).all()

