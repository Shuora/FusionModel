from pathlib import Path

import pandas as pd
import pytest

from src.experiments.stage1_binary import build_stage1_manifest, main


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
    datasets = ["ISCX", "MFCP", "MTA"]
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


def test_stage1_accepts_iscx_alias_directory(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _write_manifest(
        processed_root,
        "ISCX-VPN-NonVPN-2016",
        policy,
        [
            {
                "session_id": "iscx_alias_1",
                "dataset": "ISCX-VPN-NonVPN-2016",
                "family": "F1",
                "capture_id": "c1.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    for dataset in ("MFCP", "MTA"):
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
    assert "ISCX" in set(manifest["dataset"].unique())
    assert (manifest[manifest["dataset"] == "ISCX"]["label_binary"] == 0).all()


def test_stage1_prefers_primary_iscx_directory_when_both_present(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {
                "session_id": "iscx_primary_1",
                "dataset": "ISCX",
                "family": "F1",
                "capture_id": "c1.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    _write_manifest(
        processed_root,
        "ISCX-VPN-NonVPN-2016",
        policy,
        [
            {
                "session_id": "iscx_alias_1",
                "dataset": "ISCX-VPN-NonVPN-2016",
                "family": "F1",
                "capture_id": "c2.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )
    for dataset in ("MFCP", "MTA"):
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
    iscx_rows = manifest[manifest["dataset"] == "ISCX"]
    assert "iscx_primary_1" in set(iscx_rows["session_id"].tolist())
    assert "iscx_alias_1" not in set(iscx_rows["session_id"].tolist())
    assert (iscx_rows["dataset_raw"] == "ISCX").all()


def test_stage1_rejects_empty_required_dataset_manifest(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    for dataset in ("ISCX", "MFCP", "MTA"):
        _write_manifest(processed_root, dataset, policy, [])

    with pytest.raises((FileNotFoundError, ValueError)):
        build_stage1_manifest(processed_root=processed_root, policy=policy)


def test_stage1_does_not_require_ustc_dataset(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    for dataset in ("ISCX", "MFCP", "MTA"):
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
    assert len(manifest) == 3


def test_stage1_manifest_filters_to_mvtba_paper_subset(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    _write_manifest(
        processed_root,
        "ISCX",
        policy,
        [
            {
                "session_id": "iscx_keep",
                "dataset": "ISCX",
                "family": "F1",
                "capture_id": "vpn_facebook_chat1a.pcap",
                "split": "train",
                "policy": policy,
            },
            {
                "session_id": "iscx_drop",
                "dataset": "ISCX",
                "family": "F1",
                "capture_id": "vpn_bittorrent.pcap",
                "split": "train",
                "policy": policy,
            },
        ],
    )
    _write_manifest(
        processed_root,
        "MFCP",
        policy,
        [
            {
                "session_id": "mfcp_keep",
                "dataset": "MFCP",
                "family": "Artemis",
                "capture_id": "a.pcap",
                "split": "train",
                "policy": policy,
            },
            {
                "session_id": "mfcp_drop",
                "dataset": "MFCP",
                "family": "PUA",
                "capture_id": "b.pcap",
                "split": "train",
                "policy": policy,
            },
        ],
    )
    _write_manifest(
        processed_root,
        "MTA",
        policy,
        [
            {
                "session_id": "mta_keep",
                "dataset": "MTA",
                "family": "Dridex",
                "capture_id": "c.pcap",
                "split": "train",
                "policy": policy,
            }
        ],
    )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy)
    assert set(manifest["session_id"].tolist()) == {"iscx_keep", "mfcp_keep", "mta_keep"}


def test_stage1_main_emits_progress_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    processed_root = tmp_path / "processed"
    policy = "session_full"
    for dataset in ("ISCX", "MFCP", "MTA"):
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

    output = tmp_path / "stage1_binary_manifest.csv"
    exit_code = main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Stage1Binary" in captured.out
    assert "Manifest saved" in captured.out
