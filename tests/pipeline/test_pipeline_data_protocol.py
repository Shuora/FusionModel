from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.pipeline_data import load_policy_multimodal_data


def _write_dataset(
    root: Path,
    dataset: str,
    policy: str,
    rows: list[dict],
    labels: np.ndarray,
) -> None:
    rgb_dir = root / dataset / policy / "rgb"
    etbert_dir = root / dataset / policy / "etbert"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    etbert_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    session_ids = np.array([str(r["session_id"]) for r in rows], dtype="U64")
    rgbs = np.random.default_rng(1).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(2).integers(0, 1024, size=(n, 128), dtype=np.int32)
    attention = np.ones((n, 128), dtype=np.uint8)
    token_types = np.zeros((n, 128), dtype=np.uint8)

    np.savez_compressed(
        rgb_dir / "rgb_shard_00000.npz",
        session_id=session_ids,
        label=labels.astype(np.int32),
        rgb=rgbs,
    )
    np.savez_compressed(
        etbert_dir / "etbert_shard_00000.npz",
        session_id=session_ids,
        input_ids=input_ids,
        attention_mask=attention,
        token_type_ids=token_types,
    )
    with (manifest_dir / "session_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_policy_multimodal_data_supports_dataset_filter_and_binary_labels(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    iscx_rows = [
        {
            "session_id": "iscx_1",
            "dataset": "ISCX",
            "family": "Chat",
            "capture_id": "a.pcap",
            "split": "train",
            "policy": policy,
        },
        {
            "session_id": "iscx_2",
            "dataset": "ISCX",
            "family": "VoIP",
            "capture_id": "b.pcap",
            "split": "test",
            "policy": policy,
        },
    ]
    mfcp_rows = [
        {
            "session_id": "mfcp_1",
            "dataset": "MFCP",
            "family": "Artemis",
            "capture_id": "a.pcap",
            "split": "train",
            "policy": policy,
        },
        {
            "session_id": "mfcp_2",
            "dataset": "MFCP",
            "family": "Dridex",
            "capture_id": "b.pcap",
            "split": "test",
            "policy": policy,
        },
    ]

    _write_dataset(processed_root, "ISCX", policy, iscx_rows, labels=np.array([0, 1], dtype=np.int32))
    _write_dataset(processed_root, "MFCP", policy, mfcp_rows, labels=np.array([0, 1], dtype=np.int32))

    data = load_policy_multimodal_data(
        processed_root=processed_root,
        policy=policy,
        datasets=["ISCX", "MFCP"],
        label_mode="binary",
    )
    assert data["rgb"].shape[0] == 4
    sid_to_y = {sid: int(y) for sid, y in zip(data["session_id"].tolist(), data["y"].tolist())}
    assert sid_to_y["iscx_1"] == 0
    assert sid_to_y["iscx_2"] == 0
    assert sid_to_y["mfcp_1"] == 1
    assert sid_to_y["mfcp_2"] == 1

    mfcp_only = load_policy_multimodal_data(
        processed_root=processed_root,
        policy=policy,
        datasets=["MFCP"],
        label_mode="multiclass",
    )
    assert set(mfcp_only["session_id"].tolist()) == {"mfcp_1", "mfcp_2"}


def test_load_policy_multimodal_data_supports_session_filter_manifest(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    rows = [
        {
            "session_id": "iscx_keep",
            "dataset": "ISCX",
            "family": "Chat",
            "capture_id": "vpn_facebook_chat1a.pcap",
            "split": "train",
            "policy": policy,
        },
        {
            "session_id": "iscx_drop",
            "dataset": "ISCX",
            "family": "Chat",
            "capture_id": "vpn_bittorrent.pcap",
            "split": "test",
            "policy": policy,
        },
    ]
    _write_dataset(processed_root, "ISCX", policy, rows, labels=np.array([0, 1], dtype=np.int32))

    filter_manifest = tmp_path / "stage1_filter.csv"
    with filter_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id"])
        writer.writeheader()
        writer.writerow({"session_id": "iscx_keep"})

    data = load_policy_multimodal_data(
        processed_root=processed_root,
        policy=policy,
        datasets=["ISCX"],
        label_mode="binary",
        session_filter_manifest=filter_manifest,
    )
    assert set(data["session_id"].tolist()) == {"iscx_keep"}


def test_load_policy_multimodal_data_reads_etbert_triplet(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    rows = [
        {
            "session_id": "demo_1",
            "dataset": "MFCP",
            "family": "Artemis",
            "capture_id": "a.pcap",
            "split": "train",
            "policy": policy,
        },
        {
            "session_id": "demo_2",
            "dataset": "MFCP",
            "family": "Dridex",
            "capture_id": "b.pcap",
            "split": "test",
            "policy": policy,
        },
    ]
    _write_dataset(processed_root, "MFCP", policy, rows, labels=np.array([0, 1], dtype=np.int32))

    data = load_policy_multimodal_data(processed_root=processed_root, policy=policy, datasets=["MFCP"])

    assert data["rgb"].shape[0] == 2
    assert data["input_ids"].shape == (2, 128)
    assert data["attention_mask"].shape == (2, 128)
    assert data["token_type_ids"].shape == (2, 128)


def test_load_policy_multimodal_data_session_filter_manifest_can_override_explicit_val_split(tmp_path: Path):
    processed_root = tmp_path / "processed"
    policy = "session_full"

    rows = [
        {
            "session_id": "iscx_train",
            "dataset": "ISCX",
            "family": "Chat",
            "capture_id": "vpn_facebook_chat1a.pcap",
            "split": "train",
            "policy": policy,
        },
        {
            "session_id": "iscx_holdout",
            "dataset": "ISCX",
            "family": "Chat",
            "capture_id": "vpn_facebook_chat1b.pcap",
            "split": "train",
            "policy": policy,
        },
    ]
    _write_dataset(processed_root, "ISCX", policy, rows, labels=np.array([0, 1], dtype=np.int32))

    filter_manifest = tmp_path / "stage1_score_optimized_manifest.csv"
    with filter_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "split", "dataset"])
        writer.writeheader()
        writer.writerow({"session_id": "iscx_train", "split": "train", "dataset": "ISCX"})
        writer.writerow({"session_id": "iscx_holdout", "split": "val", "dataset": "ISCX"})

    data = load_policy_multimodal_data(
        processed_root=processed_root,
        policy=policy,
        datasets=["ISCX"],
        label_mode="binary",
        session_filter_manifest=filter_manifest,
    )
    sid_to_split = {sid: split for sid, split in zip(data["session_id"].tolist(), data["split"].tolist())}
    assert sid_to_split["iscx_train"] == "train"
    assert sid_to_split["iscx_holdout"] == "val"
