from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.moe as moe_module
from src.moe import main as moe_main
from src.train import main as train_main


def _prepare_dummy_processed(root: Path) -> None:
    dataset = "DemoSet"
    policy = "strict"
    rgb_dir = root / dataset / policy / "rgb"
    etbert_dir = root / dataset / policy / "etbert"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    etbert_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = 12
    session_ids = np.array([f"s{i}" for i in range(1, n + 1)], dtype="U64")
    labels = np.array([0, 1] * (n // 2), dtype=np.int32)
    rgbs = np.random.default_rng(10).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(11).integers(0, 1024, size=(n, 128), dtype=np.int32)
    attention = np.ones((n, 128), dtype=np.uint8)
    token_types = np.zeros((n, 128), dtype=np.uint8)

    np.savez_compressed(
        rgb_dir / "rgb_shard_00000.npz",
        session_id=session_ids,
        label=labels,
        rgb=rgbs,
    )
    np.savez_compressed(
        etbert_dir / "etbert_shard_00000.npz",
        session_id=session_ids,
        input_ids=input_ids,
        attention_mask=attention,
        token_type_ids=token_types,
    )

    splits = ["train"] * 6 + ["val"] * 4 + ["test"] * 2
    families = ["Fam0" if x == 0 else "Fam1" for x in labels]
    rows = [
        {
            "session_id": sid,
            "dataset": dataset,
            "family": fam,
            "capture_id": f"{fam}.pcap",
            "split": sp,
            "policy": policy,
        }
        for sid, fam, sp in zip(session_ids, families, splits)
    ]
    with (manifest_dir / "session_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_moe_pipeline_outputs_router_and_metrics(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed(processed_root)

    code = train_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "moe-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "moe-run"
    code = moe_main(
        [
            "--run-dir",
            str(run_dir),
            "--epochs",
            "2",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    moe_dir = run_dir / "moe"
    assert (moe_dir / "router.ckpt").exists()
    assert (moe_dir / "moe_metrics.json").exists()

    metrics = json.loads((moe_dir / "moe_metrics.json").read_text(encoding="utf-8"))
    assert "top1" in metrics
    assert "macro_f1" in metrics
    assert metrics["n_test_samples"] > 0


def test_moe_router_features_follow_shared_meta_schema_contract():
    try:
        from src.pipeline.meta_features import build_meta_feature_blocks
    except ImportError as e:  # pragma: no cover
        raise AssertionError("MoE should consume shared stage2 meta feature helper") from e

    out = {
        "logits_img": torch.tensor([[2.0, 0.0, -1.0], [0.2, 1.4, -0.5]], dtype=torch.float32),
        "logits_tls": torch.tensor([[0.5, 1.0, -0.5], [1.1, -0.1, 0.4]], dtype=torch.float32),
        "logits_fuse": torch.tensor([[1.2, 0.3, -0.2], [0.9, 0.8, -0.3]], dtype=torch.float32),
        "summary": {
            "img_pooled_norm": torch.tensor([[0.8], [0.9]], dtype=torch.float32),
            "txt_pooled_norm": torch.tensor([[0.5], [0.4]], dtype=torch.float32),
            "fused_norm": torch.tensor([[0.7], [0.6]], dtype=torch.float32),
        },
    }
    blocks = build_meta_feature_blocks(out)
    router_x = moe_module._router_features(out)

    expected_router = torch.cat(
        [
            blocks["confidence"]["entropy"],
            blocks["agreement"],
            blocks["confidence"]["max_prob"],
        ],
        dim=1,
    )
    assert router_x.shape == (2, 7)
    assert torch.allclose(router_x, expected_router, atol=1e-6)
