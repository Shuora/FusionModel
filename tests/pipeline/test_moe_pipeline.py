from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.moe import main as moe_main
from src.train import main as train_main


def _prepare_dummy_processed(root: Path) -> None:
    dataset = "DemoSet"
    policy = "strict"
    rgb_dir = root / dataset / policy / "rgb"
    seq_dir = root / dataset / policy / "seq"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = 12
    session_ids = np.array([f"s{i}" for i in range(1, n + 1)], dtype="U64")
    labels = np.array([0, 1] * (n // 2), dtype=np.int32)
    rgbs = np.random.default_rng(10).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    token_ids = np.random.default_rng(11).integers(0, 1024, size=(n, 256), dtype=np.int32)
    attention = np.ones((n, 256), dtype=np.uint8)
    segments = np.zeros((n, 256), dtype=np.uint8)

    np.savez_compressed(
        rgb_dir / "rgb_shard_00000.npz",
        session_id=session_ids,
        label=labels,
        rgb=rgbs,
    )
    np.savez_compressed(
        seq_dir / "seq_shard_00000.npz",
        session_id=session_ids,
        token_ids=token_ids,
        attention_mask=attention,
        segment_ids=segments,
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
    pd.DataFrame(rows).to_csv(manifest_dir / "session_manifest.csv", index=False)


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
