from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stacking import main as stacking_main
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
    rgbs = np.random.default_rng(7).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(8).integers(0, 1024, size=(n, 128), dtype=np.int32)
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


def test_stacking_pipeline_generates_meta_artifacts(tmp_path: Path):
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
            "stack-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "stack-run"
    code = stacking_main(
        [
            "--run-dir",
            str(run_dir),
            "--n-splits",
            "2",
            "--oof-epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    stack_dir = run_dir / "stacking"
    assert (stack_dir / "oof_meta_train.npz").exists()
    assert (stack_dir / "meta_test.npz").exists()
    assert (stack_dir / "meta_metrics.json").exists()
    assert (stack_dir / "meta_model.json").exists()

    metrics = json.loads((stack_dir / "meta_metrics.json").read_text(encoding="utf-8"))
    assert "top1" in metrics
    assert "macro_f1" in metrics
    assert metrics["n_train_samples"] > 0
