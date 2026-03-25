from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import src.train as train_module
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
    rgbs = np.random.default_rng(123).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(321).integers(0, 1024, size=(n, 128), dtype=np.int32)
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


def test_train_stage_stacking_runs_sub_pipeline(tmp_path: Path):
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
            "stacking",
            "--run-root",
            str(run_root),
            "--run-id",
            "stack-dispatch-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "stack-dispatch-run"
    assert (run_dir / "stacking" / "meta_metrics.json").exists()
    assert (run_dir / "stacking" / "meta_model.json").exists()


def test_train_stage_moe_runs_sub_pipeline(tmp_path: Path):
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
            "moe",
            "--run-root",
            str(run_root),
            "--run-id",
            "moe-dispatch-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "moe-dispatch-run"
    assert (run_dir / "moe" / "router.ckpt").exists()
    assert (run_dir / "moe" / "moe_metrics.json").exists()


def test_dispatch_stage_moe_forwards_lr(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_moe_main(argv):
        captured["argv"] = list(argv)
        return 0

    import src.moe as moe_module

    monkeypatch.setattr(moe_module, "main", fake_moe_main)

    args = argparse.Namespace(moe_epochs=7, batch_size=16, seed=99, lr=0.0005, device="cuda", num_workers=2)
    run_dir = tmp_path / "dispatch-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs = []

    def log(level: str, module: str, event: str, kv: dict) -> None:
        logs.append((level, module, event, kv))

    code = train_module._dispatch_stage(stage="moe", run_dir=run_dir, args=args, log=log)
    assert code == 0
    assert captured["argv"] == [
        "--run-dir",
        str(run_dir),
        "--epochs",
        "7",
        "--batch-size",
        "16",
        "--lr",
        "0.0005",
        "--device",
        "cuda",
        "--num-workers",
        "2",
        "--seed",
        "99",
    ]


def test_dispatch_stage_stacking_forwards_device_and_num_workers(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_stacking_main(argv):
        captured["argv"] = list(argv)
        return 0

    import src.stacking as stacking_module

    monkeypatch.setattr(stacking_module, "main", fake_stacking_main)

    args = argparse.Namespace(
        stacking_n_splits=4,
        stacking_oof_epochs=3,
        batch_size=12,
        seed=77,
        lr=0.001,
        device="auto",
        num_workers=1,
    )
    run_dir = tmp_path / "stacking-dispatch-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs = []

    def log(level: str, module: str, event: str, kv: dict) -> None:
        logs.append((level, module, event, kv))

    code = train_module._dispatch_stage(stage="stacking", run_dir=run_dir, args=args, log=log)
    assert code == 0
    argv = captured["argv"]
    assert "--run-dir" in argv
    idx = argv.index("--run-dir")
    assert argv[idx + 1] == str(run_dir)

    # Contract: runner must wire stacking in artifact-consumer mode via --meta-artifacts-dir.
    assert "--meta-artifacts-dir" in argv
    idx = argv.index("--meta-artifacts-dir")
    assert argv[idx + 1] == str(run_dir / "meta_features")

    # Keep the contract sharp without locking the entire argv ordering.
    assert "--n-splits" in argv
    assert "--oof-epochs" in argv
    assert "--batch-size" in argv
    assert "--device" in argv
    assert "--num-workers" in argv
    assert "--seed" in argv


def test_dispatch_stage_meta_classifier_passes_meta_artifacts_dir(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_stacking_main(argv):
        captured["argv"] = list(argv)
        return 0

    import src.stacking as stacking_module

    monkeypatch.setattr(stacking_module, "main", fake_stacking_main)

    args = argparse.Namespace(
        stacking_n_splits=3,
        stacking_oof_epochs=2,
        batch_size=8,
        seed=123,
        lr=0.001,
        device="cpu",
        num_workers=0,
    )
    run_dir = tmp_path / "meta-dispatch-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs = []

    def log(level: str, module: str, event: str, kv: dict) -> None:
        logs.append((level, module, event, kv))

    code = train_module._dispatch_stage(stage="stacking", run_dir=run_dir, args=args, log=log)
    assert code == 0

    argv = captured["argv"]
    assert "--meta-artifacts-dir" in argv
    idx = argv.index("--meta-artifacts-dir")
    assert argv[idx + 1] == str(run_dir / "meta_features")
