from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluate import main as eval_main
from src.report import main as report_main
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

    session_ids = np.array([f"s{i}" for i in range(1, 9)], dtype="U64")
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    rgbs = np.random.default_rng(42).integers(0, 256, size=(8, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(43).integers(0, 1024, size=(8, 128), dtype=np.int32)
    attention = np.ones((8, 128), dtype=np.uint8)
    token_types = np.zeros((8, 128), dtype=np.uint8)

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

    splits = ["train", "train", "train", "train", "val", "val", "test", "test"]
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


def _prepare_dummy_processed_without_test(root: Path) -> None:
    dataset = "DemoSet"
    policy = "strict"
    rgb_dir = root / dataset / policy / "rgb"
    etbert_dir = root / dataset / policy / "etbert"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    etbert_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    session_ids = np.array([f"m{i}" for i in range(1, 7)], dtype="U64")
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
    rgbs = np.random.default_rng(1234).integers(0, 256, size=(6, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(2345).integers(0, 1024, size=(6, 128), dtype=np.int32)
    attention = np.ones((6, 128), dtype=np.uint8)
    token_types = np.zeros((6, 128), dtype=np.uint8)

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

    splits = ["train", "train", "train", "val", "val", "val"]
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


def _write_minimal_run_dir(run_dir: Path, stage: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_dir.name,
                "stage": stage,
                "policy": "strict",
                "epochs": 2,
                "batch_size": 4,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "train_acc": 0.6, "val_acc": 0.7, "val_macro_f1": 0.68},
            {"epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "train_acc": 0.8, "val_acc": 0.85, "val_macro_f1": 0.83},
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)


def test_train_evaluate_report_smoke(tmp_path: Path):
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
            "smoke-run",
            "--epochs",
            "2",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "smoke-run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "checkpoints" / "best.ckpt").exists()
    cfg_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "model_type: MobileViTETBertFusionClassifier" in cfg_text

    code = eval_main(
        [
            "--run-dir",
            str(run_dir),
            "--split",
            "test",
        ]
    )
    assert code == 0
    assert (run_dir / "eval_test.json").exists()
    assert (run_dir / "figures" / "confusion_matrix_test.csv").exists()
    assert (run_dir / "figures" / "confusion_matrix_test.png").exists()

    eval_payload = pd.read_json(run_dir / "eval_test.json", typ="series")
    assert "macro_precision" in eval_payload.index
    assert eval_payload["split"] == "test"

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    assert (run_dir / "report.md").exists()
    assert (run_dir / "figures" / "learning_curve.png").exists()
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "gate_mean" in train_log
    assert "git_commit=" in train_log
    assert "config_summary" in train_log
    assert "dataset_stats" in train_log
    assert "train_macroF1" in train_log


def test_evaluate_fallback_uses_effective_split_and_report_discovers_it(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed_without_test(processed_root)

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
            "fallback-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "fallback-run"
    code = eval_main(
        [
            "--run-dir",
            str(run_dir),
            "--split",
            "test",
            "--allow-split-fallback",
        ]
    )
    assert code == 0
    assert not (run_dir / "eval_test.json").exists()
    assert (run_dir / "eval_val.json").exists()
    assert (run_dir / "figures" / "confusion_matrix_val.csv").exists()
    assert (run_dir / "figures" / "confusion_matrix_val.png").exists()

    payload = pd.read_json(run_dir / "eval_val.json", typ="series")
    assert payload["requested_split"] == "test"
    assert payload["effective_split"] == "val"
    assert payload["split"] == "val"
    assert bool(payload["fallback_used"]) is True

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: eval" in report_text
    assert "eval_val.json" in report_text
    assert "confusion_matrix_val.csv" in report_text


def test_report_falls_back_to_stacking_metrics_when_eval_missing(tmp_path: Path):
    run_dir = tmp_path / "runs" / "stacking-report-run"
    _write_minimal_run_dir(run_dir, stage="stacking")
    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    (stack_dir / "meta_metrics.json").write_text(
        json.dumps({"top1": 0.91, "macro_f1": 0.90, "macro_recall": 0.89, "n_test_samples": 12}),
        encoding="utf-8",
    )

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: stacking" in report_text
    assert "Top-1: 0.9100" in report_text
    assert "meta_metrics.json" in report_text
    assert "confusion_matrix_test.csv" not in report_text


def test_report_falls_back_to_moe_metrics_when_eval_missing(tmp_path: Path):
    run_dir = tmp_path / "runs" / "moe-report-run"
    _write_minimal_run_dir(run_dir, stage="moe")
    moe_dir = run_dir / "moe"
    moe_dir.mkdir(parents=True, exist_ok=True)
    (moe_dir / "moe_metrics.json").write_text(
        json.dumps({"top1": 0.88, "macro_f1": 0.87, "macro_recall": 0.86, "n_test_samples": 12}),
        encoding="utf-8",
    )

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: moe" in report_text
    assert "Top-1: 0.8800" in report_text
    assert "moe_metrics.json" in report_text
    assert "confusion_matrix_test.csv" not in report_text


def test_evaluate_fails_when_requested_split_missing(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed_without_test(processed_root)

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
            "no-test-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "no-test-run"
    code = eval_main(
        [
            "--run-dir",
            str(run_dir),
            "--split",
            "test",
        ]
    )
    assert code != 0
    assert not (run_dir / "eval_test.json").exists()
