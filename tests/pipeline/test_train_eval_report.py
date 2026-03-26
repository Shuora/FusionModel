from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import src.evaluate as evaluate_module
import src.train as train_module
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


def _prepare_dummy_processed_for_dataset(root: Path, dataset: str, *, num_classes: int = 2) -> None:
    policy = "strict"
    rgb_dir = root / dataset / policy / "rgb"
    etbert_dir = root / dataset / policy / "etbert"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    etbert_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    session_ids = np.array([f"{dataset.lower()}-{i}" for i in range(1, 9)], dtype="U64")
    labels = np.array([idx % num_classes for idx in range(8)], dtype=np.int32)
    rgbs = np.random.default_rng(142).integers(0, 256, size=(8, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(143).integers(0, 512, size=(8, 128), dtype=np.int32)
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
    rows = [
        {
            "session_id": sid,
            "dataset": dataset,
            "family": f"Fam{label}",
            "capture_id": f"{dataset.lower()}-{label}.pcap",
            "split": split,
            "policy": policy,
        }
        for sid, label, split in zip(session_ids, labels, splits)
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
    assert "paper_precision" in eval_payload.index
    assert "paper_recall" in eval_payload.index
    assert "paper_f1" in eval_payload.index
    assert "paper_macro_precision" in eval_payload.index
    assert "paper_macro_recall" in eval_payload.index
    assert "paper_macro_f1" in eval_payload.index
    assert eval_payload["split"] == "test"

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    assert (run_dir / "report.md").exists()
    assert (run_dir / "figures" / "learning_curve.png").exists()
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Paper-Compatible Metrics" in report_text
    assert "Paper Macro-F1" in report_text
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "fuse_conf_mean" in train_log
    assert "git_commit=" in train_log
    assert "config_summary" in train_log
    assert "dataset_stats" in train_log
    assert "train_macroF1" in train_log


def test_compute_classification_metrics_includes_paper_compatible_macro_f1():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    pred = np.array([0, 0, 0, 1, 1, 2, 2, 1, 1], dtype=np.int64)

    metrics = evaluate_module.compute_classification_metrics(y_true=y_true, pred=pred, num_classes=3)

    assert metrics["top1"] == pytest.approx(5 / 9)
    assert metrics["paper_macro_precision"] == pytest.approx(0.5833333333)
    assert metrics["paper_macro_recall"] == pytest.approx(0.5277777778)
    assert metrics["paper_macro_f1"] == pytest.approx(0.5541666667)
    assert metrics["macro_f1"] == pytest.approx(0.5301587302)
    assert metrics["paper_macro_f1"] != pytest.approx(metrics["macro_f1"])


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


def test_report_best_validation_prefers_best_checkpoint_epoch_when_available(tmp_path: Path):
    run_dir = tmp_path / "runs" / "report-best-ckpt-run"
    _write_minimal_run_dir(run_dir, stage="fusion")
    metrics = pd.DataFrame(
        [
            {"epoch": 1, "train_loss": 0.9, "val_loss": 0.7, "train_acc": 0.7, "val_acc": 0.95, "val_macro_f1": 0.80},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.6, "train_acc": 0.8, "val_acc": 0.93, "val_macro_f1": 0.96},
        ]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 1, "best_metric": "val_acc", "best_metric_value": 0.95}, ckpt_dir / "best.ckpt")

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Best Epoch: 1" in report_text
    assert "Val Acc: 0.9500" in report_text


def test_evaluate_writes_classification_report_artifacts(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "eval-artifacts-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "eval-artifacts-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((4, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((4, 128), dtype=np.int32),
            "attention_mask": np.ones((4, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((4, 128), dtype=np.uint8),
            "y": np.array([0, 1, 0, 1], dtype=np.int32),
            "split": np.array(["test", "test", "test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {"model_state": {}})

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, use_fusion=True):
            logits = torch.tensor(
                [
                    [3.0, 1.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [3.0, 1.0],
                ],
                dtype=torch.float32,
                device=rgb.device,
            )
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(["--run-dir", str(run_dir), "--split", "test", "--device", "cpu"])
    assert code == 0
    assert (run_dir / "figures" / "classification_report_test.csv").exists()
    assert (run_dir / "figures" / "classification_report_test.json").exists()


def test_evaluate_fallback_writes_classification_report_with_effective_split(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "eval-fallback-artifacts-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "eval-fallback-artifacts-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((4, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((4, 128), dtype=np.int32),
            "attention_mask": np.ones((4, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((4, 128), dtype=np.uint8),
            "y": np.array([0, 1, 0, 1], dtype=np.int32),
            "split": np.array(["val", "val", "val", "val"], dtype="U8"),
        },
    )
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {"model_state": {}})

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, use_fusion=True):
            logits = torch.tensor(
                [
                    [3.0, 1.0],
                    [1.0, 3.0],
                    [1.0, 3.0],
                    [3.0, 1.0],
                ],
                dtype=torch.float32,
                device=rgb.device,
            )
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(
        ["--run-dir", str(run_dir), "--split", "test", "--device", "cpu", "--allow-split-fallback"]
    )
    assert code == 0
    assert not (run_dir / "figures" / "classification_report_test.csv").exists()
    assert (run_dir / "figures" / "classification_report_val.csv").exists()
    assert (run_dir / "figures" / "classification_report_val.json").exists()


def test_report_renders_confusion_matrix_and_classification_tables(tmp_path: Path):
    run_dir = tmp_path / "runs" / "report-table-run"
    _write_minimal_run_dir(run_dir, stage="fusion")
    (run_dir / "eval_test.json").write_text(
        json.dumps(
            {
                "top1": 0.75,
                "macro_precision": 0.75,
                "macro_f1": 0.7333,
                "macro_recall": 0.75,
                "num_samples": 4,
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[1, 1], [0, 2]], columns=["0", "1"]).to_csv(fig_dir / "confusion_matrix_test.csv", index=False)
    pd.DataFrame(
        [
            {"label": "0", "precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2},
            {"label": "1", "precision": 0.6667, "recall": 1.0, "f1": 0.8, "support": 2},
        ]
    ).to_csv(fig_dir / "classification_report_test.csv", index=False)
    (fig_dir / "confusion_matrix_test.png").write_bytes(b"png")

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Confusion Matrix" in report_text
    assert "## Classification Report" in report_text
    assert "| true/pred | 0 | 1 |" in report_text
    assert "| label | precision | recall | f1 | support |" in report_text


def test_report_discovers_eval_val_and_renders_tables(tmp_path: Path):
    run_dir = tmp_path / "runs" / "report-val-table-run"
    _write_minimal_run_dir(run_dir, stage="fusion")
    (run_dir / "eval_val.json").write_text(
        json.dumps(
            {
                "top1": 0.75,
                "macro_precision": 0.75,
                "macro_f1": 0.7333,
                "macro_recall": 0.75,
                "num_samples": 4,
                "requested_split": "test",
                "effective_split": "val",
                "split": "val",
                "fallback_used": True,
            }
        ),
        encoding="utf-8",
    )
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[1, 1], [0, 2]], columns=["0", "1"]).to_csv(fig_dir / "confusion_matrix_val.csv", index=False)
    pd.DataFrame(
        [
            {"label": "0", "precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2},
            {"label": "1", "precision": 0.6667, "recall": 1.0, "f1": 0.8, "support": 2},
        ]
    ).to_csv(fig_dir / "classification_report_val.csv", index=False)
    (fig_dir / "classification_report_val.json").write_text("[]", encoding="utf-8")
    (fig_dir / "confusion_matrix_val.png").write_bytes(b"png")

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "eval_val.json" in report_text
    assert "classification_report_val.csv" in report_text
    assert "## Confusion Matrix" in report_text
    assert "## Classification Report" in report_text


def test_evaluate_accepts_short_run_dir_and_resolves_dated_partition(tmp_path: Path, monkeypatch):
    dated_run_dir = tmp_path / "runs" / "2026-03-21" / "stage1-binary"
    (dated_run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (dated_run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "stage1-binary",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((4, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((4, 128), dtype=np.int32),
            "attention_mask": np.ones((4, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((4, 128), dtype=np.uint8),
            "y": np.array([0, 1, 0, 1], dtype=np.int32),
            "split": np.array(["test", "test", "test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {"model_state": {}})

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, use_fusion=True):
            logits = torch.tensor(
                [[3.0, 1.0], [1.0, 3.0], [1.0, 3.0], [3.0, 1.0]],
                dtype=torch.float32,
                device=rgb.device,
            )
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(["--run-dir", "runs/stage1-binary", "--split", "test", "--device", "cpu"])
    assert code == 0
    assert (dated_run_dir / "eval_test.json").exists()


def test_report_accepts_short_run_dir_and_resolves_dated_partition(tmp_path: Path, monkeypatch):
    dated_run_dir = tmp_path / "runs" / "2026-03-21" / "stage1-binary"
    _write_minimal_run_dir(dated_run_dir, stage="fusion")
    (dated_run_dir / "eval_test.json").write_text(
        json.dumps(
            {
                "top1": 0.75,
                "macro_precision": 0.75,
                "macro_f1": 0.7333,
                "macro_recall": 0.75,
                "num_samples": 4,
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    fig_dir = dated_run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[1, 1], [0, 2]], columns=["0", "1"]).to_csv(fig_dir / "confusion_matrix_test.csv", index=False)
    pd.DataFrame(
        [
            {"label": "0", "precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2},
            {"label": "1", "precision": 0.6667, "recall": 1.0, "f1": 0.8, "support": 2},
        ]
    ).to_csv(fig_dir / "classification_report_test.csv", index=False)
    (fig_dir / "classification_report_test.json").write_text("[]", encoding="utf-8")
    (fig_dir / "confusion_matrix_test.png").write_bytes(b"png")
    monkeypatch.chdir(tmp_path)

    code = report_main(["--run-dir", "runs/stage1-binary"])
    assert code == 0
    assert (dated_run_dir / "report.md").exists()


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


def test_train_writes_resolved_device_and_num_workers(tmp_path: Path, monkeypatch):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed(processed_root)

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)

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
            "device-workers-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "auto",
            "--num-workers",
            "3",
        ]
    )
    assert code == 0

    cfg = yaml.safe_load((run_root / "device-workers-run" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["device"] == "cpu"
    assert int(cfg["num_workers"]) == 3


def test_train_defaults_num_workers_to_four(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    captured_num_workers = []

    class StopAfterLoaderInit(Exception):
        pass

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.random.default_rng(1).integers(0, 256, size=(4, 3, 28, 28), dtype=np.uint8),
            "input_ids": np.random.default_rng(2).integers(0, 1024, size=(4, 128), dtype=np.int32),
            "attention_mask": np.ones((4, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((4, 128), dtype=np.uint8),
            "y": np.array([0, 1, 0, 1], dtype=np.int32),
            "split": np.array(["train", "train", "val", "val"], dtype="U8"),
        },
    )

    def fake_dataloader(*args, **kwargs):
        captured_num_workers.append(kwargs["num_workers"])
        if len(captured_num_workers) == 2:
            raise StopAfterLoaderInit()
        return object()

    monkeypatch.setattr(train_module, "DataLoader", fake_dataloader)

    with pytest.raises(StopAfterLoaderInit):
        train_main(
            [
                "--processed-root",
                str(tmp_path / "outputs" / "processed"),
                "--policy",
                "strict",
                "--stage",
                "fusion",
                "--run-root",
                str(run_root),
                "--run-id",
                "default-workers-run",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--no-progress",
            ]
        )

    cfg = yaml.safe_load((run_root / "default-workers-run" / "config.yaml").read_text(encoding="utf-8"))
    assert int(cfg["num_workers"]) == 4
    assert captured_num_workers == [4, 4]


def test_evaluate_records_cpu_fallback_when_cuda_unavailable(tmp_path: Path, monkeypatch):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed(processed_root)

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
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
            "eval-device-fallback-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )
    assert code == 0

    run_dir = run_root / "eval-device-fallback-run"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg["device_requested"] = "cuda"
    cfg["device"] = "cuda"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(evaluate_module.torch.cuda, "is_available", lambda: False)
    code = eval_main(["--run-dir", str(run_dir), "--split", "test"])
    assert code == 0

    payload = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    assert payload["device_requested"] == "cuda"
    assert payload["device"] == "cpu"


def test_derive_validation_mask_from_train_is_stratified():
    train_mask = np.array([True] * 10, dtype=bool)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    new_train_mask, val_mask = train_module._derive_validation_mask_from_train(
        train_mask=train_mask,
        y=y,
        seed=7,
        val_fraction=0.2,
    )

    assert int(val_mask.sum()) == 2
    assert set(y[val_mask].tolist()) == {0, 1}
    assert set(y[new_train_mask].tolist()) == {0, 1}


def test_train_writes_best_metric_to_config(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"

    class StopAfterLoaderInit(Exception):
        pass

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.random.default_rng(11).integers(0, 256, size=(6, 3, 28, 28), dtype=np.uint8),
            "input_ids": np.random.default_rng(12).integers(0, 1024, size=(6, 128), dtype=np.int32),
            "attention_mask": np.ones((6, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((6, 128), dtype=np.uint8),
            "y": np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
            "split": np.array(["train", "train", "train", "val", "val", "val"], dtype="U8"),
        },
    )

    def fake_dataloader(*args, **kwargs):
        raise StopAfterLoaderInit()

    monkeypatch.setattr(train_module, "DataLoader", fake_dataloader)

    with pytest.raises(StopAfterLoaderInit):
        train_main(
            [
                "--processed-root",
                str(tmp_path / "outputs" / "processed"),
                "--policy",
                "strict",
                "--stage",
                "fusion",
                "--run-root",
                str(run_root),
                "--run-id",
                "best-metric-run",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--best-metric",
                "val_acc",
                "--no-progress",
            ]
        )

    cfg = yaml.safe_load((run_root / "best-metric-run" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["best_metric"] == "val_acc"


def test_train_writes_fusion_hyperparams_to_config(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"

    class StopAfterLoaderInit(Exception):
        pass

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.random.default_rng(21).integers(0, 256, size=(6, 3, 28, 28), dtype=np.uint8),
            "input_ids": np.random.default_rng(22).integers(0, 1024, size=(6, 128), dtype=np.int32),
            "attention_mask": np.ones((6, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((6, 128), dtype=np.uint8),
            "y": np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
            "split": np.array(["train", "train", "train", "val", "val", "val"], dtype="U8"),
        },
    )

    def fake_dataloader(*args, **kwargs):
        raise StopAfterLoaderInit()

    monkeypatch.setattr(train_module, "DataLoader", fake_dataloader)

    with pytest.raises(StopAfterLoaderInit):
        train_main(
            [
                "--processed-root",
                str(tmp_path / "outputs" / "processed"),
                "--policy",
                "strict",
                "--stage",
                "fusion",
                "--run-root",
                str(run_root),
                "--run-id",
                "fusion-hparams-run",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--hidden-dim",
                "160",
                "--fusion-layers",
                "3",
                "--fusion-heads",
                "5",
                "--fusion-dropout",
                "0.2",
                "--fusion-mode",
                "residual_enhancer",
                "--text-shortcut-scale",
                "0.5",
                "--alpha",
                "0.25",
                "--beta",
                "0.15",
                "--val-fraction",
                "0.2",
                "--no-progress",
            ]
        )

    cfg = yaml.safe_load((run_root / "fusion-hparams-run" / "config.yaml").read_text(encoding="utf-8"))
    assert int(cfg["hidden_dim"]) == 160
    assert int(cfg["fusion_layers"]) == 3
    assert int(cfg["fusion_heads"]) == 5
    assert float(cfg["fusion_dropout"]) == pytest.approx(0.2)
    assert cfg["fusion_mode"] == "residual_enhancer"
    assert float(cfg["text_shortcut_scale"]) == pytest.approx(0.5)
    assert float(cfg["alpha"]) == pytest.approx(0.25)
    assert float(cfg["beta"]) == pytest.approx(0.15)
    assert float(cfg["val_fraction"]) == pytest.approx(0.2)


def test_evaluate_uses_fusion_mode_and_shortcut_scale_from_config(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "eval-fusion-mode-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "eval-fusion-mode-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "fusion_mode": "residual_enhancer",
                "text_shortcut_scale": 0.5,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((2, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((2, 128), dtype=np.int32),
            "attention_mask": np.ones((2, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((2, 128), dtype=np.uint8),
            "y": np.array([0, 1], dtype=np.int32),
            "split": np.array(["test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {"model_state": {}})
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.update(kwargs)

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, use_fusion=True):
            logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float32, device=rgb.device)
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(["--run-dir", str(run_dir), "--split", "test", "--device", "cpu"])
    assert code == 0
    assert captured["fusion_mode"] == "residual_enhancer"
    assert captured["text_shortcut_scale"] == pytest.approx(0.5)


def test_evaluate_loads_stage2_unified_model_from_config(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "eval-stage2-unified-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "eval-stage2-unified-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "model_type": "Stage2UnifiedClassifier",
                "dataset_name": "MTA",
                "dataset_vocab": {"MTA": 0},
                "output_dims": {"MTA": 2},
                "num_classes": 5,
                "hidden_dim": 8,
                "num_heads": 2,
                "trunk_layers": 1,
                "fusion_dropout": 0.0,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((2, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((2, 128), dtype=np.int32),
            "attention_mask": np.ones((2, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((2, 128), dtype=np.uint8),
            "y": np.array([0, 1], dtype=np.int32),
            "split": np.array(["test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(
        evaluate_module.torch,
        "load",
        lambda *args, **kwargs: {"model_state": {}, "decision_threshold": 0.8},
    )

    class ShouldNotBeCalledFusionModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError("MobileViTETBertFusionClassifier should not be used for Stage2UnifiedClassifier config")

    captured = {"dataset_name": None}

    class DummyStage2UnifiedModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, dataset_name, return_summary=False):
            captured["dataset_name"] = dataset_name
            logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float32, device=rgb.device)
            return {"logits": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", ShouldNotBeCalledFusionModel)
    monkeypatch.setattr(evaluate_module, "Stage2UnifiedClassifier", DummyStage2UnifiedModel, raising=False)

    code = eval_main(["--run-dir", str(run_dir), "--split", "test", "--device", "cpu"])
    assert code == 0
    assert captured["dataset_name"] == "MTA"
    assert (run_dir / "eval_test.json").exists()
    eval_payload = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    assert eval_payload["decision_threshold"] == pytest.approx(0.8)
    assert eval_payload["paper_precision"] is not None
    assert eval_payload["paper_recall"] is not None
    assert eval_payload["paper_f1"] is not None


def test_report_stage2_unified_prefers_eval_test_over_stacking_final(tmp_path: Path):
    run_dir = tmp_path / "runs" / "report-stage2-unified-run"
    _write_minimal_run_dir(run_dir, stage="fusion")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg["model_type"] = "Stage2UnifiedClassifier"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    (run_dir / "eval_test.json").write_text(
        json.dumps(
            {
                "top1": 0.77,
                "macro_precision": 0.76,
                "macro_f1": 0.75,
                "macro_recall": 0.74,
                "num_samples": 10,
                "split": "test",
            }
        ),
        encoding="utf-8",
    )
    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    (stack_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "top1": 0.99,
                "macro_f1": 0.98,
                "macro_recall": 0.97,
                "n_test_samples": 10,
                "metric_source": "stacking_final",
                "is_final_stage2_result": True,
            }
        ),
        encoding="utf-8",
    )

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: eval" in report_text
    assert "Top-1: 0.7700" in report_text
    assert "eval_test.json" in report_text


def test_report_stage2_unified_does_not_fall_back_to_stacking_without_eval_artifact(tmp_path: Path):
    run_dir = tmp_path / "runs" / "report-stage2-unified-no-eval-run"
    _write_minimal_run_dir(run_dir, stage="fusion")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg["model_type"] = "Stage2UnifiedClassifier"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    (stack_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "top1": 0.99,
                "macro_f1": 0.98,
                "macro_recall": 0.97,
                "n_test_samples": 10,
                "metric_source": "stacking_final",
                "is_final_stage2_result": True,
            }
        ),
        encoding="utf-8",
    )

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: stacking" not in report_text
    assert "Top-1:" not in report_text


def _early_stopping_dummy_payload() -> dict:
    return {
        "rgb": np.random.default_rng(31).integers(0, 256, size=(8, 3, 28, 28), dtype=np.uint8),
        "input_ids": np.random.default_rng(32).integers(0, 1024, size=(8, 128), dtype=np.int32),
        "attention_mask": np.ones((8, 128), dtype=np.uint8),
        "token_type_ids": np.zeros((8, 128), dtype=np.uint8),
        "y": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        "split": np.array(["train", "train", "train", "train", "val", "val", "val", "val"], dtype="U8"),
        "dataset": np.array(["MTA", "MTA", "MTA", "MTA", "MTA", "MTA", "MTA", "MTA"], dtype="U16"),
    }


class _DummyTrainModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, rgb, input_ids, attention_mask, token_type_ids, return_features=False, use_fusion=True):
        batch = rgb.shape[0]
        logits = torch.stack(
            [
                self.bias.expand(batch),
                (-self.bias).expand(batch),
            ],
            dim=1,
        )
        return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}


class _DummyStage2UnifiedTrainModel(torch.nn.Module):
    def __init__(self, *args, dataset_vocab: dict[str, int], output_dims: dict[str, int], **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.dataset_vocab = {str(key): int(value) for key, value in dataset_vocab.items()}
        self.output_dims = {str(key): int(value) for key, value in output_dims.items()}

    def forward(self, rgb, input_ids, attention_mask, token_type_ids, dataset_name, return_summary=False):
        batch = rgb.shape[0]
        num_classes = int(self.output_dims[str(dataset_name)])
        logits = torch.zeros((batch, num_classes), dtype=torch.float32, device=rgb.device)
        logits[:, 0] = self.bias
        if num_classes > 1:
            logits[:, 1] = -self.bias
        return {"logits": logits}


class _StopAfterTrainBootstrap(Exception):
    pass


def _patch_early_stopping_train_dependencies(monkeypatch: pytest.MonkeyPatch, metric_rows: list[tuple[float, float, float, float, float | None]]) -> None:
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module, "load_policy_multimodal_data", lambda *args, **kwargs: _early_stopping_dummy_payload())
    monkeypatch.setattr(train_module, "MobileViTETBertFusionClassifier", _DummyTrainModel)
    metric_iter = iter(metric_rows)
    monkeypatch.setattr(train_module, "_evaluate_loader", lambda *args, **kwargs: next(metric_iter))


def test_train_fusion_stage_can_resume_from_warmup_checkpoint(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module, "load_policy_multimodal_data", lambda *args, **kwargs: _early_stopping_dummy_payload())
    monkeypatch.setattr(train_module, "MobileViTETBertFusionClassifier", _DummyTrainModel)

    warmup_dir = run_root / "warmup-source"
    (warmup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": _DummyTrainModel().state_dict(),
            "config": {"stage": "warmup"},
            "epoch": 2,
        },
        warmup_dir / "checkpoints" / "best.ckpt",
    )

    def fake_evaluate_loader(*args, **kwargs):
        raise _StopAfterTrainBootstrap()

    monkeypatch.setattr(train_module, "_evaluate_loader", fake_evaluate_loader)

    with pytest.raises(_StopAfterTrainBootstrap):
        train_main(
            [
                "--processed-root",
                str(tmp_path / "outputs" / "processed"),
                "--policy",
                "strict",
                "--stage",
                "fusion",
                "--run-root",
                str(run_root),
                "--run-id",
                "fusion-resume-run",
                "--epochs",
                "3",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--no-progress",
                "--warmup-checkpoint",
                str(warmup_dir / "checkpoints" / "best.ckpt"),
            ]
        )

    cfg = yaml.safe_load((run_root / "fusion-resume-run" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["stage"] == "fusion"
    assert cfg["warmup_checkpoint"] == str(warmup_dir / "checkpoints" / "best.ckpt")


def test_train_stage2_unified_entry_instantiates_unified_classifier_not_fusion(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.80, 0.78, 0.55, None),
        ],
    )
    captured: dict[str, object] = {}

    class ShouldNotBeCalledFusionModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError("MobileViTETBertFusionClassifier should not be used for --model-type stage2_unified")

    class CapturingStage2UnifiedModel(_DummyStage2UnifiedTrainModel):
        def __init__(self, *args, **kwargs):
            captured["init_kwargs"] = dict(kwargs)
            super().__init__(*args, **kwargs)

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, dataset_name, return_summary=False):
            captured["dataset_name"] = dataset_name
            return super().forward(
                rgb,
                input_ids,
                attention_mask,
                token_type_ids,
                dataset_name=dataset_name,
                return_summary=return_summary,
            )

    monkeypatch.setattr(train_module, "MobileViTETBertFusionClassifier", ShouldNotBeCalledFusionModel)
    monkeypatch.setattr(train_module, "Stage2UnifiedClassifier", CapturingStage2UnifiedModel, raising=False)

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--model-type",
            "stage2_unified",
            "--datasets",
            "MTA",
            "--num-classes",
            "2",
            "--run-root",
            str(run_root),
            "--run-id",
            "stage2-unified-select-run",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )

    assert code == 0
    assert captured["dataset_name"] == "MTA"
    assert captured["init_kwargs"]["dataset_vocab"] == {"MTA": 0}
    assert captured["init_kwargs"]["output_dims"] == {"MTA": 2}


def test_train_stage2_unified_smoke_writes_unified_config_and_artifacts(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed_for_dataset(processed_root, dataset="MTA", num_classes=2)

    code = train_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--model-type",
            "stage2_unified",
            "--datasets",
            "MTA",
            "--num-classes",
            "2",
            "--run-root",
            str(run_root),
            "--run-id",
            "stage2-unified-smoke-run",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "16",
            "--fusion-heads",
            "2",
            "--fusion-layers",
            "1",
            "--fusion-dropout",
            "0.0",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )

    assert code == 0
    run_dir = run_root / "stage2-unified-smoke-run"
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "checkpoints" / "best.ckpt").exists()

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "Stage2UnifiedClassifier"
    assert cfg["dataset_name"] == "MTA"
    assert cfg["dataset_vocab"] == {"MTA": 0}
    assert cfg["output_dims"] == {"MTA": 2}

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["epoch"].tolist() == [1]


def test_train_checkpoint_selection_prefers_stable_threshold_when_metrics_tie(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.95, 0.94, 0.55, 0.50),
            (0.8, 0.95, 0.94, 0.54, 0.95),
            (0.8, 0.94, 0.93, 0.53, 0.96),
        ],
    )

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "stable-threshold-run",
            "--epochs",
            "3",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--best-metric",
            "val_acc",
            "--checkpoint-selection",
            "score_optimized",
            "--no-progress",
        ]
    )
    assert code == 0

    best_ckpt = torch.load(run_root / "stable-threshold-run" / "checkpoints" / "best.ckpt", map_location="cpu")
    assert best_ckpt["epoch"] == 1
    assert best_ckpt["decision_threshold"] == pytest.approx(0.50)


def test_train_writes_class_weight_mode_and_scheduler_to_config(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"

    class StopAfterLoaderInit(Exception):
        pass

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module, "load_policy_multimodal_data", lambda *args, **kwargs: _early_stopping_dummy_payload())

    def fake_dataloader(*args, **kwargs):
        raise StopAfterLoaderInit()

    monkeypatch.setattr(train_module, "DataLoader", fake_dataloader)

    with pytest.raises(StopAfterLoaderInit):
        train_main(
            [
                "--processed-root",
                str(tmp_path / "outputs" / "processed"),
                "--policy",
                "strict",
                "--stage",
                "fusion",
                "--run-root",
                str(run_root),
                "--run-id",
                "class-weight-run",
                "--class-weight-mode",
                "balanced",
                "--scheduler",
                "cosine",
                "--no-progress",
            ]
        )

    cfg = yaml.safe_load((run_root / "class-weight-run" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["class_weight_mode"] == "balanced"
    assert cfg["scheduler"] == "cosine"


def test_train_can_freeze_image_backbone_for_initial_epochs(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.80, 0.78, 0.55, None),
        ],
    )

    class FreezeAwareModel(_DummyTrainModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_backbone = torch.nn.Linear(1, 1)
            self.text_backbone = torch.nn.Linear(1, 1)

    monkeypatch.setattr(train_module, "MobileViTETBertFusionClassifier", FreezeAwareModel)

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "freeze-image-run",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--freeze-image-backbone-epochs",
            "1",
            "--no-progress",
        ]
    )
    assert code == 0
    cfg = yaml.safe_load((run_root / "freeze-image-run" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["freeze_image_backbone_epochs"] == 1


def test_train_early_stopping_triggers_and_records_artifacts(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.80, 0.78, 0.55, 0.61),
            (0.8, 0.79, 0.77, 0.54, 0.61),
            (0.8, 0.78, 0.76, 0.53, 0.61),
        ],
    )

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "early-stop-run",
            "--epochs",
            "8",
            "--batch-size",
            "2",
            "--best-metric",
            "val_acc",
            "--early-stopping-patience",
            "2",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    assert code == 0

    run_dir = run_root / "early-stop-run"
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["epoch"].tolist() == [1, 2, 3]
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["early_stopping_patience"] == 2
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "early_stopping_triggered" in train_log
    best_ckpt = torch.load(run_dir / "checkpoints" / "best.ckpt", map_location="cpu")
    assert best_ckpt["epoch"] == 1
    assert best_ckpt["best_metric_value"] == pytest.approx(0.80)


def test_train_early_stopping_respects_best_metric(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.90, 0.40, 0.55, None),
            (0.8, 0.85, 0.60, 0.54, None),
            (0.8, 0.84, 0.59, 0.53, None),
        ],
    )

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "best-metric-early-stop-run",
            "--epochs",
            "6",
            "--batch-size",
            "2",
            "--best-metric",
            "val_macro_f1",
            "--early-stopping-patience",
            "1",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    assert code == 0

    best_ckpt = torch.load(run_root / "best-metric-early-stop-run" / "checkpoints" / "best.ckpt", map_location="cpu")
    assert best_ckpt["epoch"] == 2
    assert best_ckpt["best_metric"] == "val_macro_f1"
    assert best_ckpt["best_metric_value"] == pytest.approx(0.60)


def test_train_early_stopping_tie_does_not_reset_patience(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.80, 0.50, 0.55, None),
            (0.8, 0.80, 0.49, 0.54, None),
            (0.8, 0.79, 0.48, 0.53, None),
        ],
    )

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "tie-early-stop-run",
            "--epochs",
            "6",
            "--batch-size",
            "2",
            "--best-metric",
            "val_acc",
            "--early-stopping-patience",
            "1",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    assert code == 0

    metrics = pd.read_csv(run_root / "tie-early-stop-run" / "metrics.csv")
    assert metrics["epoch"].tolist() == [1, 2]


def test_train_early_stopping_disabled_by_default(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "runs"
    _patch_early_stopping_train_dependencies(
        monkeypatch,
        [
            (0.8, 0.80, 0.78, 0.55, None),
            (0.8, 0.79, 0.77, 0.54, None),
            (0.8, 0.78, 0.76, 0.53, None),
        ],
    )

    code = train_main(
        [
            "--processed-root",
            str(tmp_path / "outputs" / "processed"),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "early-stop-disabled-run",
            "--epochs",
            "3",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    assert code == 0

    run_dir = run_root / "early-stop-disabled-run"
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["epoch"].tolist() == [1, 2, 3]
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["early_stopping_patience"] == 0
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "early_stopping_triggered" not in train_log


def test_choose_best_binary_threshold_maximizes_accuracy():
    positive_probs = np.array([0.20, 0.49, 0.52, 0.90], dtype=np.float32)
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)

    threshold, best_acc = train_module.choose_best_binary_threshold(
        positive_probs=positive_probs,
        y_true=y_true,
    )

    pred = (positive_probs >= threshold).astype(np.int32)
    assert best_acc == pytest.approx(1.0)
    assert float(np.mean(pred == y_true)) == pytest.approx(1.0)


def test_evaluate_uses_binary_decision_threshold(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "threshold-eval-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "threshold-eval-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "device_requested": "cpu",
                "device": "cpu",
                "decision_threshold": 0.8,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((2, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((2, 128), dtype=np.int32),
            "attention_mask": np.ones((2, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((2, 128), dtype=np.uint8),
            "y": np.array([0, 1], dtype=np.int32),
            "split": np.array(["test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(
        evaluate_module.torch,
        "load",
        lambda *args, **kwargs: {"model_state": {}, "decision_threshold": 0.8},
    )

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, use_fusion=True):
            logits = torch.tensor(
                [
                    [0.0, 0.8],
                    [0.0, 2.0],
                ],
                dtype=torch.float32,
                device=rgb.device,
            )
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(["--run-dir", str(run_dir), "--split", "test", "--device", "cpu"])
    assert code == 0

    payload = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    assert payload["decision_threshold"] == pytest.approx(0.8)
    assert payload["top1"] == pytest.approx(1.0)


def test_evaluate_warmup_disables_fusion_path(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "warmup-eval-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "warmup-eval-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "warmup",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((2, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((2, 128), dtype=np.int32),
            "attention_mask": np.ones((2, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((2, 128), dtype=np.uint8),
            "y": np.array([0, 1], dtype=np.int32),
            "split": np.array(["test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {"model_state": {}})

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, return_features=False, use_fusion=True):
            assert use_fusion is False
            logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float32, device=rgb.device)
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(["--run-dir", str(run_dir), "--split", "test", "--device", "cpu"])
    assert code == 0


def test_evaluate_batches_forward_pass_to_avoid_full_split_oom(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "batched-eval-run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "batched-eval-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "stage": "fusion",
                "num_classes": 2,
                "hidden_dim": 8,
                "vocab_size": 32,
                "device_requested": "cpu",
                "device": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.zeros((5, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((5, 128), dtype=np.int32),
            "attention_mask": np.ones((5, 128), dtype=np.uint8),
            "token_type_ids": np.zeros((5, 128), dtype=np.uint8),
            "y": np.array([0, 1, 0, 1, 0], dtype=np.int32),
            "split": np.array(["test", "test", "test", "test", "test"], dtype="U8"),
        },
    )
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {"model_state": {}})

    batch_sizes = []

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def load_state_dict(self, state):
            return

        def forward(self, rgb, input_ids, attention_mask, token_type_ids, return_features=False, use_fusion=True):
            batch_sizes.append(int(rgb.shape[0]))
            logits = torch.zeros((rgb.shape[0], 2), dtype=torch.float32, device=rgb.device)
            logits[:, 0] = 3.0
            logits[:, 1] = 1.0
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits}

    monkeypatch.setattr(evaluate_module, "MobileViTETBertFusionClassifier", DummyModel)

    code = eval_main(
        [
            "--run-dir",
            str(run_dir),
            "--split",
            "test",
            "--device",
            "cpu",
            "--eval-batch-size",
            "2",
        ]
    )
    assert code == 0

    payload = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    assert batch_sizes == [2, 2, 1]
    assert payload["num_samples"] == 5
