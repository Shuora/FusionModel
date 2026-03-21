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
    assert "gate_mean" in train_log
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

        def forward(self, rgb, input_ids, attention_mask, token_type_ids):
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
            gate = torch.full((rgb.shape[0], 1), 0.5, dtype=torch.float32, device=rgb.device)
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits, "gate": gate}

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

        def forward(self, rgb, input_ids, attention_mask, token_type_ids):
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
            gate = torch.full((rgb.shape[0], 1), 0.5, dtype=torch.float32, device=rgb.device)
            return {"logits_fuse": logits, "logits_img": logits, "logits_tls": logits, "gate": gate}

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
