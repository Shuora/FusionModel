from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.experiments.stage1_binary as stage1_module
from src.experiments.stage1_binary import main as stage1_main
from src.experiments.stage2_multiclass import main as stage2_main


def _write_processed_dataset(
    root: Path,
    dataset: str,
    policy: str,
    labels: np.ndarray,
    families: list[str],
) -> None:
    rgb_dir = root / dataset / policy / "rgb"
    etbert_dir = root / dataset / policy / "etbert"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    etbert_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = int(labels.shape[0])
    session_ids = np.array([f"{dataset.lower()}_{i}" for i in range(n)], dtype="U64")
    rgbs = np.random.default_rng(101).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(202).integers(0, 1024, size=(n, 128), dtype=np.int32)
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

    splits = ["train", "train", "val", "val", "test", "test"][:n]
    rows = []
    for i in range(n):
        rows.append(
            {
                "session_id": session_ids[i],
                "dataset": dataset,
                "family": families[i % len(families)],
                "capture_id": f"cap_{i}.pcap",
                "split": splits[i],
                "policy": policy,
            }
        )
    with (manifest_dir / "session_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _patch_minimal_stage1_execute_specs(monkeypatch) -> None:
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_ISCX_SPECS",
        [{"name": "toy_iscx", "capture_prefixes": ("cap_",), "train": 1, "test": 1}],
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MTA_SPECS",
        [{"family": "D", "train": 1, "test": 1}],
    )
    monkeypatch.setattr(
        stage1_module,
        "PAPER_STAGE1_MFCP_SPECS",
        [{"family": "A", "train": 1, "test": 1}],
    )


def test_stage1_binary_execute_runs_train_eval_report(tmp_path: Path, monkeypatch):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"
    _patch_minimal_stage1_execute_specs(monkeypatch)

    _write_processed_dataset(processed_root, "ISCX", policy, np.array([0, 1, 0, 1, 0, 1]), ["Chat", "VoIP"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 0, 1, 0, 1]), ["A", "B"])
    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 0, 1, 0, 1]), ["D", "E"])
    _write_processed_dataset(
        processed_root, "USTC-TFC2016", policy, np.array([0, 1, 0, 1, 0, 1]), ["U1", "U2"]
    )

    out_manifest = tmp_path / "outputs" / "protocol" / "stage1_binary_manifest.csv"
    code = stage1_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--output",
            str(out_manifest),
            "--execute",
            "--run-root",
            str(run_root),
            "--run-id",
            "stage1-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ]
    )
    assert code == 0

    run_dir = run_root / "stage1-run"
    assert out_manifest.exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "eval_test.json").exists()
    assert (run_dir / "report.md").exists()

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["label_mode"] == "binary"
    assert int(cfg["num_classes"]) == 2
    assert str(cfg["session_filter_manifest"]).endswith("stage1_binary_manifest.csv")


def test_stage1_binary_execute_stacking_reports_stacking_metrics(tmp_path: Path, monkeypatch):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"
    _patch_minimal_stage1_execute_specs(monkeypatch)

    _write_processed_dataset(processed_root, "ISCX", policy, np.array([0, 1, 0, 1, 0, 1]), ["Chat", "VoIP"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 0, 1, 0, 1]), ["A", "B"])
    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 0, 1, 0, 1]), ["D", "E"])

    out_manifest = tmp_path / "outputs" / "protocol" / "stage1_binary_manifest.csv"
    code = stage1_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--output",
            str(out_manifest),
            "--execute",
            "--run-root",
            str(run_root),
            "--run-id",
            "stage1-stacking-run",
            "--stage",
            "stacking",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ]
    )
    assert code == 0

    run_dir = run_root / "stage1-stacking-run"
    assert (run_dir / "stacking" / "meta_metrics.json").exists()
    assert not (run_dir / "eval_test.json").exists()
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: stacking" in report_text


def test_stage1_binary_execute_moe_reports_moe_metrics(tmp_path: Path, monkeypatch):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"
    _patch_minimal_stage1_execute_specs(monkeypatch)

    _write_processed_dataset(processed_root, "ISCX", policy, np.array([0, 1, 0, 1, 0, 1]), ["Chat", "VoIP"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 0, 1, 0, 1]), ["A", "B"])
    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 0, 1, 0, 1]), ["D", "E"])

    out_manifest = tmp_path / "outputs" / "protocol" / "stage1_binary_manifest.csv"
    code = stage1_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--output",
            str(out_manifest),
            "--execute",
            "--run-root",
            str(run_root),
            "--run-id",
            "stage1-moe-run",
            "--stage",
            "moe",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ]
    )
    assert code == 0

    run_dir = run_root / "stage1-moe-run"
    assert (run_dir / "moe" / "moe_metrics.json").exists()
    assert not (run_dir / "eval_test.json").exists()
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: moe" in report_text


def test_stage2_multiclass_execute_runs_three_dataset_jobs(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"

    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 2, 0, 1, 2]), ["D", "E", "F"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 2, 0, 1, 2]), ["A", "B", "C"])
    _write_processed_dataset(
        processed_root,
        "USTC-TFC2016",
        policy,
        np.array([0, 1, 2, 0, 1, 2]),
        ["U1", "U2", "U3"],
    )

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    code = stage2_main(
        [
            "--output",
            str(out_tasks),
            "--execute",
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--run-root",
            str(run_root),
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0
    assert out_tasks.exists()

    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = run_root / f"stage2-{dataset.lower()}"
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "eval_test.json").exists()
        assert (run_dir / "report.md").exists()

    for limit in (4000, 3000, 2000):
        run_dir = run_root / f"stage2-ustc-tfc2016-train{limit}"
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "eval_test.json").exists()
        assert (run_dir / "report.md").exists()


def test_stage2_multiclass_execute_stacking_reports_stage_metrics(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"

    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 2, 0, 1, 2]), ["D", "E", "F"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 2, 0, 1, 2]), ["A", "B", "C"])
    _write_processed_dataset(processed_root, "USTC-TFC2016", policy, np.array([0, 1, 2, 0, 1, 2]), ["U1", "U2", "U3"])

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    code = stage2_main(
        [
            "--output",
            str(out_tasks),
            "--execute",
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--run-root",
            str(run_root),
            "--stage",
            "stacking",
            "--skip-ustc-limited",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = run_root / f"stage2-{dataset.lower()}"
        assert (run_dir / "stacking" / "meta_metrics.json").exists()
        assert not (run_dir / "eval_test.json").exists()
        report_text = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "Metric Source: stacking" in report_text


def test_stage2_multiclass_execute_moe_reports_stage_metrics(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"

    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 2, 0, 1, 2]), ["D", "E", "F"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 2, 0, 1, 2]), ["A", "B", "C"])
    _write_processed_dataset(processed_root, "USTC-TFC2016", policy, np.array([0, 1, 2, 0, 1, 2]), ["U1", "U2", "U3"])

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    code = stage2_main(
        [
            "--output",
            str(out_tasks),
            "--execute",
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--run-root",
            str(run_root),
            "--stage",
            "moe",
            "--skip-ustc-limited",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = run_root / f"stage2-{dataset.lower()}"
        assert (run_dir / "moe" / "moe_metrics.json").exists()
        assert not (run_dir / "eval_test.json").exists()
        report_text = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "Metric Source: moe" in report_text



def test_stage1_binary_default_output_path_matches_docs(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [{"dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal"}]
        ),
    )

    code = stage1_mod.main(["--processed-root", str(tmp_path / "processed")])
    assert code == 0
    assert (tmp_path / "outputs" / "protocol" / "stage1_binary_manifest.csv").exists()


def test_stage2_multiclass_default_output_path_matches_docs(tmp_path: Path, monkeypatch):
    from src.experiments import stage2_multiclass as stage2_mod

    monkeypatch.chdir(tmp_path)
    code = stage2_mod.main([])
    assert code == 0
    assert (tmp_path / "outputs" / "protocol" / "stage2_tasks.json").exists()


def test_stage1_binary_execute_forwards_device_and_num_workers_to_train(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    captured = {}

    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [{"session_id": "s1", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal"}]
        ),
    )

    def fake_train_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(stage1_mod, "train_main", fake_train_main)
    monkeypatch.setattr(stage1_mod, "_run_stage_report", lambda run_dir, stage, device: 0)

    code = stage1_mod.main(
        [
            "--processed-root",
            str(tmp_path / "processed"),
            "--execute",
            "--device",
            "cuda",
            "--num-workers",
            "2",
        ]
    )
    assert code == 0
    assert "--device" in captured["argv"]
    assert "cuda" in captured["argv"]
    assert "--num-workers" in captured["argv"]
    assert "2" in captured["argv"]


def test_stage1_binary_execute_forwards_best_metric_to_train(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    captured = {}

    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [{"session_id": "s1", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal"}]
        ),
    )

    def fake_train_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(stage1_mod, "train_main", fake_train_main)
    monkeypatch.setattr(stage1_mod, "_run_stage_report", lambda run_dir, stage, device: 0)

    code = stage1_mod.main(
        [
            "--processed-root",
            str(tmp_path / "processed"),
            "--execute",
            "--best-metric",
            "val_acc",
        ]
    )
    assert code == 0
    assert "--best-metric" in captured["argv"]
    assert "val_acc" in captured["argv"]


def test_stage1_binary_execute_defaults_num_workers_to_four(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    captured = {}

    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [{"session_id": "s1", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal"}]
        ),
    )

    def fake_train_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(stage1_mod, "train_main", fake_train_main)
    monkeypatch.setattr(stage1_mod, "_run_stage_report", lambda run_dir, stage, device: 0)

    code = stage1_mod.main(
        [
            "--processed-root",
            str(tmp_path / "processed"),
            "--execute",
        ]
    )
    assert code == 0
    assert "--num-workers" in captured["argv"]
    assert "4" in captured["argv"]


def test_stage2_multiclass_execute_forwards_device_and_num_workers_to_train(tmp_path: Path, monkeypatch):
    from src.experiments import stage2_multiclass as stage2_mod

    captured = []

    def fake_train_main(argv):
        captured.append(list(argv))
        return 0

    monkeypatch.setattr(stage2_mod, "train_main", fake_train_main)
    monkeypatch.setattr(stage2_mod, "_run_stage_report", lambda run_dir, stage, device: 0)

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    code = stage2_mod.main(
        [
            "--output",
            str(out_tasks),
            "--execute",
            "--processed-root",
            str(tmp_path / "processed"),
            "--skip-ustc-limited",
            "--device",
            "auto",
            "--num-workers",
            "1",
        ]
    )
    assert code == 0
    assert captured
    for argv in captured:
        assert "--device" in argv
        assert "auto" in argv
        assert "--num-workers" in argv
        assert "1" in argv


def test_stage2_multiclass_execute_defaults_num_workers_to_four(tmp_path: Path, monkeypatch):
    from src.experiments import stage2_multiclass as stage2_mod

    captured = []

    def fake_train_main(argv):
        captured.append(list(argv))
        return 0

    monkeypatch.setattr(stage2_mod, "train_main", fake_train_main)
    monkeypatch.setattr(stage2_mod, "_run_stage_report", lambda run_dir, stage, device: 0)

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    code = stage2_mod.main(
        [
            "--output",
            str(out_tasks),
            "--execute",
            "--processed-root",
            str(tmp_path / "processed"),
            "--skip-ustc-limited",
        ]
    )
    assert code == 0
    assert captured
    for argv in captured:
        assert "--num-workers" in argv
        assert "4" in argv
