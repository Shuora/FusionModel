from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.experiments.stage1_binary as stage1_module
from src.experiments.stage1_binary import main as stage1_main
from src.experiments.stage2_multiclass import main as stage2_main


def _single_date_partition(run_root: Path) -> Path:
    date_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(date_dirs) == 1
    date_dir = date_dirs[0]
    assert len(date_dir.name.split("-")) == 3
    return date_dir


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

    date_dir = _single_date_partition(run_root)
    run_dir = date_dir / "stage1-run"
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

    run_dir = _single_date_partition(run_root) / "stage1-stacking-run"
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

    run_dir = _single_date_partition(run_root) / "stage1-moe-run"
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

    date_dir = _single_date_partition(run_root)
    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = date_dir / f"stage2-{dataset.lower()}"
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "eval_test.json").exists()
        assert (run_dir / "report.md").exists()

    for limit in (4000, 3000, 2000):
        run_dir = date_dir / f"stage2-ustc-tfc2016-train{limit}"
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "eval_test.json").exists()
        assert (run_dir / "report.md").exists()

    summary_path = date_dir / "stage2_execution_summary.json"
    assert summary_path.exists()
    summary = pd.read_json(summary_path)
    assert set(summary.columns) >= {"dataset", "run_id", "run_dir", "run_date", "code"}
    assert set(summary["run_date"].tolist()) == {date_dir.name}
    assert str(date_dir / "stage2-mta") in set(summary["run_dir"].tolist())


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

    date_dir = _single_date_partition(run_root)
    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = date_dir / f"stage2-{dataset.lower()}"
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

    date_dir = _single_date_partition(run_root)
    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = date_dir / f"stage2-{dataset.lower()}"
        assert (run_dir / "moe" / "moe_metrics.json").exists()
        assert not (run_dir / "eval_test.json").exists()
        report_text = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "Metric Source: moe" in report_text


def test_stage2_fusion_then_stacking_requires_level2_meta_artifacts_and_shared_run_family(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"

    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 2, 0, 1, 2]), ["D", "E", "F"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 2, 0, 1, 2]), ["A", "B", "C"])
    _write_processed_dataset(processed_root, "USTC-TFC2016", policy, np.array([0, 1, 2, 0, 1, 2]), ["U1", "U2", "U3"])

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    try:
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
                "fusion",
                "--meta-classifier",
                "stacking",
                "--skip-ustc-limited",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--num-workers",
                "0",
            ]
        )
    except SystemExit as exc:
        # argparse uses SystemExit for unknown flags; surface that as a test failure (RED) until CLI exists.
        code = int(getattr(exc, "code", 1) or 1)

    assert code == 0

    date_dir = _single_date_partition(run_root)
    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = date_dir / f"stage2-{dataset.lower()}"
        # Level 1 fusion must run first under the dataset run family.
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "eval_test.json").exists()
        assert (run_dir / "report.md").exists()

        # Level 2 meta artifacts must be generated under the same run family.
        meta_dir = run_dir / "meta_features"
        assert meta_dir.exists()
        assert (meta_dir / "oof_meta_train.npz").exists()
        assert (meta_dir / "meta_test.npz").exists()

        # Level 2 stacking must run second and persist its outputs under the same run family.
        assert (run_dir / "stacking" / "meta_metrics.json").exists()
        assert (run_dir / "stacking" / "meta_model.json").exists()


def test_stage2_execution_summary_records_level1_run_dir_and_final_metric_source(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"

    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 2, 0, 1, 2]), ["D", "E", "F"])
    _write_processed_dataset(processed_root, "MFCP", policy, np.array([0, 1, 2, 0, 1, 2]), ["A", "B", "C"])
    _write_processed_dataset(processed_root, "USTC-TFC2016", policy, np.array([0, 1, 2, 0, 1, 2]), ["U1", "U2", "U3"])

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    try:
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
                "fusion",
                "--meta-classifier",
                "stacking",
                "--skip-ustc-limited",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--num-workers",
                "0",
            ]
        )
    except SystemExit as exc:
        code = int(getattr(exc, "code", 1) or 1)

    assert code == 0

    date_dir = _single_date_partition(run_root)
    summary_path = date_dir / "stage2_execution_summary.json"
    assert summary_path.exists()
    summary = pd.read_json(summary_path)
    assert "level1_run_dir" in summary.columns
    assert "final_metric_source" in summary.columns
    assert set(summary["run_date"].tolist()) == {date_dir.name}

    for dataset in ("MTA", "MFCP", "USTC-TFC2016"):
        run_dir = date_dir / f"stage2-{dataset.lower()}"
        rows = summary.loc[summary["dataset"] == dataset]
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["level1_run_dir"] == str(run_dir)
        # Contract: final_metric_source is semantic (e.g., "stacking"), not a concrete file path.
        assert row["final_metric_source"] == "stacking"


def test_stage2_runner_meta_classifier_stacking_wires_to_stacking_main(tmp_path: Path, monkeypatch):
    """RED (Task 4): stage2 runner must support --meta-classifier stacking after a fusion run.

    This test is intentionally unit-ish: it avoids training by stubbing stage2 task execution, but
    still locks the runner-level wiring contract for the meta-classifier stage.
    """

    from src.experiments import stage2_multiclass as stage2_mod

    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    policy = "session_full"

    _write_processed_dataset(processed_root, "MTA", policy, np.array([0, 1, 2, 0, 1, 2]), ["D", "E", "F"])

    monkeypatch.setattr(stage2_mod, "build_stage2_tasks", lambda: [{"dataset": "MTA", "num_classes": 7}])

    def fake_run_stage2_task(*, dated_run_root: Path, dataset: str, **kwargs) -> int:
        run_dir = dated_run_root / f"stage2-{dataset.lower()}"
        (run_dir / "meta_features").mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(stage2_mod, "_run_stage2_task", fake_run_stage2_task)

    captured = {"stacking_calls": []}

    import src.stacking as stacking_module

    def fake_stacking_main(argv):
        captured["stacking_calls"].append(list(argv))
        return 0

    monkeypatch.setattr(stacking_module, "main", fake_stacking_main)

    out_tasks = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"
    try:
        code = stage2_mod.main(
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
                "fusion",
                "--meta-classifier",
                "stacking",
                "--skip-ustc-limited",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--num-workers",
                "0",
            ]
        )
    except SystemExit as exc:
        # argparse uses SystemExit for unknown flags; surface that as a test failure (RED) until CLI exists.
        code = int(getattr(exc, "code", 1) or 1)

    assert code == 0
    assert len(captured["stacking_calls"]) == 1
    argv = captured["stacking_calls"][0]
    assert "--run-dir" in argv
    assert "--meta-artifacts-dir" in argv



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


def test_stage1_binary_execute_forwards_early_stopping_patience_to_train(tmp_path: Path, monkeypatch):
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
            "--early-stopping-patience",
            "3",
        ]
    )
    assert code == 0
    assert "--early-stopping-patience" in captured["argv"]
    assert "3" in captured["argv"]


def test_stage1_binary_execute_score_optimized_skips_test_holdout_until_explicit_request(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    captured = {"report": 0}

    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [{"session_id": "s1", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "train"}]
        ),
    )
    monkeypatch.setattr(stage1_mod, "train_main", lambda argv: 0)
    monkeypatch.setattr(stage1_mod, "evaluate_main", lambda argv: (_ for _ in ()).throw(AssertionError("holdout test should be skipped")))
    monkeypatch.setattr(stage1_mod, "report_main", lambda argv: captured.__setitem__("report", captured["report"] + 1) or 0)

    code = stage1_mod.main(
        [
            "--processed-root",
            str(tmp_path / "processed"),
            "--execute",
            "--protocol-mode",
            "score_optimized",
            "--holdout-eval",
            "final_only",
        ]
    )
    assert code == 0
    assert captured["report"] == 1


def test_stage1_binary_execute_score_optimized_forwards_checkpoint_selection(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    captured = {}

    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [
                {"session_id": "s_train", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "train"},
                {"session_id": "s_val", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "val"},
                {"session_id": "s_test", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "test"},
            ]
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
            "--protocol-mode",
            "score_optimized",
        ]
    )
    assert code == 0
    assert "--checkpoint-selection" in captured["argv"]
    idx = captured["argv"].index("--checkpoint-selection")
    assert captured["argv"][idx + 1] == "score_optimized"


def test_stage1_binary_execute_score_optimized_can_run_warmup_then_fusion(tmp_path: Path, monkeypatch):
    from src.experiments import stage1_binary as stage1_mod

    captured = []

    monkeypatch.setattr(
        stage1_mod,
        "build_stage1_manifest",
        lambda processed_root, policy, protocol_mode="paper_balanced": pd.DataFrame(
            [
                {"session_id": "s_train", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "train"},
                {"session_id": "s_val", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "val"},
                {"session_id": "s_test", "dataset": "ISCX", "dataset_raw": "ISCX", "label_binary": 0, "label_text": "normal", "split": "test"},
            ]
        ),
    )

    def fake_train_main(argv):
        captured.append(list(argv))
        return 0

    monkeypatch.setattr(stage1_mod, "train_main", fake_train_main)
    monkeypatch.setattr(stage1_mod, "_run_stage_report", lambda *args, **kwargs: 0)

    code = stage1_mod.main(
        [
            "--processed-root",
            str(tmp_path / "processed"),
            "--execute",
            "--protocol-mode",
            "score_optimized",
            "--two-stage",
            "--warmup-epochs",
            "2",
        ]
    )
    assert code == 0
    assert len(captured) == 2

    warmup_argv, fusion_argv = captured
    assert "--stage" in warmup_argv and warmup_argv[warmup_argv.index("--stage") + 1] == "warmup"
    assert "--epochs" in warmup_argv and warmup_argv[warmup_argv.index("--epochs") + 1] == "2"
    assert "--holdout-eval" not in warmup_argv

    assert "--stage" in fusion_argv and fusion_argv[fusion_argv.index("--stage") + 1] == "fusion"
    assert "--warmup-checkpoint" in fusion_argv
    assert "--fusion-mode" in fusion_argv and fusion_argv[fusion_argv.index("--fusion-mode") + 1] == "residual_enhancer"
    assert "--text-shortcut-scale" in fusion_argv and fusion_argv[fusion_argv.index("--text-shortcut-scale") + 1] == "0.5"


def test_stage1_binary_execute_forwards_latest_fusion_train_args(tmp_path: Path, monkeypatch):
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
            "--hidden-dim",
            "160",
            "--fusion-layers",
            "3",
            "--fusion-heads",
            "5",
            "--fusion-dropout",
            "0.2",
            "--alpha",
            "0.25",
            "--beta",
            "0.15",
            "--val-fraction",
            "0.2",
            "--train-max-samples",
            "123",
        ]
    )
    assert code == 0
    argv = captured["argv"]
    assert "--hidden-dim" in argv and "160" in argv
    assert "--fusion-layers" in argv and "3" in argv
    assert "--fusion-heads" in argv and "5" in argv
    assert "--fusion-dropout" in argv and "0.2" in argv
    assert "--alpha" in argv and "0.25" in argv
    assert "--beta" in argv and "0.15" in argv
    assert "--val-fraction" in argv and "0.2" in argv
    assert "--train-max-samples" in argv and "123" in argv
    assert "--datasets" in argv
    assert "--session-filter-manifest" in argv
    assert "--label-mode" in argv and "binary" in argv
    assert "--num-classes" in argv and "2" in argv


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


def test_stage2_multiclass_execute_forwards_latest_fusion_train_args(tmp_path: Path, monkeypatch):
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
            "--hidden-dim",
            "160",
            "--fusion-layers",
            "3",
            "--fusion-heads",
            "5",
            "--fusion-dropout",
            "0.2",
            "--alpha",
            "0.25",
            "--beta",
            "0.15",
            "--val-fraction",
            "0.2",
            "--best-metric",
            "val_acc",
        ]
    )
    assert code == 0
    assert captured
    for argv in captured:
        assert "--hidden-dim" in argv
        assert "160" in argv
        assert "--fusion-layers" in argv
        assert "3" in argv
        assert "--fusion-heads" in argv
        assert "5" in argv
        assert "--fusion-dropout" in argv
        assert "0.2" in argv
        assert "--alpha" in argv
        assert "0.25" in argv
        assert "--beta" in argv
        assert "0.15" in argv
        assert "--val-fraction" in argv
        assert "0.2" in argv
        assert "--best-metric" in argv
        assert "val_acc" in argv
