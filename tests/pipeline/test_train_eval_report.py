from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import main as eval_main
from src.report import main as report_main
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

    session_ids = np.array([f"s{i}" for i in range(1, 9)], dtype="U64")
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    rgbs = np.random.default_rng(42).integers(0, 256, size=(8, 3, 28, 28), dtype=np.uint8)
    token_ids = np.random.default_rng(43).integers(0, 1024, size=(8, 256), dtype=np.int32)
    attention = np.ones((8, 256), dtype=np.uint8)
    segments = np.zeros((8, 256), dtype=np.uint8)

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
    pd.DataFrame(rows).to_csv(manifest_dir / "session_manifest.csv", index=False)


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
    assert "model_type: TinyFusionClassifier" in cfg_text

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
