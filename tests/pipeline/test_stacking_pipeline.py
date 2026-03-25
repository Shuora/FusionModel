from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

import src.stacking as stacking_module
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

    oof = np.load(stack_dir / "oof_meta_train.npz", allow_pickle=True)
    test_meta = np.load(stack_dir / "meta_test.npz", allow_pickle=True)
    for arr in (oof, test_meta):
        assert "X" in arr.files
        assert "y" in arr.files
        assert "feature_names" in arr.files
        assert "feature_schema" in arr.files
        assert "feature_schema_version" in arr.files
        assert arr["X"].ndim == 2
        assert arr["X"].shape[1] == len(arr["feature_names"])
        assert arr["feature_schema_version"].item() == "stage2_meta_v1"

    metrics = json.loads((stack_dir / "meta_metrics.json").read_text(encoding="utf-8"))
    assert "top1" in metrics
    assert "macro_f1" in metrics
    assert metrics["n_train_samples"] > 0
    assert metrics["meta_schema_version"] == "stage2_meta_v1"
    assert metrics["meta_feature_dim"] == int(oof["X"].shape[1])
    assert metrics["meta_feature_names"] == list(oof["feature_names"])


def test_stacking_reuses_run_hyperparams_for_base_model(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-hparams-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "stack-hparams-run",
                "processed_root": str(tmp_path / "outputs" / "processed"),
                "policy": "strict",
                "label_mode": "multiclass",
                "hidden_dim": 160,
                "fusion_layers": 3,
                "fusion_heads": 5,
                "fusion_dropout": 0.2,
                "lr": 0.0007,
                "alpha": 0.25,
                "beta": 0.15,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stacking_module,
        "load_policy_multimodal_data",
        lambda *args, **kwargs: {
            "rgb": np.random.default_rng(1).integers(0, 256, size=(6, 3, 28, 28), dtype=np.uint8),
            "input_ids": np.random.default_rng(2).integers(0, 256, size=(6, 16), dtype=np.int32),
            "attention_mask": np.ones((6, 16), dtype=np.uint8),
            "token_type_ids": np.zeros((6, 16), dtype=np.uint8),
            "y": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
            "split": np.array(["train", "train", "val", "val", "test", "test"], dtype="U8"),
        },
    )

    captured = []

    def fake_train_base_model(*args, **kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(stacking_module, "_train_base_model", fake_train_base_model)

    def fake_predict_meta(*args, **kwargs):
        n_rows = int(args[1].shape[0])
        feature_names = [f"f{i}" for i in range(16)]
        return (
            np.zeros((n_rows, len(feature_names)), dtype=np.float32),
            feature_names,
            {
                "version": "stage2_meta_v1",
                "dim": len(feature_names),
                "feature_names": feature_names,
            },
        )

    monkeypatch.setattr(
        stacking_module,
        "_predict_meta",
        fake_predict_meta,
    )

    class DummyMetaModel:
        def predict(self, x):
            return np.zeros((x.shape[0],), dtype=np.int64)

        def save_model(self, path):
            Path(path).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel())

    code = stacking_main(
        [
            "--run-dir",
            str(run_dir),
            "--n-splits",
            "2",
            "--oof-epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )
    assert code == 0
    assert captured
    for item in captured:
        assert item["hidden_dim"] == 160
        assert item["fusion_layers"] == 3
        assert item["fusion_heads"] == 5
        assert item["fusion_dropout"] == 0.2
        assert item["lr"] == 0.0007
        assert item["alpha"] == 0.25
        assert item["beta"] == 0.15
