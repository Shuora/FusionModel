from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.stacking as stacking_module
from src.report import main as report_main
from src.stacking import main as stacking_main


def _write_run_config(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_dir.name,
                "processed_root": str(run_dir / "unused_processed_root"),
                "policy": "strict",
                "label_mode": "multiclass",
                "stage": "fusion",
                "epochs": 1,
                "batch_size": 4,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_meta_artifact(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    fold_ids: np.ndarray,
    split_provenance: dict,
    schema_version: str = "stage2_meta_v1",
) -> None:
    schema = {
        "version": schema_version,
        "dim": int(x.shape[1]),
        "feature_names": list(feature_names),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int32),
        feature_names=np.asarray(feature_names, dtype=np.str_),
        feature_schema=np.array(json.dumps(schema, ensure_ascii=False, sort_keys=True), dtype=np.str_),
        feature_schema_version=np.array(schema_version, dtype=np.str_),
        fold_ids=np.asarray(fold_ids, dtype=np.int32),
        split_provenance=np.array(json.dumps(split_provenance, ensure_ascii=False, sort_keys=True), dtype=np.str_),
    )


def _prepare_meta_inputs(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    _write_run_config(run_dir)
    feature_names = ["f0", "f1", "f2", "f3"]
    oof_x = np.asarray(
        [
            [0.2, 0.1, 0.9, 0.3],
            [0.8, 0.2, 0.1, 0.7],
            [0.3, 0.9, 0.2, 0.4],
            [0.7, 0.8, 0.3, 0.6],
        ],
        dtype=np.float32,
    )
    oof_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    eval_x = np.asarray(
        [
            [0.25, 0.15, 0.85, 0.35],
            [0.75, 0.85, 0.25, 0.65],
        ],
        dtype=np.float32,
    )
    eval_y = np.asarray([0, 1], dtype=np.int32)

    meta_dir = run_dir / "meta_features"
    _write_meta_artifact(
        meta_dir / "oof_meta_train.npz",
        x=oof_x,
        y=oof_y,
        feature_names=feature_names,
        fold_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
        split_provenance={
            "generator": "runner_kfold_oof",
            "split": "train_val",
            "n_splits": 2,
        },
    )
    _write_meta_artifact(
        meta_dir / "meta_test.npz",
        x=eval_x,
        y=eval_y,
        feature_names=feature_names,
        fold_ids=np.asarray([-1, -1], dtype=np.int32),
        split_provenance={
            "generator": "runner_holdout_export",
            "split": "test",
            "source": "manifest:test",
        },
    )
    return oof_x, oof_y, eval_x, eval_y, feature_names


def _rewrite_npz(path: Path, **updates) -> None:
    loaded = np.load(path, allow_pickle=True)
    payload = {key: loaded[key] for key in loaded.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


class DummyMetaModel:
    def __init__(self, pred: np.ndarray | None = None):
        self._pred = pred

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._pred is not None:
            return np.asarray(self._pred, dtype=np.int64)
        return np.zeros((x.shape[0],), dtype=np.int64)

    def save_model(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")


def test_stacking_pipeline_generates_meta_artifacts(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-run"
    _prepare_meta_inputs(run_dir)
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))
    code = stacking_main(
        [
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--num-workers",
            "0",
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
        assert "fold_ids" in arr.files
        assert "split_provenance" in arr.files
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
    assert metrics["n_splits"] == 2


def test_stacking_oof_only_consumes_exported_meta_artifacts(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-oof-only-run"
    oof_x, oof_y, _, _, _ = _prepare_meta_inputs(run_dir)

    def fail_data_loading(*args, **kwargs):
        raise AssertionError("stacking must not reload multimodal raw tensors")

    monkeypatch.setattr(stacking_module, "load_policy_multimodal_data", fail_data_loading, raising=False)
    monkeypatch.setattr(
        stacking_module,
        "_train_base_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OOF generator must be external")),
        raising=False,
    )

    captured = {}

    def fake_fit_meta_learner(x_train, y_train, num_classes):
        captured["x_train"] = np.asarray(x_train)
        captured["y_train"] = np.asarray(y_train)
        captured["num_classes"] = int(num_classes)
        return DummyMetaModel(pred=np.asarray([0, 1]))

    monkeypatch.setattr(stacking_module, "_fit_meta_learner", fake_fit_meta_learner)
    code = stacking_main(
        [
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )
    assert code == 0
    assert np.array_equal(captured["x_train"], oof_x)
    assert np.array_equal(captured["y_train"], oof_y)
    assert captured["num_classes"] == 2


def test_stacking_final_metric_source_is_explicit(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-final-metric-source-run"
    _prepare_meta_inputs(run_dir)
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    code = stacking_main(
        [
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )
    assert code == 0

    final_metrics_path = run_dir / "stacking" / "final_metrics.json"
    assert final_metrics_path.exists()
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    assert final_metrics["metric_source"] == "stacking_final"
    assert final_metrics["is_final_stage2_result"] is True


def test_metric_source_prefers_level2_final_artifact_over_eval_json(tmp_path: Path):
    run_dir = tmp_path / "runs" / "stage2-final-report-source"
    _write_run_config(run_dir)
    (run_dir / "metrics.csv").write_text(
        "epoch,train_loss,val_loss,train_acc,val_acc,val_macro_f1\n1,0.8,0.7,0.6,0.5,0.4\n",
        encoding="utf-8",
    )
    (run_dir / "eval_test.json").write_text(
        json.dumps({"top1": 0.11, "macro_f1": 0.12, "macro_recall": 0.13, "num_samples": 2}),
        encoding="utf-8",
    )
    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    (stack_dir / "meta_metrics.json").write_text(
        json.dumps(
            {
                "top1": 0.91,
                "macro_f1": 0.9,
                "macro_recall": 0.89,
                "n_test_samples": 2,
                "metric_source": "stacking_final",
                "is_final_stage2_result": True,
            }
        ),
        encoding="utf-8",
    )

    code = report_main(["--run-dir", str(run_dir)])
    assert code == 0
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Metric Source: stacking" in report_text
    assert "stacking/meta_metrics.json" in report_text


def test_stacking_rejects_eval_artifact_with_oof_semantics(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-eval-semantic-mismatch"
    _prepare_meta_inputs(run_dir)
    eval_path = run_dir / "meta_features" / "meta_test.npz"
    _rewrite_npz(
        eval_path,
        split_provenance=np.array(
            json.dumps({"generator": "runner_kfold_oof", "split": "train_val", "n_splits": 2}, ensure_ascii=False),
            dtype=np.str_,
        ),
        fold_ids=np.asarray([0, 1], dtype=np.int32),
    )
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="evaluation artifacts"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_rejects_invalid_label_contract(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-invalid-labels"
    _prepare_meta_inputs(run_dir)
    oof_path = run_dir / "meta_features" / "oof_meta_train.npz"
    _rewrite_npz(oof_path, y=np.asarray([0, 2, 0, 2], dtype=np.int32))
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="labels"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_rejects_fractional_labels_on_load(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-fractional-labels"
    _prepare_meta_inputs(run_dir)
    oof_path = run_dir / "meta_features" / "oof_meta_train.npz"
    _rewrite_npz(oof_path, y=np.asarray([0.0, 1.5, 0.0, 1.0], dtype=np.float64))
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="integer values without fractional part"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_rejects_oof_split_value_mismatch(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-oof-split-mismatch"
    _prepare_meta_inputs(run_dir)
    oof_path = run_dir / "meta_features" / "oof_meta_train.npz"
    _rewrite_npz(
        oof_path,
        split_provenance=np.array(
            json.dumps({"generator": "runner_kfold_oof", "split": "train", "n_splits": 2}, ensure_ascii=False),
            dtype=np.str_,
        ),
    )
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="split_provenance.split must be 'train_val'"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_rejects_oof_n_splits_provenance_conflict(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-oof-n-splits-conflict"
    _prepare_meta_inputs(run_dir)
    oof_path = run_dir / "meta_features" / "oof_meta_train.npz"
    _rewrite_npz(
        oof_path,
        split_provenance=np.array(
            json.dumps({"generator": "runner_kfold_oof", "split": "train_val", "n_splits": 9}, ensure_ascii=False),
            dtype=np.str_,
        ),
    )
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="conflicts with fold_ids-inferred n_splits"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_rejects_non_contiguous_oof_fold_ids(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-oof-foldids-non-contiguous"
    _prepare_meta_inputs(run_dir)
    oof_path = run_dir / "meta_features" / "oof_meta_train.npz"
    _rewrite_npz(oof_path, fold_ids=np.asarray([0, 2, 0, 2], dtype=np.int32))
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="fold_ids must be contiguous"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_rejects_schema_version_mismatch(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-schema-version-mismatch"
    _, _, eval_x, eval_y, feature_names = _prepare_meta_inputs(run_dir)
    eval_path = run_dir / "meta_features" / "meta_test.npz"
    schema = {"version": "stage2_meta_v2", "dim": int(eval_x.shape[1]), "feature_names": feature_names}
    _write_meta_artifact(
        eval_path,
        x=eval_x,
        y=eval_y,
        feature_names=feature_names,
        fold_ids=np.asarray([-1, -1], dtype=np.int32),
        split_provenance={"generator": "runner_holdout_export", "split": "test", "source": "manifest:test"},
        schema_version="stage2_meta_v2",
    )
    _rewrite_npz(
        eval_path,
        feature_schema=np.array(json.dumps(schema, ensure_ascii=False, sort_keys=True), dtype=np.str_),
    )
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.raises(ValueError, match="unsupported feature_schema_version"):
        stacking_main(["--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0"])


def test_stacking_warns_legacy_cli_knobs_not_authoritative(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "stack-legacy-cli-warning"
    _prepare_meta_inputs(run_dir)
    monkeypatch.setattr(stacking_module, "_fit_meta_learner", lambda x, y, num_classes: DummyMetaModel(pred=np.asarray([0, 1])))

    with pytest.warns(RuntimeWarning, match="artifact provenance"):
        code = stacking_main(
            [
                "--run-dir",
                str(run_dir),
                "--n-splits",
                "9",
                "--oof-epochs",
                "5",
                "--batch-size",
                "64",
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--seed",
                "100",
            ]
        )
    assert code == 0
