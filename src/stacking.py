from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.meta_features import STAGE2_META_SCHEMA_VERSION

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass(frozen=True)
class MetaArtifact:
    path: Path
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    feature_schema: dict[str, Any]
    feature_schema_version: str
    fold_ids: np.ndarray
    split_provenance: dict[str, Any]


def _empty_meta_schema() -> dict[str, Any]:
    return {"version": STAGE2_META_SCHEMA_VERSION, "dim": 0, "feature_names": []}


def _decode_text_scalar(value: Any, *, key: str, path: Path) -> str:
    raw = np.asarray(value)
    if raw.shape == ():
        return str(raw.item())
    if raw.size == 1:
        return str(raw.reshape(-1)[0])
    raise ValueError(f"{path}: key '{key}' must be a scalar string field")


def _fit_meta_learner(x_train: np.ndarray, y_train: np.ndarray, num_classes: int):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for stacking but not available")
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    clf = XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective=objective,
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(x_train, y_train)
    return clf


def _load_meta_artifact(path: Path) -> MetaArtifact:
    if not path.exists():
        raise FileNotFoundError(f"meta artifact not found: {path}")

    artifact = np.load(path, allow_pickle=True)
    required = {
        "X",
        "y",
        "feature_names",
        "feature_schema",
        "feature_schema_version",
        "fold_ids",
        "split_provenance",
    }
    missing = sorted(required.difference(set(artifact.files)))
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")

    x = np.asarray(artifact["X"], dtype=np.float32)
    y = np.asarray(artifact["y"], dtype=np.int32)
    feature_names = [str(name) for name in np.asarray(artifact["feature_names"]).tolist()]
    feature_schema = json.loads(_decode_text_scalar(artifact["feature_schema"], key="feature_schema", path=path))
    feature_schema_version = _decode_text_scalar(
        artifact["feature_schema_version"], key="feature_schema_version", path=path
    )
    fold_ids = np.asarray(artifact["fold_ids"], dtype=np.int32)
    split_provenance = json.loads(_decode_text_scalar(artifact["split_provenance"], key="split_provenance", path=path))

    if x.ndim != 2:
        raise ValueError(f"{path}: X must be 2D, got ndim={x.ndim}")
    if y.ndim != 1:
        raise ValueError(f"{path}: y must be 1D, got ndim={y.ndim}")
    if fold_ids.ndim != 1:
        raise ValueError(f"{path}: fold_ids must be 1D, got ndim={fold_ids.ndim}")
    if x.shape[0] != y.shape[0] or x.shape[0] != fold_ids.shape[0]:
        raise ValueError(
            f"{path}: sample count mismatch among X/y/fold_ids -> {x.shape[0]}/{y.shape[0]}/{fold_ids.shape[0]}"
        )
    if x.shape[1] != len(feature_names):
        raise ValueError(f"{path}: X dim {x.shape[1]} does not match feature_names size {len(feature_names)}")
    if feature_schema_version != STAGE2_META_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported feature_schema_version='{feature_schema_version}', expected '{STAGE2_META_SCHEMA_VERSION}'"
        )
    if not isinstance(split_provenance, dict):
        raise ValueError(f"{path}: split_provenance must decode to an object")
    if str(feature_schema.get("version", "")) != feature_schema_version:
        raise ValueError(f"{path}: feature_schema.version must match feature_schema_version")
    if int(feature_schema.get("dim", -1)) != x.shape[1]:
        raise ValueError(f"{path}: feature_schema.dim must equal X.shape[1]")
    if list(feature_schema.get("feature_names", [])) != feature_names:
        raise ValueError(f"{path}: feature_schema.feature_names must match feature_names")

    return MetaArtifact(
        path=path,
        x=x,
        y=y,
        feature_names=feature_names,
        feature_schema=feature_schema,
        feature_schema_version=feature_schema_version,
        fold_ids=fold_ids,
        split_provenance=split_provenance,
    )


def _validate_oof_boundary(oof_artifact: MetaArtifact) -> None:
    generator = str(oof_artifact.split_provenance.get("generator", ""))
    if generator != "runner_kfold_oof":
        raise ValueError(
            f"{oof_artifact.path}: split_provenance.generator must be 'runner_kfold_oof' for training artifacts"
        )
    if np.any(oof_artifact.fold_ids < 0):
        raise ValueError(f"{oof_artifact.path}: fold_ids must be non-negative for OOF train/val artifacts")


def _validate_schema_alignment(train_meta: MetaArtifact, eval_meta: MetaArtifact) -> None:
    if train_meta.feature_schema_version != eval_meta.feature_schema_version:
        raise ValueError("train/eval schema version mismatch")
    if train_meta.feature_names != eval_meta.feature_names:
        raise ValueError("train/eval feature_names mismatch")
    if train_meta.x.shape[1] != eval_meta.x.shape[1]:
        raise ValueError("train/eval feature dimension mismatch")


def _write_meta_dump(
    out_path: Path,
    *,
    artifact: MetaArtifact,
    split_role: str,
    source_path: Path,
    holdout_meta_path: Path | None,
) -> None:
    provenance = dict(artifact.split_provenance)
    provenance.update(
        {
            "consumer": "src.stacking",
            "split_role": split_role,
            "source_path": str(source_path),
            "holdout_meta_path": str(holdout_meta_path) if holdout_meta_path is not None else None,
        }
    )
    schema = {
        "version": artifact.feature_schema_version,
        "dim": int(artifact.x.shape[1]),
        "feature_names": list(artifact.feature_names),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=artifact.x,
        y=artifact.y,
        feature_names=np.asarray(artifact.feature_names, dtype=np.str_),
        feature_schema=np.array(json.dumps(schema, ensure_ascii=False, sort_keys=True), dtype=np.str_),
        feature_schema_version=np.array(artifact.feature_schema_version, dtype=np.str_),
        fold_ids=artifact.fold_ids.astype(np.int32, copy=False),
        split_provenance=np.array(json.dumps(provenance, ensure_ascii=False, sort_keys=True), dtype=np.str_),
    )


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.0
    top1 = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return top1, macro_f1, macro_recall


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train stacking meta-learner from exported OOF/meta artifacts")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--meta-artifacts-dir", default=None)
    parser.add_argument("--oof-meta-file", default="oof_meta_train.npz")
    parser.add_argument("--eval-meta-file", default="meta_test.npz")
    parser.add_argument("--holdout-meta-path", default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--oof-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    run_dir = Path(args.run_dir)
    meta_artifacts_dir = Path(args.meta_artifacts_dir) if args.meta_artifacts_dir else run_dir / "meta_features"
    oof_source_path = meta_artifacts_dir / args.oof_meta_file
    holdout_meta_path = Path(args.holdout_meta_path) if args.holdout_meta_path else None
    eval_source_path = holdout_meta_path if holdout_meta_path else (meta_artifacts_dir / args.eval_meta_file)

    oof_meta = _load_meta_artifact(oof_source_path)
    eval_meta = _load_meta_artifact(eval_source_path)
    _validate_oof_boundary(oof_meta)
    _validate_schema_alignment(oof_meta, eval_meta)

    if oof_meta.y.size == 0:
        return 2

    num_classes = int(np.max(oof_meta.y)) + 1
    meta_model = _fit_meta_learner(oof_meta.x, oof_meta.y, num_classes=num_classes)
    pred = np.asarray(meta_model.predict(eval_meta.x), dtype=np.int32)
    if pred.shape[0] != eval_meta.y.shape[0]:
        raise ValueError(
            f"meta learner prediction size mismatch: got {pred.shape[0]}, expected {eval_meta.y.shape[0]}"
        )
    top1, macro_f1, macro_recall = _safe_metrics(eval_meta.y, pred)

    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    _write_meta_dump(
        stack_dir / "oof_meta_train.npz",
        artifact=oof_meta,
        split_role="train_val_oof",
        source_path=oof_source_path,
        holdout_meta_path=holdout_meta_path,
    )
    _write_meta_dump(
        stack_dir / "meta_test.npz",
        artifact=eval_meta,
        split_role="eval",
        source_path=eval_source_path,
        holdout_meta_path=holdout_meta_path,
    )

    schema_version = oof_meta.feature_schema_version or STAGE2_META_SCHEMA_VERSION
    metrics = {
        "top1": top1,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "n_train_samples": int(oof_meta.y.shape[0]),
        "n_test_samples": int(eval_meta.y.shape[0]),
        "n_splits": int(args.n_splits),
        "meta_schema_version": schema_version,
        "meta_feature_dim": int(oof_meta.x.shape[1]),
        "meta_feature_names": list(oof_meta.feature_names),
        "metric_source": "stacking_final",
        "is_final_stage2_result": True,
        "stage": "stage2",
        "train_meta_source": str(oof_source_path),
        "eval_meta_source": str(eval_source_path),
        "meta_artifacts_dir": str(meta_artifacts_dir),
        "eval_split": str(eval_meta.split_provenance.get("split", "test")),
        "holdout_meta_path": str(holdout_meta_path) if holdout_meta_path is not None else None,
    }
    (stack_dir / "meta_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (stack_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta_model.save_model(str(stack_dir / "meta_model.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
