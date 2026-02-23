from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier

from src.common.config import load_yaml


def _safe_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    return top1 - top2


def build_meta_features(
    pred_img: np.ndarray,
    pred_tls: np.ndarray,
    pred_fuse: np.ndarray,
    folds: int = 5,
) -> pd.DataFrame:
    """Build OOF-style meta-features from branch probabilities."""
    pred_img = np.asarray(pred_img, dtype=float)
    pred_tls = np.asarray(pred_tls, dtype=float)
    pred_fuse = np.asarray(pred_fuse, dtype=float)

    if not (pred_img.shape == pred_tls.shape == pred_fuse.shape):
        raise ValueError("pred_img/pred_tls/pred_fuse must have same shape")

    meta = pd.DataFrame(
        {
            "entropy_img": _safe_entropy(pred_img),
            "entropy_tls": _safe_entropy(pred_tls),
            "entropy_fuse": _safe_entropy(pred_fuse),
            "margin_img": _margin(pred_img),
            "margin_tls": _margin(pred_tls),
            "margin_fuse": _margin(pred_fuse),
            "gate_proxy": pred_fuse.max(axis=1) - pred_img.max(axis=1),
            "norm_img": np.linalg.norm(pred_img, axis=1),
            "norm_tls": np.linalg.norm(pred_tls, axis=1),
            "norm_fuse": np.linalg.norm(pred_fuse, axis=1),
            "folds": folds,
        }
    )

    for idx in range(pred_fuse.shape[1]):
        meta[f"logit_img_{idx}"] = pred_img[:, idx]
        meta[f"logit_tls_{idx}"] = pred_tls[:, idx]
        meta[f"logit_fuse_{idx}"] = pred_fuse[:, idx]

    return meta


def fit_meta_learner(meta_features: pd.DataFrame, labels: np.ndarray) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=42)
    model.fit(meta_features.values, labels)
    return model


def run_stacking(cfg: Dict[str, Any]) -> Path:
    run_name = str(cfg.get("run_name", "stacking_run"))
    output_root = Path(str(cfg.get("output_root", "outputs/runs")))
    run_dir = output_root / run_name / "stacking"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal synthetic OOF placeholders for pipeline wiring.
    num_samples = int(cfg.get("num_samples", 16))
    num_classes = int(cfg.get("num_classes", 3))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    raw = rng.random((num_samples, num_classes))
    pred_img = raw / raw.sum(axis=1, keepdims=True)
    raw = rng.random((num_samples, num_classes))
    pred_tls = raw / raw.sum(axis=1, keepdims=True)
    raw = rng.random((num_samples, num_classes))
    pred_fuse = raw / raw.sum(axis=1, keepdims=True)

    meta = build_meta_features(pred_img, pred_tls, pred_fuse, folds=int(cfg.get("folds", 5)))
    meta.to_csv(run_dir / "meta_features.csv", index=False)

    labels = rng.integers(0, num_classes, size=num_samples)
    model = fit_meta_learner(meta, labels)

    with (run_dir / "meta_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "n_samples": num_samples,
                "n_features": int(meta.shape[1]),
                "meta_learner": model.__class__.__name__,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    return run_dir


def main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run stacking meta-learner pipeline")
    parser.add_argument("--config", required=True, help="Path to stacking YAML config")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    return run_stacking(cfg)


if __name__ == "__main__":
    main()
