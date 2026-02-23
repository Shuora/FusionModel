from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.common.logging_utils import build_file_logger


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "acc": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }


def save_basic_figures(run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, str]:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_path = figures_dir / "confusion_matrix_smoke.png"
    curve_path = figures_dir / "metrics_curve_smoke.png"

    cm = np.array(metrics.get("confusion_matrix", [[1]]), dtype=float)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    plt.figure(figsize=(4, 3))
    x = [1, 2, 3]
    y = [metrics.get("macro_f1", 0.0) * 0.8, metrics.get("macro_f1", 0.0) * 0.9, metrics.get("macro_f1", 0.0)]
    plt.plot(x, y)
    plt.title("Metrics Curve")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    return {
        "confusion_matrix": str(cm_path.relative_to(run_dir)),
        "metrics_curve": str(curve_path.relative_to(run_dir)),
    }


def _infer_outputs_root(run_dir: Path) -> Path:
    # Expected run_dir layout: outputs/runs/<run_id>
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent
    return run_dir.parent


def run_evaluate(run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs_root = _infer_outputs_root(run_dir)
    log_path = outputs_root / "logs" / "evaluate" / f"{run_dir.name}.log"
    logger = build_file_logger(log_path, name="fusion.evaluate")
    logger.info("start evaluate run_dir=%s", run_dir)

    # Placeholder labels for smoke evaluation.
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = evaluate_predictions(y_true, y_pred)
    figures = save_basic_figures(run_dir, metrics)

    payload = {"metrics": metrics, "figures": figures}
    (run_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("metrics=%s", metrics)
    logger.info("figures=%s", figures)
    logger.info("finish evaluate log_path=%s", log_path)
    return payload


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate fusion run")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    args = parser.parse_args(argv)
    return run_evaluate(Path(args.run_dir))


if __name__ == "__main__":
    main()
