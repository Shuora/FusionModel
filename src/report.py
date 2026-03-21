from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate run report")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "metrics.csv")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_path = fig_dir / "learning_curve.png"
    _plot_learning_curve(metrics, curve_path)

    eval_payload, metric_source, metric_path = _discover_metric_payload(run_dir)
    best_row = metrics.sort_values("val_macro_f1", ascending=False).iloc[0]

    lines = [
        f"# Run Report: {cfg['run_id']}",
        "",
        "## Experiment Info",
        f"- Stage: {cfg['stage']}",
        f"- Policy: {cfg['policy']}",
        f"- Epochs: {cfg['epochs']}",
        f"- Batch Size: {cfg['batch_size']}",
        "",
        "## Best Validation",
        f"- Best Epoch: {int(best_row['epoch'])}",
        f"- Val Macro-F1: {best_row['val_macro_f1']:.4f}",
        f"- Val Acc: {best_row['val_acc']:.4f}",
        "",
    ]
    if eval_payload:
        num_samples = eval_payload.get("num_samples", eval_payload.get("n_test_samples", 0))
        lines.extend(
            [
                "## Evaluation",
                f"- Metric Source: {metric_source}",
                f"- Top-1: {eval_payload['top1']:.4f}",
                f"- Macro-Precision: {float(eval_payload.get('macro_precision', 0.0)):.4f}",
                f"- Macro-F1: {eval_payload['macro_f1']:.4f}",
                f"- Macro-Recall: {eval_payload['macro_recall']:.4f}",
                f"- Num Samples: {int(num_samples)}",
                f"- Metrics File: `{metric_path.as_posix()}`",
                "",
            ]
        )
        if any(key in eval_payload for key in ("paper_macro_precision", "paper_macro_recall", "paper_macro_f1")):
            lines.extend(
                [
                    "## Paper-Compatible Metrics",
                    f"- Paper Precision: {_format_metric(eval_payload.get('paper_precision'))}",
                    f"- Paper Recall: {_format_metric(eval_payload.get('paper_recall'))}",
                    f"- Paper F1: {_format_metric(eval_payload.get('paper_f1'))}",
                    f"- Paper Macro-Precision: {_format_metric(eval_payload.get('paper_macro_precision'))}",
                    f"- Paper Macro-Recall: {_format_metric(eval_payload.get('paper_macro_recall'))}",
                    f"- Paper Macro-F1: {_format_metric(eval_payload.get('paper_macro_f1'))}",
                    "",
                ]
            )

    artifact_lines = [
        "## Artifacts",
        f"- Metrics: `{(run_dir / 'metrics.csv').as_posix()}`",
        f"- Learning Curve: `{curve_path.as_posix()}`",
    ]

    effective_split = str(eval_payload.get("effective_split", eval_payload.get("split", "test"))) if eval_payload else "test"
    confusion_csv = run_dir / "figures" / f"confusion_matrix_{effective_split}.csv"
    confusion_png = run_dir / "figures" / f"confusion_matrix_{effective_split}.png"
    if metric_source == "eval" and confusion_csv.exists() and confusion_png.exists():
        artifact_lines.extend(
            [
                f"- Confusion Matrix CSV: `{confusion_csv.as_posix()}`",
                f"- Confusion Matrix PNG: `{confusion_png.as_posix()}`",
            ]
        )

    stacking_metrics = run_dir / "stacking" / "meta_metrics.json"
    stacking_model = run_dir / "stacking" / "meta_model.json"
    if stacking_metrics.exists():
        artifact_lines.append(f"- Stacking Metrics: `{stacking_metrics.as_posix()}`")
    if stacking_model.exists():
        artifact_lines.append(f"- Stacking Model: `{stacking_model.as_posix()}`")

    moe_metrics = run_dir / "moe" / "moe_metrics.json"
    moe_router = run_dir / "moe" / "router.ckpt"
    if moe_metrics.exists():
        artifact_lines.append(f"- MoE Metrics: `{moe_metrics.as_posix()}`")
    if moe_router.exists():
        artifact_lines.append(f"- MoE Router: `{moe_router.as_posix()}`")

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        artifact_lines.append(f"- Checkpoints: `{checkpoints_dir.as_posix()}`")

    lines.extend(artifact_lines)
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


def _discover_metric_payload(run_dir: Path) -> Tuple[dict, str, Path]:
    eval_test = run_dir / "eval_test.json"
    if eval_test.exists():
        return json.loads(eval_test.read_text(encoding="utf-8")), "eval", eval_test

    other_eval_files = sorted(p for p in run_dir.glob("eval_*.json") if p.name != "eval_test.json")
    if other_eval_files:
        path = other_eval_files[0]
        return json.loads(path.read_text(encoding="utf-8")), "eval", path

    stacking_file = run_dir / "stacking" / "meta_metrics.json"
    if stacking_file.exists():
        return json.loads(stacking_file.read_text(encoding="utf-8")), "stacking", stacking_file

    moe_file = run_dir / "moe" / "moe_metrics.json"
    if moe_file.exists():
        return json.loads(moe_file.read_text(encoding="utf-8")), "moe", moe_file

    return {}, "none", eval_test


def _plot_learning_curve(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train_loss")
    axes[0].plot(metrics["epoch"], metrics["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(metrics["epoch"], metrics["train_acc"], label="train_acc")
    axes[1].plot(metrics["epoch"], metrics["val_acc"], label="val_acc")
    axes[1].plot(metrics["epoch"], metrics["val_macro_f1"], label="val_macro_f1")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
