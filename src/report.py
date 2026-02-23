from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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

    eval_payload = {}
    eval_file = run_dir / "eval_test.json"
    if eval_file.exists():
        eval_payload = json.loads(eval_file.read_text(encoding="utf-8"))

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
        lines.extend(
            [
                "## Test Evaluation",
                f"- Top-1: {eval_payload['top1']:.4f}",
                f"- Macro-F1: {eval_payload['macro_f1']:.4f}",
                f"- Macro-Recall: {eval_payload['macro_recall']:.4f}",
                f"- Num Samples: {eval_payload['num_samples']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            f"- Metrics: `{(run_dir / 'metrics.csv').as_posix()}`",
            f"- Learning Curve: `{curve_path.as_posix()}`",
            f"- Checkpoints: `{(run_dir / 'checkpoints').as_posix()}`",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
