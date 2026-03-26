from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from src.run_dir import resolve_run_dir


def resolve_canonical_final_metric_source_and_path(run_dir: Path) -> Tuple[str, Path]:
    """
    Canonical final-metric rule:
    - explicit level3 final (moe/final_metrics.json) beats level2 final
    - level2 final (stacking/final_metrics.json) beats level2 meta fallback
    - plain moe/moe_metrics.json is not treated as a level3 final unless no stronger final artifact exists
    - eval artifacts are only used when no stage2 final artifacts exist
    """

    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _is_stage2_final(payload: dict) -> bool:
        return bool(payload.get("is_final_stage2_result")) or str(payload.get("metric_source", "")) == "stacking_final"

    cfg_path = run_dir / "config.yaml"
    is_stage2_unified = False
    if cfg_path.exists():
        try:
            cfg_payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg_payload = {}
        is_stage2_unified = str(cfg_payload.get("model_type", "")) == "Stage2UnifiedClassifier"

    eval_test = run_dir / "eval_test.json"
    other_eval_files = sorted(p for p in run_dir.glob("eval_*.json") if p.name != "eval_test.json")

    # Unified stage2 主路径优先采用端到端 eval 结果。
    if is_stage2_unified:
        if eval_test.exists():
            return "eval", eval_test
        if other_eval_files:
            return "eval", other_eval_files[0]
        return "none", eval_test

    # Later-stage final (if present) should supersede stacking-final.
    moe_final = run_dir / "moe" / "final_metrics.json"
    if moe_final.exists():
        return "moe", moe_final

    # Canonical level2 final artifact.
    stacking_final = run_dir / "stacking" / "final_metrics.json"
    if stacking_final.exists():
        return "stacking", stacking_final

    # Legacy / fallback: only treat meta_metrics as canonical when it explicitly declares final semantics.
    stacking_meta = run_dir / "stacking" / "meta_metrics.json"
    if stacking_meta.exists():
        try:
            payload = _read_json(stacking_meta)
        except Exception:
            payload = {}
        if _is_stage2_final(payload):
            return "stacking", stacking_meta

    # Plain moe_metrics is a fallback metric artifact, but not a stronger "final"
    # than an explicit stacking final.
    moe_file = run_dir / "moe" / "moe_metrics.json"
    if moe_file.exists():
        return "moe", moe_file

    # Non-stage2 runs fall back to eval artifacts.
    if eval_test.exists():
        return "eval", eval_test

    if other_eval_files:
        return "eval", other_eval_files[0]

    # As a final fallback, return whatever exists (even if non-final) to keep report generation usable.
    if stacking_meta.exists():
        return "stacking", stacking_meta

    return "none", eval_test


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate run report")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = resolve_run_dir(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "metrics.csv")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_path = fig_dir / "learning_curve.png"
    _plot_learning_curve(metrics, curve_path)

    eval_payload, metric_source, metric_path = _discover_metric_payload(run_dir)
    best_row = _select_report_best_row(run_dir=run_dir, metrics=metrics)

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
    classification_csv = run_dir / "figures" / f"classification_report_{effective_split}.csv"
    classification_json = run_dir / "figures" / f"classification_report_{effective_split}.json"
    if metric_source == "eval" and confusion_csv.exists() and confusion_png.exists():
        artifact_lines.extend(
            [
                f"- Confusion Matrix CSV: `{confusion_csv.as_posix()}`",
                f"- Confusion Matrix PNG: `{confusion_png.as_posix()}`",
            ]
        )
    if metric_source == "eval" and classification_csv.exists():
        artifact_lines.append(f"- Classification Report CSV: `{classification_csv.as_posix()}`")
    if metric_source == "eval" and classification_json.exists():
        artifact_lines.append(f"- Classification Report JSON: `{classification_json.as_posix()}`")

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
    if metric_source == "eval" and confusion_csv.exists():
        confusion_df = pd.read_csv(confusion_csv)
        lines.extend(["", "## Confusion Matrix", _markdown_table(_confusion_markdown_frame(confusion_df))])
    if metric_source == "eval" and classification_csv.exists():
        classification_df = pd.read_csv(classification_csv)
        lines.extend(["", "## Classification Report", _markdown_table(classification_df)])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


def _discover_metric_payload(run_dir: Path) -> Tuple[dict, str, Path]:
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    metric_source, metric_path = resolve_canonical_final_metric_source_and_path(run_dir)
    if not metric_path.exists():
        return {}, metric_source, metric_path
    return _read_json(metric_path), metric_source, metric_path


def _select_report_best_row(run_dir: Path, metrics: pd.DataFrame) -> pd.Series:
    best_ckpt = run_dir / "checkpoints" / "best.ckpt"
    if best_ckpt.exists():
        try:
            payload = torch.load(best_ckpt, map_location="cpu")
            best_epoch = int(payload.get("epoch"))
            matched = metrics.loc[metrics["epoch"] == best_epoch]
            if not matched.empty:
                return matched.iloc[0]
        except Exception:
            pass
    return metrics.sort_values("val_macro_f1", ascending=False).iloc[0]


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


def _confusion_markdown_frame(confusion_df: pd.DataFrame) -> pd.DataFrame:
    table = confusion_df.copy()
    table.columns = [str(col) for col in table.columns]
    table.insert(0, "true/pred", [str(i) for i in range(len(table))])
    return table


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        rendered = [_format_markdown_cell(value) for value in row]
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)


def _format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
