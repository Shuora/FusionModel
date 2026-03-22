from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_PROCESSED_ROOT = "<processed_root>"
DEFAULT_POLICY = "session_full"
DEFAULT_DATASET = "USTC-TFC2016"
DEFAULT_LABEL_MODE = "multiclass"
DEFAULT_NUM_CLASSES = 10
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
ABLATION_RUN_ROOT = "runs/ablation"


def _run_root(group: str) -> str:
    return f"{ABLATION_RUN_ROOT}/{group}"


def _run_dir(group: str, name: str) -> str:
    return f"{_run_root(group)}/{name}"


def _train_cmd(
    *,
    group: str,
    name: str,
    stage: str,
    train_max_samples: int | None = None,
) -> str:
    parts = [
        "python -m src.train",
        f"--processed-root {DEFAULT_PROCESSED_ROOT}",
        f"--policy {DEFAULT_POLICY}",
        f"--datasets {DEFAULT_DATASET}",
        f"--label-mode {DEFAULT_LABEL_MODE}",
        f"--num-classes {DEFAULT_NUM_CLASSES}",
        f"--stage {stage}",
        f"--epochs {DEFAULT_EPOCHS}",
        f"--batch-size {DEFAULT_BATCH_SIZE}",
        f"--seed {DEFAULT_SEED}",
        f"--run-root {_run_root(group)}",
        f"--run-id {name}",
    ]
    if train_max_samples is not None:
        parts.append(f"--train-max-samples {train_max_samples}")
    return " ".join(parts)


def _evaluate_cmd(*, group: str, name: str) -> str:
    return f"python -m src.evaluate --run-dir {_run_dir(group, name)} --split test"


def _stacking_cmd(*, group: str, name: str) -> str:
    return f"python -m src.stacking --run-dir {_run_dir(group, name)}"


def _moe_cmd(*, group: str, name: str) -> str:
    return f"python -m src.moe --run-dir {_run_dir(group, name)}"


def build_ablation_grid() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    backbone_group = "backbone_stage"
    rows.append(
        {
            "group": backbone_group,
            "name": "warmup_eval",
            "notes": "Warmup stage baseline with test evaluation",
            "run_cmd": " && ".join(
                [
                    _train_cmd(group=backbone_group, name="warmup_eval", stage="warmup"),
                    _evaluate_cmd(group=backbone_group, name="warmup_eval"),
                ]
            ),
        }
    )
    rows.append(
        {
            "group": backbone_group,
            "name": "fusion_eval",
            "notes": "Fusion stage baseline with test evaluation",
            "run_cmd": " && ".join(
                [
                    _train_cmd(group=backbone_group, name="fusion_eval", stage="fusion"),
                    _evaluate_cmd(group=backbone_group, name="fusion_eval"),
                ]
            ),
        }
    )

    sample_group = "sample_budget"
    for limit in (4000, 2000, 1000):
        name = f"train{limit}"
        rows.append(
            {
                "group": sample_group,
                "name": name,
                "notes": f"Fusion stage with train_max_samples={limit}",
                "run_cmd": " && ".join(
                    [
                        _train_cmd(group=sample_group, name=name, stage="fusion", train_max_samples=limit),
                        _evaluate_cmd(group=sample_group, name=name),
                    ]
                ),
            }
        )

    ensemble_group = "ensemble_complexity"
    rows.append(
        {
            "group": ensemble_group,
            "name": "fusion_eval",
            "notes": "Fusion model evaluated directly as ensemble baseline",
            "run_cmd": " && ".join(
                [
                    _train_cmd(group=ensemble_group, name="fusion_eval", stage="fusion"),
                    _evaluate_cmd(group=ensemble_group, name="fusion_eval"),
                ]
            ),
        }
    )
    rows.append(
        {
            "group": ensemble_group,
            "name": "stacking_meta",
            "notes": "Fusion backbone followed by stacking meta-learner",
            "run_cmd": " && ".join(
                [
                    _train_cmd(group=ensemble_group, name="stacking_meta", stage="fusion"),
                    _stacking_cmd(group=ensemble_group, name="stacking_meta"),
                ]
            ),
        }
    )
    rows.append(
        {
            "group": ensemble_group,
            "name": "moe_router",
            "notes": "Fusion backbone followed by MoE router training",
            "run_cmd": " && ".join(
                [
                    _train_cmd(group=ensemble_group, name="moe_router", stage="fusion"),
                    _moe_cmd(group=ensemble_group, name="moe_router"),
                ]
            ),
        }
    )

    return rows


def write_ablation_plan(output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_ablation_grid())
    df.to_csv(output_csv, index=False)
    return output_csv


def collect_ablation_summary(run_root: str | Path) -> pd.DataFrame:
    run_root = Path(run_root)
    base_dir = run_root / "ablation"
    rows: List[Dict[str, object]] = []
    if not base_dir.exists():
        return pd.DataFrame(
            columns=[
                "group",
                "run_id",
                "metric_source",
                "top1",
                "macro_f1",
                "macro_recall",
                "paper_macro_precision",
                "paper_macro_recall",
                "paper_macro_f1",
                "best_val_macro_f1",
                "best_val_acc",
            ]
        )

    for group_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        group = group_dir.name
        for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            row = _read_one_ablation_run(group=group, run_dir=run_dir)
            rows.append(row)
    return pd.DataFrame(rows)


def write_ablation_summary(run_root: str | Path, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = collect_ablation_summary(run_root)
    df.to_csv(output_csv, index=False)
    return output_csv


def _read_one_ablation_run(group: str, run_dir: Path) -> Dict[str, object]:
    top1 = np.nan
    macro_f1 = np.nan
    macro_recall = np.nan
    paper_macro_precision = np.nan
    paper_macro_recall = np.nan
    paper_macro_f1 = np.nan
    metric_source = "none"

    stacking_json = run_dir / "stacking" / "meta_metrics.json"
    moe_json = run_dir / "moe" / "moe_metrics.json"
    eval_json = run_dir / "eval_test.json"

    metrics_payload: Dict[str, object] = {}
    if stacking_json.exists():
        metric_source = "stacking"
        metrics_payload = _read_json(stacking_json)
    elif moe_json.exists():
        metric_source = "moe"
        metrics_payload = _read_json(moe_json)
    elif eval_json.exists():
        metric_source = "eval"
        metrics_payload = _read_json(eval_json)

    if metrics_payload:
        top1 = float(metrics_payload.get("top1", np.nan))
        macro_f1 = float(metrics_payload.get("macro_f1", np.nan))
        macro_recall = float(metrics_payload.get("macro_recall", np.nan))
        paper_macro_precision = float(metrics_payload.get("paper_macro_precision", np.nan))
        paper_macro_recall = float(metrics_payload.get("paper_macro_recall", np.nan))
        paper_macro_f1 = float(metrics_payload.get("paper_macro_f1", np.nan))

    best_val_macro_f1 = np.nan
    best_val_acc = np.nan
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if not df.empty:
            if "val_macro_f1" in df.columns:
                best_val_macro_f1 = float(df["val_macro_f1"].max())
            if "val_acc" in df.columns:
                best_val_acc = float(df["val_acc"].max())

    return {
        "group": group,
        "run_id": run_dir.name,
        "metric_source": metric_source,
        "top1": top1,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "paper_macro_precision": paper_macro_precision,
        "paper_macro_recall": paper_macro_recall,
        "paper_macro_f1": paper_macro_f1,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_acc": best_val_acc,
    }


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 ablation 实验计划")
    parser.add_argument("--mode", choices=["plan", "summary"], default="plan")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--output", default="runs/ablation/ablation_plan.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "plan":
        out = write_ablation_plan(args.output)
        print(f"ablation 计划已保存: {out}")
    else:
        out = write_ablation_summary(run_root=args.run_root, output_csv=args.output)
        print(f"ablation 汇总已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
