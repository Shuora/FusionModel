from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def build_ablation_grid() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    temporal = [
        ("bilstm_att", "BiLSTM-Att baseline"),
        ("tls_field_bert", "TLS-Field-BERT main"),
        ("byte_bert", "Byte-BERT enhanced"),
    ]
    for name, notes in temporal:
        rows.append(
            {
                "group": "temporal_branch",
                "name": name,
                "notes": notes,
                "run_cmd": f"python src/train.py --stage fusion --temporal {name}",
            }
        )

    fusion = [
        ("linear_w", "linear weighted fusion"),
        ("learnable_weight", "old learnable fusion"),
        ("cross_attn_gating", "cross-attn + gating"),
        ("cross_attn_gating_no_aux", "cross-attn + gating without aux loss"),
    ]
    for name, notes in fusion:
        rows.append(
            {
                "group": "fusion_mechanism",
                "name": name,
                "notes": notes,
                "run_cmd": f"python src/train.py --stage fusion --fusion-variant {name}",
            }
        )

    rgb = [
        ("r_only", "R channel only"),
        ("g_only", "G channel only"),
        ("b_only", "B channel only"),
        ("rgb", "full RGB"),
        ("g_no_sni", "G channel remove SNI"),
    ]
    for name, notes in rgb:
        rows.append(
            {
                "group": "rgb_channels",
                "name": name,
                "notes": notes,
                "run_cmd": f"python src/train.py --stage fusion --rgb-variant {name}",
            }
        )

    ensemble = [
        ("xgboost_prob", "old probability + XGBoost"),
        ("stacking_enhanced", "enhanced stacking"),
        ("moe_router", "moe router"),
    ]
    for name, notes in ensemble:
        if name == "stacking_enhanced":
            run_cmd = "python src/stacking.py --run-dir runs/<run_id>"
        elif name == "moe_router":
            run_cmd = "python src/moe.py --run-dir runs/<run_id>"
        else:
            run_cmd = "python src/stacking.py --run-dir runs/<run_id> --legacy-prob-only"
        rows.append(
            {
                "group": "ensemble_complexity",
                "name": name,
                "notes": notes,
                "run_cmd": run_cmd,
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
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_acc": best_val_acc,
    }


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ablation experiment plan")
    parser.add_argument("--mode", choices=["plan", "summary"], default="plan")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--output", default="runs/ablation/ablation_plan.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "plan":
        out = write_ablation_plan(args.output)
        print(f"ablation plan saved: {out}")
    else:
        out = write_ablation_summary(run_root=args.run_root, output_csv=args.output)
        print(f"ablation summary saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
