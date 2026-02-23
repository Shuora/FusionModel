from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ablation import build_ablation_grid, write_ablation_plan, write_ablation_summary


def test_ablation_grid_contains_required_groups_and_variants():
    grid = build_ablation_grid()
    groups = {}
    for row in grid:
        groups.setdefault(row["group"], set()).add(row["name"])

    assert groups["temporal_branch"] == {"bilstm_att", "tls_field_bert", "byte_bert"}
    assert groups["fusion_mechanism"] == {
        "linear_w",
        "learnable_weight",
        "cross_attn_gating",
        "cross_attn_gating_no_aux",
    }
    assert groups["rgb_channels"] == {"r_only", "g_only", "b_only", "rgb", "g_no_sni"}
    assert groups["ensemble_complexity"] == {"xgboost_prob", "stacking_enhanced", "moe_router"}


def test_write_ablation_plan_outputs_csv(tmp_path: Path):
    out_csv = tmp_path / "ablation_plan.csv"
    write_ablation_plan(out_csv)
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"group", "name", "notes", "run_cmd"}.issubset(df.columns)
    assert len(df) >= 15


def test_write_ablation_summary_collects_run_metrics(tmp_path: Path):
    run_root = tmp_path / "runs"
    run_a = run_root / "ablation" / "temporal_branch" / "run-a"
    run_b = run_root / "ablation" / "ensemble_complexity" / "run-b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    (run_a / "eval_test.json").write_text(
        json.dumps({"top1": 0.95, "macro_f1": 0.94, "macro_recall": 0.93}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"epoch": 1, "val_macro_f1": 0.90, "val_acc": 0.91},
            {"epoch": 2, "val_macro_f1": 0.92, "val_acc": 0.93},
        ]
    ).to_csv(run_a / "metrics.csv", index=False)
    (run_b / "stacking").mkdir(parents=True, exist_ok=True)
    (run_b / "stacking" / "meta_metrics.json").write_text(
        json.dumps({"top1": 0.97, "macro_f1": 0.96, "macro_recall": 0.95}),
        encoding="utf-8",
    )

    out_csv = tmp_path / "ablation_summary.csv"
    write_ablation_summary(run_root=run_root, output_csv=out_csv)
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 2
    assert {"group", "run_id", "metric_source", "top1", "macro_f1", "macro_recall", "best_val_macro_f1"}.issubset(
        df.columns
    )
    source_map = {row["run_id"]: row["metric_source"] for _, row in df.iterrows()}
    assert source_map["run-a"] == "eval"
    assert source_map["run-b"] == "stacking"
