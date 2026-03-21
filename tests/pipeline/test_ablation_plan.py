from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ablation import build_ablation_grid, write_ablation_plan, write_ablation_summary


UNSUPPORTED_FLAGS = ("--temporal", "--fusion-variant", "--rgb-variant", "--legacy-prob-only")


def test_ablation_grid_uses_supported_entrypoints_and_flags():
    grid = build_ablation_grid()
    groups = {}
    for row in grid:
        groups.setdefault(row["group"], set()).add(row["name"])
        run_cmd = row["run_cmd"]
        assert "python -m src." in run_cmd
        assert "python src/" not in run_cmd
        for flag in UNSUPPORTED_FLAGS:
            assert flag not in run_cmd
        assert "--processed-root <processed_root>" in run_cmd
        assert f"--run-root runs/ablation/{row['group']}" in run_cmd
        assert f"--run-id {row['name']}" in run_cmd

    assert groups == {
        "backbone_stage": {"warmup_eval", "fusion_eval"},
        "sample_budget": {"train4000", "train2000", "train1000"},
        "ensemble_complexity": {"fusion_eval", "stacking_meta", "moe_router"},
    }


def test_write_ablation_plan_outputs_csv_with_consistent_run_layout(tmp_path: Path):
    out_csv = tmp_path / "ablation_plan.csv"
    write_ablation_plan(out_csv)
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"group", "name", "notes", "run_cmd"}.issubset(df.columns)
    assert len(df) == 8

    for row in df.to_dict(orient="records"):
        run_cmd = row["run_cmd"]
        expected_group_root = f"runs/ablation/{row['group']}"
        expected_run_dir = f"{expected_group_root}/{row['name']}"
        assert f"--run-root {expected_group_root}" in run_cmd
        assert f"--run-id {row['name']}" in run_cmd
        if "src.evaluate" in run_cmd or "src.stacking" in run_cmd or "src.moe" in run_cmd:
            assert expected_run_dir in run_cmd


def test_write_ablation_summary_collects_run_metrics(tmp_path: Path):
    run_root = tmp_path / "runs"
    run_a = run_root / "ablation" / "backbone_stage" / "run-a"
    run_b = run_root / "ablation" / "ensemble_complexity" / "run-b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    (run_a / "eval_test.json").write_text(
        json.dumps(
            {
                "top1": 0.95,
                "macro_f1": 0.94,
                "macro_recall": 0.93,
                "paper_macro_precision": 0.92,
                "paper_macro_recall": 0.91,
                "paper_macro_f1": 0.915,
            }
        ),
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
    assert {
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
    }.issubset(df.columns)
    source_map = {row["run_id"]: row["metric_source"] for _, row in df.iterrows()}
    assert source_map["run-a"] == "eval"
    assert source_map["run-b"] == "stacking"
    row_a = df[df["run_id"] == "run-a"].iloc[0]
    assert row_a["paper_macro_f1"] == 0.915
