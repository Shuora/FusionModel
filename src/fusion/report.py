from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.common.logging_utils import build_file_logger


def generate_report(
    metrics: Dict[str, Any],
    figures: Dict[str, str],
    output_path: Path,
    run_info: Dict[str, Any] | None = None,
) -> Path:
    run_info = run_info or {}

    lines = [
        "# 实验报告",
        "",
        "## 实验信息",
        f"- run_id: {run_info.get('run_id', 'smoke-run')}",
        f"- dataset: {run_info.get('dataset', 'tls_full')}",
        "",
        "## 指标",
        f"- Acc: {metrics.get('acc', 0.0):.4f}",
        f"- Macro-F1: {metrics.get('macro_f1', 0.0):.4f}",
        f"- Macro-Recall: {metrics.get('macro_recall', 0.0):.4f}",
        "",
        "## 混淆",
        f"- confusion_matrix: ![]({figures.get('confusion_matrix', '')})",
        "",
        "## 错分分析",
        "- Top errors: placeholder",
        "",
        "## 学习曲线",
        f"- metrics_curve: ![]({figures.get('metrics_curve', '')})",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_report(run_dir: Path) -> Path:
    outputs_root = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
    log_path = outputs_root / "logs" / "report" / f"{run_dir.name}.log"
    logger = build_file_logger(log_path, name="fusion.report")
    logger.info("start report run_dir=%s", run_dir)

    eval_file = run_dir / "evaluation.json"
    if eval_file.exists():
        payload = json.loads(eval_file.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        figures = payload.get("figures", {})
    else:
        metrics = {}
        figures = {}

    report_path = generate_report(metrics, figures, run_dir / "report.md", run_info={"run_id": run_dir.name})
    logger.info("finish report report_path=%s", report_path)
    return report_path


def main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate markdown report for one run")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    args = parser.parse_args(argv)
    return run_report(Path(args.run_dir))


if __name__ == "__main__":
    main()
