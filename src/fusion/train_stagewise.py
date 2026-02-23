from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from src.common.config import load_yaml
from src.common.logging_utils import build_file_logger
from src.fusion.datasets import DummyFusionDataset


def _write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "stage", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def run_train(cfg: Dict[str, Any]) -> Path:
    run_name = str(cfg.get("run_name", "fusion_run"))
    output_root = Path(str(cfg.get("output_root", "outputs/runs")))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    logger = build_file_logger(run_dir / "train.log")

    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", cfg.get("num_classes", 2)))
    dataset = DummyFusionDataset(size=int(cfg.get("train_size", 16)), num_classes=num_classes)

    logger.info("start run=%s", run_name)
    logger.info("dataset_size=%d", len(dataset))

    metrics_rows: List[Dict[str, Any]] = []
    stages = ["stage1_branch_warmup", "stage2_fusion_train", "stage3_pre_stacking"]
    for epoch, stage in enumerate(stages, start=1):
        loss = round(1.0 / epoch, 6)
        metrics_rows.append({"epoch": epoch, "stage": stage, "loss": loss})
        logger.info("%s loss=%.6f", stage, loss)

    _write_metrics_csv(run_dir / "metrics.csv", metrics_rows)

    torch.save(
        {
            "best_stage": "stage2_fusion_train",
            "num_classes": num_classes,
            "metric": "loss",
            "value": min(row["loss"] for row in metrics_rows),
        },
        checkpoints_dir / "best.pt",
    )
    logger.info("saved checkpoint at %s", checkpoints_dir / "best.pt")

    return run_dir


def main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run stagewise fusion training")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    return run_train(cfg)


if __name__ == "__main__":
    main()
