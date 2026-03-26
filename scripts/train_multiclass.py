from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multiclass classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration describing dataset splits.",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist.")
    return yaml.safe_load(path.read_text()) or {}


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")
    return torch.device("cuda")


def build_run_layout(task_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "runs" / task_name / timestamp
    for child in ("checkpoints", "logs", "tmp"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = ensure_cuda()
    task_name = config.get("task_name", "multiclass_run")
    run_dir = build_run_layout(task_name)

    print(f"Task: {task_name}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
