from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import yaml

from fusion_malicious.config import build_run_layout

def build_parser() -> argparse.ArgumentParser:
    default_config = repo_root / "configs" / "binary.yaml"
    parser = argparse.ArgumentParser(description="Train binary models using the MTA/MFCP dataset manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to the training configuration YAML.",
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


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = ensure_cuda()
    task_name = config.get("task_name", "unknown_binary")
    layout = build_run_layout(repo_root / "runs", task_name)
    run_dir = layout.run_dir
    for child in ("checkpoints", "logs", "tmp"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    print(f"Task: {task_name}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
