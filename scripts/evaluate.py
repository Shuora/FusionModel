from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration describing the evaluation task.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file to evaluate.",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist.")
    return yaml.safe_load(path.read_text()) or {}


def validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    validate_checkpoint(args.checkpoint)
    task_name = config.get("task_name", "evaluation")

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Task: {task_name}")
    print(f"Num classes: {config.get('num_classes', 'unknown')}")


if __name__ == "__main__":
    main()
