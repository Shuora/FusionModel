from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


STAGE2_TASKS = [
    {"dataset": "MTA", "num_classes": 7},
    {"dataset": "MFCP", "num_classes": 6},
    {"dataset": "USTC-TFC2016", "num_classes": 10},
]


def build_stage2_tasks() -> List[dict]:
    return [dict(item) for item in STAGE2_TASKS]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build stage2 multiclass task list")
    parser.add_argument("--output", default="outputs/stage2_tasks.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tasks = build_stage2_tasks()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

