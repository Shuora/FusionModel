"""Unified runner for attention-based CharBERT + MobileViT experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _ensure_local_imports() -> None:
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))


_ensure_local_imports()

from fusion_common import add_common_args, build_common_kwargs, ensure_output_dirs, run_fusion_experiment, run_stacking_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run attention-only fusion experiments')
    p.add_argument(
        '--mode',
        choices=['attention', 'attention_stacking', 'all'],
        default='all',
        help='Which attention-based experiment to run',
    )
    add_common_args(p)
    task_action = next(action for action in p._actions if action.dest == 'task_name')
    task_action.required = True
    task_action.help = 'ProcessedData task name to run'
    return p


def main() -> int:
    args = build_parser().parse_args()
    kwargs = build_common_kwargs(args)
    ensure_output_dirs(kwargs['output_dir'])

    start = time.time()
    if args.mode in ('attention', 'all'):
        run_fusion_experiment(fusion_mode='attention', **kwargs)

    if args.mode in ('attention_stacking', 'all'):
        run_stacking_experiment(
            base_fusion_mode='attention',
            meta_methods=['xgboost'],
            ensemble_tag='attention_stacking',
            **kwargs,
        )

    elapsed = time.time() - start
    print(f"All done. elapsed={elapsed:.1f}s, outputs={kwargs['output_dir']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
