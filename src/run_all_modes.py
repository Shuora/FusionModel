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

from fusion_common import (
    add_common_args,
    build_common_kwargs,
    ensure_output_dirs,
    run_fusion_experiment,
    run_stacking_experiment,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run attention-based CharBERT + MobileViT experiments')
    p.add_argument(
        '--mode',
        choices=['attention', 'attention_stacking', 'stress_test', 'all'],
        default='all',
        help='Which attention-based experiment to run',
    )
    add_common_args(p)
    task_action = next(action for action in p._actions if action.dest == 'task_name')
    task_action.required = True
    task_action.help = 'ProcessedData task name or prefix for stress_test (e.g. mta)'
    return p


def main() -> int:
    args = build_parser().parse_args()
    
    if args.mode == 'stress_test':
        # args.task_name is used as a prefix (e.g., 'mta' or 'mfcp')
        ratios = [2, 5, 10, 15]
        tasks = [f"{args.task_name}_ratio{r}" for r in ratios]
        print(f"Starting stress test for prefix: {args.task_name}, tasks: {tasks}")
        
        for task in tasks:
            print(f"\n{'='*20} Running task: {task} {'='*20}")
            args.task_name = task
            kwargs = build_common_kwargs(args)
            ensure_output_dirs(kwargs['output_dir'])
            
            # Run both base and stacking for each ratio
            run_fusion_experiment(fusion_mode='attention', **kwargs)
            run_stacking_experiment(
                base_fusion_mode='attention',
                meta_methods=['xgboost'],
                ensemble_tag='attention_stacking',
                **kwargs,
            )
        return 0

    kwargs = build_common_kwargs(args)
    ensure_output_dirs(kwargs['output_dir'])

    start = time.time()
    if args.mode in ('attention', 'all'):
        set_seed(args.seed)
        run_fusion_experiment(fusion_mode='attention', **kwargs)

    if args.mode in ('attention_stacking', 'all'):
        set_seed(args.seed)
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
