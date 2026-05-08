from __future__ import annotations
import argparse
import inspect
from fusion_common import add_common_args, build_common_kwargs, run_fusion_experiment

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Attention fusion (MobileViT + CharBERT)')
    add_common_args(p)
    task_action = next((action for action in p._actions if action.dest == 'task_name'))
    task_action.required = True
    task_action.help = 'ProcessedData task name to train'
    return p

def main() -> int:
    args = build_parser().parse_args()
    kwargs = build_common_kwargs(args)
    sig = inspect.signature(run_fusion_experiment)
    valid_kwargs = {k: v for (k, v) in kwargs.items() if k in sig.parameters}
    run_fusion_experiment(**valid_kwargs)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())