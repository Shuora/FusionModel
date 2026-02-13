"""Unified runner for CharBERT + MobileViT experiments.

Modes:
  - concat:   feature concatenation fusion
  - weighted: weighted feature fusion
  - attention: cross-attention fusion
  - concat_stacking: concat + XGBoost stacking
  - weighted_stacking: weighted + XGBoost stacking
  - attention_stacking: attention + XGBoost stacking
  - concat_all_ensembles: concat + multi-strategy ensemble
  - attention_all_ensembles: attention + multi-strategy ensemble
  - all: run the above sequentially

This script is designed to be executed from either the repo root or this folder.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fusion_common import (
    add_common_args,
    build_common_kwargs,
    ensure_output_dirs,
    run_fusion_experiment,
    run_stacking_experiment,
)


def _ensure_local_imports() -> None:
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))


def main() -> int:
    _ensure_local_imports()
    p = argparse.ArgumentParser(description="Run CharBERT+MobileViT fusion experiments")
    p.add_argument(
        "--mode",
        choices=[
            "concat",
            "weighted",
            "attention",
            "concat_stacking",
            "weighted_stacking",
            "attention_stacking",
            "concat_all_ensembles",
            "attention_all_ensembles",
            "all",
        ],
        default="all",
        help="Which fusion/ensemble approach to run",
    )
    add_common_args(p)
    args = p.parse_args()

    kwargs = build_common_kwargs(args)
    ensure_output_dirs(kwargs["output_dir"])

    start = time.time()
    if args.mode in ("concat", "all"):
        run_fusion_experiment(fusion_mode="concat", **kwargs)

    if args.mode in ("weighted", "all"):
        run_fusion_experiment(fusion_mode="weighted", **kwargs)

    if args.mode in ("attention", "all"):
        run_fusion_experiment(fusion_mode="attention", **kwargs)

    if args.mode in ("concat_stacking", "all"):
        run_stacking_experiment(
            base_fusion_mode="concat",
            meta_methods=["xgboost"],
            ensemble_tag="concat_stacking",
            **kwargs,
        )

    if args.mode in ("weighted_stacking", "all"):
        run_stacking_experiment(
            base_fusion_mode="weighted",
            meta_methods=["xgboost"],
            ensemble_tag="weighted_stacking",
            **kwargs,
        )

    if args.mode in ("attention_stacking", "all"):
        run_stacking_experiment(
            base_fusion_mode="attention",
            meta_methods=["xgboost"],
            ensemble_tag="attention_stacking",
            **kwargs,
        )

    if args.mode in ("concat_all_ensembles", "all"):
        run_stacking_experiment(
            base_fusion_mode="concat",
            meta_methods=["xgboost", "lightgbm", "catboost", "mlp"],
            ensemble_tag="concat_all_ensembles",
            **kwargs,
        )

    if args.mode in ("attention_all_ensembles", "all"):
        run_stacking_experiment(
            base_fusion_mode="attention",
            meta_methods=["xgboost", "lightgbm", "catboost", "mlp"],
            ensemble_tag="attention_all_ensembles",
            **kwargs,
        )

    elapsed = time.time() - start
    print(f"All done. elapsed={elapsed:.1f}s, outputs={kwargs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
