"""
(8) 图像和序列 attention 融合 + 所有不同策略的集成学习
"""
from __future__ import annotations

import argparse

from fusion_common import add_common_args, build_common_kwargs, parse_methods, run_stacking_experiment


def main() -> int:
    p = argparse.ArgumentParser(description="Attention fusion + multi-strategy ensemble")
    add_common_args(p)
    p.add_argument(
        "--meta_methods",
        default="xgboost,lightgbm,catboost,mlp",
        help="Comma-separated meta learners",
    )
    args = p.parse_args()

    kwargs = build_common_kwargs(args)
    methods = parse_methods(args.meta_methods)
    run_stacking_experiment(
        base_fusion_mode="attention",
        meta_methods=methods,
        ensemble_tag="attention_all_ensembles",
        **kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
