"""
(5) 图像和序列 weighted 融合 + 集成学习 (XGBoost)
"""
from __future__ import annotations

import argparse

from fusion_common import add_common_args, build_common_kwargs, parse_methods, run_stacking_experiment


def main() -> int:
    p = argparse.ArgumentParser(description="Weighted fusion + XGBoost stacking")
    add_common_args(p)
    p.add_argument("--meta_methods", default="xgboost", help="Comma-separated meta learners")
    args = p.parse_args()

    kwargs = build_common_kwargs(args)
    methods = parse_methods(args.meta_methods) or ["xgboost"]
    run_stacking_experiment(base_fusion_mode="weighted", meta_methods=methods, ensemble_tag="weighted_stacking", **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
