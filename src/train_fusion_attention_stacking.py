"""
(6) 图像和序列 attention 融合 + 集成学习 (XGBoost)
"""

from __future__ import annotations



import argparse

import inspect



from fusion_common import add_common_args, build_common_kwargs, parse_methods, run_stacking_experiment





def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(description="Attention fusion + XGBoost stacking")

    add_common_args(p)

    task_action = next(action for action in p._actions if action.dest == "task_name")

    task_action.required = True

    task_action.help = "ProcessedData task name to train"

    p.add_argument("--meta_methods", default="xgboost", help="Comma-separated meta learners")

    return p





def main() -> int:

    args = build_parser().parse_args()

    kwargs = build_common_kwargs(args)

    methods = parse_methods(args.meta_methods) or ["xgboost"]



                                                                 

    base_mode = kwargs.pop("fusion_mode", "attention")



                                                                                  

    sig = inspect.signature(run_stacking_experiment)

    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}



    run_stacking_experiment(

        base_fusion_mode=base_mode,

        meta_methods=methods,

        ensemble_tag="attention_stacking",

        **valid_kwargs

    )

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

