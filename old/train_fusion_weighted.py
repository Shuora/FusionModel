"""
(3) 图像和序列 weighted 融合
"""
from __future__ import annotations

import argparse

from fusion_common import add_common_args, build_common_kwargs, run_fusion_experiment


def main() -> int:
    p = argparse.ArgumentParser(description="Weighted fusion (MobileViT + CharBERT)")
    add_common_args(p)
    args = p.parse_args()

    kwargs = build_common_kwargs(args)
    run_fusion_experiment(fusion_mode="weighted", **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
