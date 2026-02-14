from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_profiles(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    profiles = data.get("profiles", data)
    if not isinstance(profiles, dict):
        raise ValueError(f"无效配置文件: {path}")
    return profiles


def _to_cli_args(config: dict) -> List[str]:
    args: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        args.extend([flag, str(value)])
    return args


def _run(script: Path, cli_args: List[str]) -> None:
    cmd = [sys.executable, str(script)] + cli_args
    print("▶️ 执行命令:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行 attention / attention_stacking 训练")
    parser.add_argument("--profile", required=True, help="训练配置名，例如 cic5_balanced 或 ustc_baseline")
    parser.add_argument(
        "--profiles",
        default=str(Path("configs") / "train_profiles.yaml"),
        help="训练配置 YAML 路径",
    )
    parser.add_argument(
        "--mode",
        choices=["attention", "attention_stacking", "all"],
        default="all",
        help="执行模式",
    )
    parser.add_argument("--dataset_name", default="", help="可选：覆盖 profile 中 dataset_name")
    parser.add_argument("--dataset_root", default="", help="可选：覆盖 profile 中 dataset_root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = _load_profiles(Path(args.profiles))
    if args.profile not in profiles:
        raise KeyError(f"profile 不存在: {args.profile}，可选: {sorted(profiles)}")

    config = dict(profiles[args.profile] or {})
    if args.dataset_name:
        config["dataset_name"] = args.dataset_name
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root

    cli_args = _to_cli_args(config)
    attention_script = PROJECT_ROOT / "src" / "fusion" / "train_fusion_attention.py"
    stacking_script = PROJECT_ROOT / "src" / "fusion" / "train_fusion_attention_stacking.py"

    if args.mode in ("attention", "all"):
        print("🚀 启动 attention 训练")
        _run(attention_script, cli_args)
    if args.mode in ("attention_stacking", "all"):
        print("🚀 启动 attention_stacking 训练")
        _run(stacking_script, cli_args)
    print("🏁 suite 执行完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
