from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def _snapshot_files(output_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    对输出目录做快照，返回:
    rel_path -> (size, mtime_ns)
    """
    snapshot: Dict[str, Tuple[int, int]] = {}
    if not output_dir.exists():
        return snapshot
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir)
        if rel.parts and rel.parts[0].lower() == "archive":
            continue
        stat = path.stat()
        snapshot[str(rel)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def _diff_snapshot(
    before: Dict[str, Tuple[int, int]],
    after: Dict[str, Tuple[int, int]],
) -> Set[str]:
    touched: Set[str] = set()
    for rel, info in after.items():
        if rel not in before or before[rel] != info:
            touched.add(rel)
    return touched


def _archive_outputs(
    *,
    output_dir: Path,
    touched_files: Set[str],
    profile: str,
    mode: str,
    archive_dir: Path | None = None,
    archive_tag: str = "",
    move_files: bool = False,
) -> Path:
    if not touched_files:
        raise ValueError("没有可归档的新产物")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = archive_dir if archive_dir is not None else (output_dir / "archive")
    run_tag = archive_tag.strip() or f"{profile}_{mode}_{ts}"
    archive_path = archive_root / run_tag
    archive_path.mkdir(parents=True, exist_ok=True)

    copied = []
    for rel in sorted(touched_files):
        src = output_dir / rel
        if not src.exists() or not src.is_file():
            continue
        dst = archive_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move_files:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        copied.append(rel)

    manifest = {
        "profile": profile,
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "archive_dir": str(archive_path),
        "move_files": bool(move_files),
        "file_count": len(copied),
        "files": copied,
    }
    with (archive_path / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return archive_path


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
    parser.add_argument("--no_archive", action="store_true", help="不自动归档本次运行输出")
    parser.add_argument("--archive_dir", default="", help="归档根目录，默认 <output_dir>/archive")
    parser.add_argument("--archive_tag", default="", help="归档目录名，默认 <profile>_<mode>_<timestamp>")
    parser.add_argument("--archive_move", action="store_true", help="归档时移动文件（默认复制）")
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
    output_dir = Path(str(config.get("output_dir", "outputs")))
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    attention_script = PROJECT_ROOT / "src" / "fusion" / "train_fusion_attention.py"
    stacking_script = PROJECT_ROOT / "src" / "fusion" / "train_fusion_attention_stacking.py"
    touched_files: Set[str] = set()

    if args.mode in ("attention", "all"):
        print("🚀 启动 attention 训练")
        before = _snapshot_files(output_dir)
        _run(attention_script, cli_args)
        after = _snapshot_files(output_dir)
        touched_files.update(_diff_snapshot(before, after))
    if args.mode in ("attention_stacking", "all"):
        print("🚀 启动 attention_stacking 训练")
        before = _snapshot_files(output_dir)
        _run(stacking_script, cli_args)
        after = _snapshot_files(output_dir)
        touched_files.update(_diff_snapshot(before, after))

    if not args.no_archive:
        archive_dir = Path(args.archive_dir).resolve() if args.archive_dir else None
        archive_path = _archive_outputs(
            output_dir=output_dir,
            touched_files=touched_files,
            profile=args.profile,
            mode=args.mode,
            archive_dir=archive_dir,
            archive_tag=args.archive_tag,
            move_files=bool(args.archive_move),
        )
        print(f"🗂️ 已归档本次输出到: {archive_path}")
    else:
        print("ℹ️ 已跳过自动归档（--no_archive）")

    print("🏁 suite 执行完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
