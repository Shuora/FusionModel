from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
from tqdm import tqdm

from feature_rgb import build_rgb_image, save_rgb_image
from pcap_session import (
    collect_cic_pcaps,
    collect_ustc_pcaps,
    extract_sessions_from_pcap,
    normalize_session_bytes,
    session_sha1,
    split_class_files,
)


LOGGER = logging.getLogger("dataset_builder")


@dataclass
class BuildStats:
    profile: str
    dataset_name: str
    source_dir: str
    byte_mode: str
    processed_pcap: int = 0
    processed_session: int = 0
    saved_session: int = 0
    duplicate_session: int = 0
    empty_session: int = 0
    skipped_existing: int = 0
    split_counts: Dict[str, Dict[str, int]] | None = None


def _setup_logging(dataset_name: str, output_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"preprocess_{dataset_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def _load_profiles(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    profiles = data.get("profiles", data)
    if not isinstance(profiles, dict):
        raise ValueError(f"profiles 字段格式错误: {path}")
    return profiles


def _sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _validate_pairs(dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    split_counts: Dict[str, Dict[str, int]] = {}
    for split in ("Train", "Test"):
        split_counts[split] = {}
        image_split = dataset_dir / "image_data" / split
        pcap_split = dataset_dir / "pcap_data" / split
        image_classes = {p.name for p in image_split.iterdir() if p.is_dir()} if image_split.exists() else set()
        pcap_classes = {p.name for p in pcap_split.iterdir() if p.is_dir()} if pcap_split.exists() else set()
        class_names = sorted(
            image_classes | pcap_classes
        )
        for class_name in class_names:
            image_dir = image_split / class_name
            pcap_dir = pcap_split / class_name
            image_count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0
            pcap_count = len(list(pcap_dir.glob("*.bin"))) if pcap_dir.exists() else 0
            if image_count != pcap_count:
                LOGGER.warning("⚠️ 类别计数不一致 %s/%s: image=%s pcap=%s", split, class_name, image_count, pcap_count)
            split_counts[split][class_name] = min(image_count, pcap_count)
    return split_counts


def _collect_pcaps(profile: dict) -> Dict[str, List[Path]]:
    dataset_type = str(profile.get("dataset_type", "")).strip().lower()
    source_dir = Path(profile["source_dir"]).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir 不存在: {source_dir}")

    if dataset_type == "ustc_flat":
        class_to_pcaps = collect_ustc_pcaps(source_dir)
    elif dataset_type == "cic_hierarchical":
        class_to_pcaps = collect_cic_pcaps(
            source_dir,
            include_majors=profile.get("include_majors"),
            label_map=profile.get("label_map"),
        )
    else:
        raise ValueError(f"不支持的 dataset_type: {dataset_type}")

    if not class_to_pcaps:
        raise RuntimeError(f"未找到可处理 pcap: source_dir={source_dir}, dataset_type={dataset_type}")
    return class_to_pcaps


def _write_summary(stats: BuildStats, dataset_dir: Path, log_path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_dir = dataset_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"preprocess_summary_{ts}.json"
    payload = {
        "profile": stats.profile,
        "dataset_name": stats.dataset_name,
        "source_dir": stats.source_dir,
        "byte_mode": stats.byte_mode,
        "processed_pcap": stats.processed_pcap,
        "processed_session": stats.processed_session,
        "saved_session": stats.saved_session,
        "duplicate_session": stats.duplicate_session,
        "empty_session": stats.empty_session,
        "skipped_existing": stats.skipped_existing,
        "split_counts": stats.split_counts or {},
        "log_file": str(log_path),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return report_path


def build_dataset(profile_name: str, profile: dict, dataset_root: Path, overwrite: bool) -> None:
    dataset_name = str(profile.get("dataset_name", profile_name))
    dataset_dir = dataset_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log_path = _setup_logging(dataset_name, dataset_dir)
    LOGGER.info("🚀 开始预处理 profile=%s dataset=%s", profile_name, dataset_name)
    LOGGER.info("📁 source_dir=%s", profile.get("source_dir"))
    LOGGER.info("📁 output_dir=%s", dataset_dir)

    class_to_pcaps = _collect_pcaps(profile)
    split_ratio = float(profile.get("train_ratio", 0.8))
    seed = int(profile.get("seed", 42))
    allow_session_split_fallback = bool(profile.get("allow_session_split_fallback", True))
    byte_mode = str(profile.get("byte_mode", "payload")).strip().lower()
    merge_bidirectional = bool(profile.get("merge_bidirectional", True))
    sanitize_headers = bool(profile.get("sanitize_headers", True))
    min_chunk_bytes = int(profile.get("min_chunk_bytes", 1))
    min_session_bytes = int(profile.get("min_session_bytes", 1))
    target_length = int(profile.get("target_length", 784))
    image_size = int(profile.get("image_size", 28))

    splits = split_class_files(class_to_pcaps, train_ratio=split_ratio, seed=seed)
    stats = BuildStats(
        profile=profile_name,
        dataset_name=dataset_name,
        source_dir=str(profile.get("source_dir")),
        byte_mode=byte_mode,
    )

    split_iter = []
    for class_name in sorted(class_to_pcaps):
        total_pcaps = len(class_to_pcaps[class_name])
        if allow_session_split_fallback and total_pcaps < 2:
            LOGGER.warning("⚠️ 类别 %s 仅有 %s 个 pcap，启用 session 级回退切分", class_name, total_pcaps)
            for pcap_path in sorted(class_to_pcaps[class_name]):
                split_iter.append((class_name, "SESSION_SPLIT", pcap_path))
            continue
        for split_name in ("Train", "Test"):
            for pcap_path in splits[class_name][split_name]:
                split_iter.append((class_name, split_name, pcap_path))

    dedup_seen: Dict[str, set[str]] = {}
    progress = tqdm(split_iter, desc=f"[{dataset_name}] pcap", ncols=110)
    for class_name, split_name, pcap_path in progress:
        stats.processed_pcap += 1
        try:
            sessions = extract_sessions_from_pcap(
                pcap_path,
                byte_mode=byte_mode,
                merge_bidirectional=merge_bidirectional,
                sanitize_headers=sanitize_headers,
                min_chunk_bytes=min_chunk_bytes,
                min_session_bytes=min_session_bytes,
            )
        except Exception as exc:
            LOGGER.warning("❌ 解析失败: %s, err=%s", pcap_path, exc)
            continue

        if not sessions:
            continue

        for sess_key, sess_bytes in sessions.items():
            stats.processed_session += 1
            if not sess_bytes:
                stats.empty_session += 1
                continue

            normalized = normalize_session_bytes(sess_bytes, target_length=target_length)
            effective_split = split_name
            if split_name == "SESSION_SPLIT":
                bucket = int(session_sha1(normalized)[:8], 16) / float(0xFFFFFFFF)
                effective_split = "Train" if bucket < split_ratio else "Test"

            dedup = dedup_seen.setdefault(f"{effective_split}/{class_name}", set())
            digest = session_sha1(normalized)
            if digest in dedup:
                stats.duplicate_session += 1
                continue
            dedup.add(digest)

            sample_id = _sanitize_name(f"{pcap_path.stem}.{sess_key.to_id()}")
            pcap_out_dir = dataset_dir / "pcap_data" / effective_split / class_name
            image_out_dir = dataset_dir / "image_data" / effective_split / class_name
            pcap_out_dir.mkdir(parents=True, exist_ok=True)
            image_out_dir.mkdir(parents=True, exist_ok=True)
            bin_out = pcap_out_dir / f"{sample_id}.bin"
            img_out = image_out_dir / f"{sample_id}.png"

            if not overwrite and bin_out.exists() and img_out.exists():
                stats.skipped_existing += 1
                continue

            with bin_out.open("wb") as f:
                f.write(normalized)

            rgb = build_rgb_image(
                sess_bytes,
                target_length=target_length,
                image_size=image_size,
            )
            save_rgb_image(rgb, img_out)
            stats.saved_session += 1

        progress.set_postfix(
            cls=class_name,
            split=split_name,
            saved=stats.saved_session,
            dup=stats.duplicate_session,
            refresh=False,
        )

    stats.split_counts = _validate_pairs(dataset_dir)
    report_path = _write_summary(stats, dataset_dir, log_path)

    LOGGER.info("✅ 预处理完成 profile=%s", profile_name)
    LOGGER.info("📊 pcap=%s sessions=%s saved=%s dup=%s empty=%s skipped=%s",
                stats.processed_pcap,
                stats.processed_session,
                stats.saved_session,
                stats.duplicate_session,
                stats.empty_session,
                stats.skipped_existing)
    LOGGER.info("🧾 summary=%s", report_path)
    LOGGER.info("📝 log=%s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 USTC/CIC 的 RGB+Bin 融合数据集")
    parser.add_argument("--profile", required=True, help="配置名称，例如 ustc / cic5_payload / cic5_fullpacket")
    parser.add_argument(
        "--profiles",
        default=str(Path("configs") / "dataset_profiles.yaml"),
        help="配置文件路径（YAML）",
    )
    parser.add_argument(
        "--dataset_root",
        default="dataset",
        help="输出数据集根目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在样本（默认跳过已存在样本）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles_path = Path(args.profiles).resolve()
    profiles = _load_profiles(profiles_path)
    if args.profile not in profiles:
        raise KeyError(f"profile 不存在: {args.profile}. 可用: {sorted(profiles)}")

    profile = dict(profiles[args.profile] or {})
    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    build_dataset(args.profile, profile, dataset_root=dataset_root, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
