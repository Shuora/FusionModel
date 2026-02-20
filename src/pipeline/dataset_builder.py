from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import yaml
from tqdm import tqdm

from feature_rgb import build_rgb_image, save_rgb_image
from pcap_session import (
    SessionRecord,
    collect_cic_pcaps,
    collect_ustc_pcaps,
    extract_session_records_from_pcap,
    normalize_session_bytes,
    session_sha1,
    split_class_files,
)


LOGGER = logging.getLogger("dataset_builder")

SessionFallbackStrategy = Literal["hash", "time", "none"]
SinglePcapPolicy = Literal["train_only", "test_only"]


@dataclass
class BuildStats:
    profile: str
    dataset_name: str
    source_dir: str
    byte_mode: str
    session_fallback_strategy: str
    single_pcap_policy: str
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
        raise ValueError(f"Invalid profiles file: {path}")
    profiles = data.get("profiles", data)
    if not isinstance(profiles, dict):
        raise ValueError(f"Invalid `profiles` section in: {path}")
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
        class_names = sorted(image_classes | pcap_classes)
        for class_name in class_names:
            image_dir = image_split / class_name
            pcap_dir = pcap_split / class_name
            image_count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0
            pcap_count = len(list(pcap_dir.glob("*.bin"))) if pcap_dir.exists() else 0
            if image_count != pcap_count:
                LOGGER.warning(
                    "Class count mismatch %s/%s: image=%s pcap=%s",
                    split,
                    class_name,
                    image_count,
                    pcap_count,
                )
            split_counts[split][class_name] = min(image_count, pcap_count)
    return split_counts


def _collect_pcaps(profile: dict) -> Dict[str, List[Path]]:
    dataset_type = str(profile.get("dataset_type", "")).strip().lower()
    source_dir = Path(profile["source_dir"]).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    if dataset_type == "ustc_flat":
        class_to_pcaps = collect_ustc_pcaps(source_dir)
    elif dataset_type == "cic_hierarchical":
        class_to_pcaps = collect_cic_pcaps(
            source_dir,
            include_majors=profile.get("include_majors"),
            label_map=profile.get("label_map"),
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if not class_to_pcaps:
        raise RuntimeError(f"No pcap found: source_dir={source_dir}, dataset_type={dataset_type}")
    return class_to_pcaps


def _parse_fallback_strategy(profile: dict) -> SessionFallbackStrategy:
    raw = str(profile.get("session_fallback_strategy", "hash")).strip().lower()
    allowed = {"hash", "time", "none"}
    if raw not in allowed:
        raise ValueError(f"Invalid session_fallback_strategy: {raw}. Allowed: {sorted(allowed)}")
    return raw  # type: ignore[return-value]


def _parse_single_pcap_policy(profile: dict) -> SinglePcapPolicy:
    raw = str(profile.get("single_pcap_policy", "train_only")).strip().lower()
    allowed = {"train_only", "test_only"}
    if raw not in allowed:
        raise ValueError(f"Invalid single_pcap_policy: {raw}. Allowed: {sorted(allowed)}")
    return raw  # type: ignore[return-value]


def _policy_to_split(single_pcap_policy: SinglePcapPolicy) -> str:
    return "Train" if single_pcap_policy == "train_only" else "Test"


def _compute_train_cut(total: int, train_ratio: float) -> int:
    if total <= 1:
        return total
    cut = int(round(total * float(train_ratio)))
    return max(1, min(total - 1, cut))


def _build_time_split_map(
    records: List[SessionRecord],
    split_ratio: float,
    *,
    single_pcap_policy: SinglePcapPolicy,
) -> Dict[str, str]:
    if not records:
        return {}
    ordered = sorted(records, key=lambda item: (item.first_ts, item.key.to_id()))
    if len(ordered) <= 1:
        fallback_split = _policy_to_split(single_pcap_policy)
        return {ordered[0].key.to_id(): fallback_split}
    cut = _compute_train_cut(len(ordered), split_ratio)
    assignment: Dict[str, str] = {}
    for idx, rec in enumerate(ordered):
        assignment[rec.key.to_id()] = "Train" if idx < cut else "Test"
    return assignment


def _iter_effective_splits(
    records: List[SessionRecord],
    *,
    split_name: str,
    split_ratio: float,
    target_length: int,
    session_fallback_strategy: SessionFallbackStrategy,
    single_pcap_policy: SinglePcapPolicy,
) -> List[Tuple[SessionRecord, str, bytes]]:
    if not records:
        return []

    time_assignment: Dict[str, str] = {}
    if split_name == "SESSION_SPLIT" and session_fallback_strategy == "time":
        time_assignment = _build_time_split_map(
            records,
            split_ratio=split_ratio,
            single_pcap_policy=single_pcap_policy,
        )

    result: List[Tuple[SessionRecord, str, bytes]] = []
    for record in records:
        normalized = normalize_session_bytes(record.payload, target_length=target_length)
        effective_split = split_name
        if split_name == "SESSION_SPLIT":
            if session_fallback_strategy == "hash":
                bucket = int(session_sha1(normalized)[:8], 16) / float(0xFFFFFFFF)
                effective_split = "Train" if bucket < split_ratio else "Test"
            elif session_fallback_strategy == "time":
                effective_split = time_assignment.get(record.key.to_id(), _policy_to_split(single_pcap_policy))
            else:
                effective_split = _policy_to_split(single_pcap_policy)
        result.append((record, effective_split, normalized))
    return result


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
        "session_fallback_strategy": stats.session_fallback_strategy,
        "single_pcap_policy": stats.single_pcap_policy,
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
    LOGGER.info("Build dataset profile=%s dataset=%s", profile_name, dataset_name)
    LOGGER.info("source_dir=%s", profile.get("source_dir"))
    LOGGER.info("output_dir=%s", dataset_dir)

    class_to_pcaps = _collect_pcaps(profile)
    split_ratio = float(profile.get("train_ratio", 0.8))
    seed = int(profile.get("seed", 42))
    allow_session_split_fallback = bool(profile.get("allow_session_split_fallback", True))
    session_fallback_strategy = _parse_fallback_strategy(profile)
    single_pcap_policy = _parse_single_pcap_policy(profile)
    byte_mode = str(profile.get("byte_mode", "payload")).strip().lower()
    merge_bidirectional = bool(profile.get("merge_bidirectional", True))
    sanitize_headers = bool(profile.get("sanitize_headers", True))
    min_chunk_bytes = int(profile.get("min_chunk_bytes", 1))
    min_session_bytes = int(profile.get("min_session_bytes", 1))
    target_length = int(profile.get("target_length", 784))
    image_size = int(profile.get("image_size", 28))

    LOGGER.info(
        "split_config allow_session_split_fallback=%s session_fallback_strategy=%s single_pcap_policy=%s train_ratio=%.4f seed=%s",
        allow_session_split_fallback,
        session_fallback_strategy,
        single_pcap_policy,
        split_ratio,
        seed,
    )

    splits = split_class_files(class_to_pcaps, train_ratio=split_ratio, seed=seed)
    stats = BuildStats(
        profile=profile_name,
        dataset_name=dataset_name,
        source_dir=str(profile.get("source_dir")),
        byte_mode=byte_mode,
        session_fallback_strategy=session_fallback_strategy,
        single_pcap_policy=single_pcap_policy,
    )

    split_iter: List[Tuple[str, str, Path]] = []
    for class_name in sorted(class_to_pcaps):
        total_pcaps = len(class_to_pcaps[class_name])
        if total_pcaps < 2:
            if allow_session_split_fallback and session_fallback_strategy in {"hash", "time"}:
                LOGGER.warning(
                    "Class %s has %s pcap(s), using SESSION_SPLIT strategy=%s",
                    class_name,
                    total_pcaps,
                    session_fallback_strategy,
                )
                for pcap_path in sorted(class_to_pcaps[class_name]):
                    split_iter.append((class_name, "SESSION_SPLIT", pcap_path))
                continue

            fixed_split = _policy_to_split(single_pcap_policy)
            LOGGER.warning(
                "Class %s has %s pcap(s), fallback disabled/effectively disabled; assigning all samples to %s",
                class_name,
                total_pcaps,
                fixed_split,
            )
            for pcap_path in sorted(class_to_pcaps[class_name]):
                split_iter.append((class_name, fixed_split, pcap_path))
            continue

        for split_name in ("Train", "Test"):
            for pcap_path in splits[class_name][split_name]:
                split_iter.append((class_name, split_name, pcap_path))

    dedup_seen: Dict[str, set[str]] = {}
    progress = tqdm(split_iter, desc=f"[{dataset_name}] pcap", ncols=110)
    for class_name, split_name, pcap_path in progress:
        stats.processed_pcap += 1
        try:
            records = extract_session_records_from_pcap(
                pcap_path,
                byte_mode=byte_mode,
                merge_bidirectional=merge_bidirectional,
                sanitize_headers=sanitize_headers,
                min_chunk_bytes=min_chunk_bytes,
                min_session_bytes=min_session_bytes,
            )
        except Exception as exc:
            LOGGER.warning("Failed to parse %s, err=%s", pcap_path, exc)
            continue

        if not records:
            continue

        effective_samples = _iter_effective_splits(
            records,
            split_name=split_name,
            split_ratio=split_ratio,
            target_length=target_length,
            session_fallback_strategy=session_fallback_strategy,
            single_pcap_policy=single_pcap_policy,
        )

        for record, effective_split, normalized in effective_samples:
            stats.processed_session += 1
            if not record.payload:
                stats.empty_session += 1
                continue

            dedup = dedup_seen.setdefault(f"{effective_split}/{class_name}", set())
            digest = session_sha1(normalized)
            if digest in dedup:
                stats.duplicate_session += 1
                continue
            dedup.add(digest)

            sample_id = _sanitize_name(f"{pcap_path.stem}.{record.key.to_id()}")
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
                record.payload,
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

    LOGGER.info("Done profile=%s", profile_name)
    LOGGER.info(
        "pcap=%s sessions=%s saved=%s dup=%s empty=%s skipped=%s",
        stats.processed_pcap,
        stats.processed_session,
        stats.saved_session,
        stats.duplicate_session,
        stats.empty_session,
        stats.skipped_existing,
    )
    LOGGER.info("summary=%s", report_path)
    LOGGER.info("log=%s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build USTC/CIC RGB+BIN fusion datasets.")
    parser.add_argument("--profile", required=True, help="Profile name, e.g. ustc / cic5_payload / cic5_fullpacket")
    parser.add_argument(
        "--profiles",
        default=str(Path("configs") / "dataset_profiles.yaml"),
        help="Profiles yaml path",
    )
    parser.add_argument(
        "--dataset_root",
        default="dataset",
        help="Output dataset root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image/bin files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles_path = Path(args.profiles).resolve()
    profiles = _load_profiles(profiles_path)
    if args.profile not in profiles:
        raise KeyError(f"Profile not found: {args.profile}. Available: {sorted(profiles)}")

    profile = dict(profiles[args.profile] or {})
    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    build_dataset(args.profile, profile, dataset_root=dataset_root, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
