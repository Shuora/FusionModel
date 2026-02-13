"""Preprocess raw pcap data into dual-branch training inputs.

Pipeline:
1) session segmentation via SplitCap (default) or python fallback,
2) cleaning (drop empty + deduplicate),
3) length unification to 784 bytes,
4) RGB conversion (R paper-style + custom G/B),
5) optional temporal exports (.npy/.pt), plus rich preprocessing logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from rgb_builder import build_rgb_image
from session_postprocess import (
    extract_payload_from_session_pcap,
    make_sample_stem,
    sha1_hex,
    split_sessions_python_fallback,
    unify_length,
)
from splitcap_runner import default_splitcap_path, run_splitcap

LOGGER = logging.getLogger("preprocess_splitcap_rgb")
PCAP_EXTS = {".pcap", ".pcapng"}
SUPPORTED_TEMPORAL_FORMATS = {"bin", "npy", "pt"}


@dataclass(frozen=True)
class SourcePcap:
    path: Path
    label: str


@dataclass
class Summary:
    total_source_pcaps: int = 0
    total_sessions_seen: int = 0
    total_sessions_written: int = 0
    empty_sessions_dropped: int = 0
    duplicate_sessions_dropped: int = 0
    splitcap_failures: int = 0
    python_fallback_used: int = 0
    source_errors: int = 0


def parse_temporal_formats(value: str) -> List[str]:
    formats = [v.strip().lower() for v in str(value).split(",") if v.strip()]
    if not formats:
        formats = ["bin"]

    invalid = [v for v in formats if v not in SUPPORTED_TEMPORAL_FORMATS]
    if invalid:
        raise ValueError(f"Unsupported temporal formats: {invalid}. Supported: {sorted(SUPPORTED_TEMPORAL_FORMATS)}")

    if "bin" not in formats:
        LOGGER.warning("'bin' is required for training. Automatically adding it to temporal formats.")
        formats.append("bin")

    return sorted(set(formats))


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def discover_pcaps(input_root: Path, label_mode: str) -> List[SourcePcap]:
    files = sorted([p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in PCAP_EXTS])
    entries: List[SourcePcap] = []
    for p in files:
        label = resolve_label(input_root=input_root, pcap_path=p, label_mode=label_mode)
        entries.append(SourcePcap(path=p, label=label))
    return entries


def resolve_label(input_root: Path, pcap_path: Path, label_mode: str) -> str:
    rel = pcap_path.relative_to(input_root)
    parts = rel.parts

    if label_mode == "file_stem":
        return pcap_path.stem

    if label_mode == "group":
        if len(parts) >= 2:
            return parts[0]
        return pcap_path.stem

    if label_mode == "family":
        if len(parts) >= 3:
            return parts[1]
        if len(parts) >= 2:
            return parts[0]
        return pcap_path.stem

    # auto
    if len(parts) >= 3:
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return pcap_path.stem


def split_sources_by_pcap(entries: Sequence[SourcePcap], train_ratio: float, seed: int) -> Dict[Path, str]:
    by_label: Dict[str, List[SourcePcap]] = {}
    for e in entries:
        by_label.setdefault(e.label, []).append(e)

    rng = random.Random(seed)
    split_map: Dict[Path, str] = {}
    for label, items in by_label.items():
        items_copy = list(items)
        rng.shuffle(items_copy)

        if len(items_copy) <= 1:
            cut = len(items_copy)
        else:
            cut = int(len(items_copy) * train_ratio)
            cut = max(1, min(cut, len(items_copy) - 1))

        for i, item in enumerate(items_copy):
            split_map[item.path] = "Train" if i < cut else "Test"

        LOGGER.info("Split label=%s: train=%s test=%s total=%s", label, cut, len(items_copy) - cut, len(items_copy))

    return split_map


def write_temporal_exports(base_path: Path, unified: bytes, temporal_formats: Iterable[str]) -> None:
    arr = np.frombuffer(unified, dtype=np.uint8)
    fmts = set(temporal_formats)

    if "bin" in fmts:
        with base_path.with_suffix(".bin").open("wb") as f:
            f.write(unified)

    if "npy" in fmts:
        np.save(str(base_path.with_suffix(".npy")), arr)

    if "pt" in fmts:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for PT export. Install torch or remove 'pt' from --temporal_formats") from exc
        tensor = torch.from_numpy(arr.copy())
        torch.save(tensor, str(base_path.with_suffix(".pt")))


def process_payload(
    payload: bytes,
    *,
    source_pcap: Path,
    split: str,
    label: str,
    session_idx: int,
    max_len: int,
    temporal_formats: Sequence[str],
    output_root: Path,
    dedup_enabled: bool,
    seen_hashes: Set[str],
    label_split_counts: Dict[Tuple[str, str], int],
    summary: Summary,
) -> None:
    summary.total_sessions_seen += 1

    if not payload:
        summary.empty_sessions_dropped += 1
        return

    digest = sha1_hex(payload)
    if dedup_enabled and digest in seen_hashes:
        summary.duplicate_sessions_dropped += 1
        return

    if dedup_enabled:
        seen_hashes.add(digest)

    unified = unify_length(payload, max_len=max_len)
    stem = make_sample_stem(source_pcap, session_idx, payload)

    img_dir = output_root / "image_data" / split / label
    pcap_dir = output_root / "pcap_data" / split / label
    temporal_dir = output_root / "temporal_data" / split / label

    img_dir.mkdir(parents=True, exist_ok=True)
    pcap_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / f"{stem}.png"
    pcap_base = pcap_dir / stem
    temporal_base = temporal_dir / stem

    write_temporal_exports(pcap_base, unified, ["bin"])
    optional_temporal = [f for f in temporal_formats if f != "bin"]
    if optional_temporal:
        write_temporal_exports(temporal_base, unified, optional_temporal)

    img = build_rgb_image(unified_784=unified, raw_payload=payload)
    tmp_img = img_path.with_suffix(".tmp.png")
    img.save(tmp_img, format="PNG")
    tmp_img.replace(img_path)

    label_split_counts[(split, label)] = label_split_counts.get((split, label), 0) + 1
    summary.total_sessions_written += 1


def export_summary(
    artifact_dir: Path,
    summary: Summary,
    *,
    run_config: dict,
    label_split_counts: Dict[Tuple[str, str], int],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_path = artifact_dir / "preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        payload = {
            "config": run_config,
            "summary": {
                "total_source_pcaps": summary.total_source_pcaps,
                "total_sessions_seen": summary.total_sessions_seen,
                "total_sessions_written": summary.total_sessions_written,
                "empty_sessions_dropped": summary.empty_sessions_dropped,
                "duplicate_sessions_dropped": summary.duplicate_sessions_dropped,
                "splitcap_failures": summary.splitcap_failures,
                "python_fallback_used": summary.python_fallback_used,
                "source_errors": summary.source_errors,
            },
        }
        json.dump(payload, f, ensure_ascii=False, indent=2)

    dist_path = artifact_dir / "label_distribution.csv"
    with dist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "count"])
        for (split, label), cnt in sorted(label_split_counts.items(), key=lambda x: (x[0][0], x[0][1])):
            writer.writerow([split, label, cnt])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SplitCap-based preprocessing for CharBERT + MobileViT")
    p.add_argument("--input_root", required=True, help="Input root containing raw pcap files")
    p.add_argument("--output_root", required=True, help="Output dataset root")
    p.add_argument("--splitcap_exe", default="", help="Path to SplitCap executable")
    p.add_argument("--splitcap_mode", choices=["external", "auto", "python"], default="external")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_mode", choices=["auto", "group", "family", "file_stem"], default="auto")
    p.add_argument("--max_len", type=int, default=784)
    p.add_argument("--temporal_formats", default="bin", help="Comma separated: bin,npy,pt")
    p.add_argument("--no_dedup", action="store_true", help="Disable SHA1 deduplication")
    p.add_argument("--save_temp_sessions", action="store_true", help="Keep temporary splitcap sessions")
    p.add_argument("--sanitize_headers", action="store_true", help="Reserved for full-packet mode; no-op in payload-only mode")
    p.add_argument("--artifact_root", default="outputs/preprocess", help="Where preprocess logs/summaries are stored")
    p.add_argument("--run_name", default="", help="Optional run id override")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input_root does not exist or is not a directory: {input_root}")

    run_id = args.run_name.strip() if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = artifact_root / run_id
    configure_logging(artifact_dir / "preprocess.log")

    temporal_formats = parse_temporal_formats(args.temporal_formats)
    if args.sanitize_headers:
        LOGGER.warning("--sanitize_headers is currently a no-op in payload-only preprocessing mode.")

    splitcap_exe = Path(args.splitcap_exe).resolve() if args.splitcap_exe else default_splitcap_path()
    LOGGER.info("Using SplitCap executable: %s", splitcap_exe)

    entries = discover_pcaps(input_root, args.label_mode)
    if not entries:
        LOGGER.warning("No pcap files found under input_root: %s", input_root)
        return

    split_map = split_sources_by_pcap(entries, args.train_ratio, args.seed)

    summary = Summary(total_source_pcaps=len(entries))
    seen_hashes: Set[str] = set()
    label_split_counts: Dict[Tuple[str, str], int] = {}

    tmp_root = artifact_dir / "tmp_splitcap_sessions"
    tmp_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Start preprocessing: source_pcaps=%s", len(entries))
    for idx, item in enumerate(entries, start=1):
        split = split_map[item.path]
        LOGGER.info("[%s/%s] Processing pcap=%s label=%s split=%s", idx, len(entries), item.path.name, item.label, split)

        payloads: List[bytes] = []
        try:
            if args.splitcap_mode in ("external", "auto"):
                pcap_tmp_dir = tmp_root / f"{item.path.stem}_{idx:06d}"
                result = run_splitcap(
                    splitcap_exe=splitcap_exe,
                    input_pcap=item.path,
                    output_dir=pcap_tmp_dir,
                    split_group="session",
                    output_filetype="pcap",
                    delete_previous=True,
                )

                if result.success:
                    for session_file in result.session_files:
                        if session_file.suffix.lower() in PCAP_EXTS:
                            payload = extract_payload_from_session_pcap(session_file)
                        else:
                            payload = session_file.read_bytes()
                        payloads.append(payload)
                else:
                    summary.splitcap_failures += 1
                    LOGGER.warning(
                        "SplitCap failed for %s (rc=%s). stderr=%s",
                        item.path,
                        result.returncode,
                        (result.stderr or "").strip(),
                    )

                if not args.save_temp_sessions and pcap_tmp_dir.exists():
                    shutil.rmtree(pcap_tmp_dir, ignore_errors=True)

            if (not payloads) and args.splitcap_mode in ("auto", "python"):
                fallback_sessions = split_sessions_python_fallback(item.path)
                payloads = list(fallback_sessions.values())
                summary.python_fallback_used += 1

            for session_idx, payload in enumerate(payloads, start=1):
                process_payload(
                    payload,
                    source_pcap=item.path,
                    split=split,
                    label=item.label,
                    session_idx=session_idx,
                    max_len=int(args.max_len),
                    temporal_formats=temporal_formats,
                    output_root=output_root,
                    dedup_enabled=(not args.no_dedup),
                    seen_hashes=seen_hashes,
                    label_split_counts=label_split_counts,
                    summary=summary,
                )

        except Exception as exc:
            summary.source_errors += 1
            LOGGER.exception("Failed processing source pcap %s: %s", item.path, exc)

    run_config = {
        "run_id": run_id,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splitcap_mode": args.splitcap_mode,
        "splitcap_exe": str(splitcap_exe),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "label_mode": args.label_mode,
        "max_len": int(args.max_len),
        "temporal_formats": temporal_formats,
        "dedup_enabled": (not args.no_dedup),
        "save_temp_sessions": bool(args.save_temp_sessions),
    }

    export_summary(
        artifact_dir=artifact_dir,
        summary=summary,
        run_config=run_config,
        label_split_counts=label_split_counts,
    )

    LOGGER.info(
        "Preprocess done. written=%s seen=%s empty_drop=%s dup_drop=%s splitcap_fail=%s fallback=%s errors=%s",
        summary.total_sessions_written,
        summary.total_sessions_seen,
        summary.empty_sessions_dropped,
        summary.duplicate_sessions_dropped,
        summary.splitcap_failures,
        summary.python_fallback_used,
        summary.source_errors,
    )
    LOGGER.info("Dataset output root: %s", output_root)
    LOGGER.info("Artifact output root: %s", artifact_dir)


if __name__ == "__main__":
    main()
