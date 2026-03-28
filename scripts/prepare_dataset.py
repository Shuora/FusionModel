from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from shutil import which
from typing import Callable, Sequence

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fusion_malicious.data.cache import write_cached_sample
from fusion_malicious.data.cleaning import (
    anonymize_session_pcap,
    fingerprint_session_bytes,
)
from fusion_malicious.data.etbert_tokens import load_etbert_tokenizer, tokenize_session_bytes
from fusion_malicious.data.image_features import bytes_to_rgb_image
from fusion_malicious.data.manifest import build_manifest_dataframe
from fusion_malicious.data.records import SessionRecord
from fusion_malicious.data.session_bytes import normalize_session_bytes, read_session_bytes
from fusion_malicious.data.split import stratified_split_records
from fusion_malicious.data.splitcap import build_splitcap_command

TASK_DATASETS = {
    "binary": {"ISCX-VPN-NonVPN-2016", "MTA", "MFCP"},
    "mta": {"MTA"},
    "mfcp": {"MFCP"},
    "ustc": {"USTC-TFC2016"},
}

PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
SPLITCAP_DONE_FLAG = ".splitcap.done"
DEFAULT_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_PROGRESS_EVERY = 250
_WORKER_TOKENIZER = None
_WORKER_TOKENIZER_MODEL = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cached multimodal samples from raw traffic captures.")
    parser.add_argument("--task", choices=sorted(TASK_DATASETS), required=True)
    parser.add_argument("--source-root", type=Path, default=repo_root / "SourceData")
    parser.add_argument("--output-root", type=Path, default=repo_root / "dataset")
    parser.add_argument("--splitcap-exe", type=Path, default=repo_root / "Tools" / "SplitCap.exe")
    parser.add_argument("--splitcap-launcher", type=str, default="mono")
    parser.add_argument("--editcap-path", type=str, default="editcap")
    parser.add_argument("--skip-splitcap", action="store_true")
    parser.add_argument("--resume-splitcap", dest="resume_splitcap", action="store_true", default=True)
    parser.add_argument("--no-resume-splitcap", dest="resume_splitcap", action="store_false")
    parser.add_argument("--include-path", action="append", default=[])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.2)
    return parser


def _normalize_include_paths(include_paths: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for fragment in include_paths or []:
        cleaned = fragment.replace("\\", "/").strip("/")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def discover_capture_files(
    source_root: Path,
    task: str,
    include_paths: Sequence[str] | None = None,
) -> list[Path]:
    allowed = TASK_DATASETS[task]
    filters = _normalize_include_paths(include_paths)
    files: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pcap", ".pcapng"}:
            continue
        if not any(part in allowed for part in path.parts):
            continue
        if filters:
            relative = path.relative_to(source_root).as_posix()
            if not any(fragment in relative for fragment in filters):
                continue
        files.append(path)
    return sorted(files)


def run_splitcap(
    splitcap_exe: Path,
    input_pcap: Path,
    output_dir: Path,
    *,
    launcher: list[str] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_splitcap_command(splitcap_exe, input_pcap, output_dir, launcher=launcher)
    subprocess.run(command, check=True)
    return sorted(output_dir.rglob("*.pcap"))


def _looks_like_pcapng(raw_path: Path) -> bool:
    if raw_path.suffix.lower() == ".pcapng":
        return True
    try:
        with raw_path.open("rb") as handle:
            return handle.read(4) == PCAPNG_MAGIC
    except OSError:
        return False


def prepare_splitcap_input(
    raw_path: Path,
    *,
    working_dir: Path,
    editcap_path: str,
) -> Path:
    """Convert pcapng (including misnamed .pcap files) to pcap for SplitCap."""
    if not _looks_like_pcapng(raw_path):
        return raw_path

    editcap_bin = which(editcap_path)
    if editcap_bin is None:
        raise RuntimeError(f"editcap executable '{editcap_path}' was not found in PATH.")

    working_dir.mkdir(parents=True, exist_ok=True)
    converted_path = working_dir / f"{raw_path.stem}__splitcap_input.pcap"
    subprocess.run(
        [editcap_bin, "-F", "pcap", str(raw_path), str(converted_path)],
        check=True,
    )
    return converted_path


def collect_session_paths(
    raw_paths: list[Path],
    *,
    task: str,
    output_root: Path,
    splitcap_exe: Path,
    splitcap_launcher: list[str] | None,
    editcap_path: str,
    skip_splitcap: bool,
    resume_splitcap: bool,
) -> list[Path]:
    if skip_splitcap:
        return raw_paths

    session_root = output_root / task / "sessions_raw"
    session_paths: list[Path] = []
    failed_inputs: list[Path] = []

    for raw_path in raw_paths:
        try:
            relative = raw_path.relative_to(repo_root / "SourceData")
        except ValueError:
            relative = Path(raw_path.name)

        target_dir = session_root / relative.with_suffix("")
        done_flag = target_dir / SPLITCAP_DONE_FLAG

        if resume_splitcap and done_flag.exists():
            existing = sorted(target_dir.rglob("*.pcap"))
            if existing:
                session_paths.extend(existing)
                continue

        try:
            prepared_input = prepare_splitcap_input(
                raw_path,
                working_dir=target_dir,
                editcap_path=editcap_path,
            )
            split_sessions = run_splitcap(
                splitcap_exe,
                prepared_input,
                target_dir,
                launcher=splitcap_launcher,
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[WARN] SplitCap failed for {raw_path}: {exc}")
            failed_inputs.append(raw_path)
            continue

        done_flag.parent.mkdir(parents=True, exist_ok=True)
        done_flag.write_text("ok\n", encoding="utf-8")
        session_paths.extend(split_sessions)

    if failed_inputs:
        print(f"[WARN] SplitCap failed on {len(failed_inputs)} input file(s); rerun to continue from checkpoints.")

    return session_paths


def _task_root(output_root: Path, task: str) -> Path:
    return output_root / task


def inspect_session_payload(source_path: str) -> dict[str, object]:
    payload_bytes = read_session_bytes(source_path)
    if not payload_bytes:
        return {"empty": True, "fingerprint": None}
    return {
        "empty": False,
        "fingerprint": fingerprint_session_bytes(payload_bytes),
    }


def prepare_cached_rows(
    session_paths: list[Path],
    *,
    task: str,
    output_root: Path,
    num_workers: int = 1,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    log_fn: Callable[[str], None] = print,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    manifest = build_manifest_dataframe(session_paths, task_name=task)
    task_root = _task_root(output_root, task)
    cleaned_root = task_root / "sessions_clean"
    cache_root = task_root / "cache"
    seen_fingerprints: set[str] = set()
    ready_rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, object]] = []
    stats = {
        "empty": 0,
        "duplicates": 0,
        "clean_hits": 0,
        "cache_hits": 0,
    }
    total = len(manifest.index)
    started_at = time.perf_counter()

    if total:
        log_fn(f"[plan] scanning {total} session(s)")

    rows = manifest.to_dict(orient="records")
    source_paths = [str(row["source_path"]) for row in rows]
    worker_count = max(1, num_workers)
    chunksize = max(1, total // (worker_count * 8)) if total and worker_count > 1 else 1

    def process_inspection(order: int, row: dict[str, object], inspection: dict[str, object]) -> None:
        source_path = Path(str(row["source_path"]))
        if bool(inspection["empty"]):
            stats["empty"] += 1
        else:
            fingerprint = str(inspection["fingerprint"])
            if fingerprint in seen_fingerprints:
                stats["duplicates"] += 1
            else:
                seen_fingerprints.add(fingerprint)
                sample_id = str(row["sample_id"])
                cleaned_path = cleaned_root / f"{sample_id}.pcap"
                cache_path = cache_root / f"{sample_id}.npz"
                row["order"] = order
                row["cleaned_path"] = str(cleaned_path)
                row["cache_path"] = str(cache_path)
                row["skip_cleaning"] = cleaned_path.exists()
                if row["skip_cleaning"]:
                    stats["clean_hits"] += 1

                if cleaned_path.exists() and cache_path.exists():
                    stats["cache_hits"] += 1
                    ready_rows.append(row)
                else:
                    pending_rows.append(row)

        scanned = order + 1
        if progress_every > 0 and (scanned % progress_every == 0 or scanned == total):
            elapsed = time.perf_counter() - started_at
            log_fn(
                "[plan] scanned "
                f"{scanned}/{total} ready={len(ready_rows)} pending={len(pending_rows)} "
                f"cache_hits={stats['cache_hits']} clean_hits={stats['clean_hits']} "
                f"duplicates={stats['duplicates']} empty={stats['empty']} "
                f"elapsed={elapsed:.1f}s"
            )

    if worker_count == 1:
        for order, (row, inspection) in enumerate(zip(rows, map(inspect_session_payload, source_paths))):
            process_inspection(order, row, inspection)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            inspections = executor.map(inspect_session_payload, source_paths, chunksize=chunksize)
            for order, (row, inspection) in enumerate(zip(rows, inspections)):
                process_inspection(order, row, inspection)

    return ready_rows, pending_rows, stats


def _load_worker_tokenizer(model_name_or_path: str):
    global _WORKER_TOKENIZER, _WORKER_TOKENIZER_MODEL
    if _WORKER_TOKENIZER is None or _WORKER_TOKENIZER_MODEL != model_name_or_path:
        _WORKER_TOKENIZER = load_etbert_tokenizer(model_name_or_path)
        _WORKER_TOKENIZER_MODEL = model_name_or_path
    return _WORKER_TOKENIZER


def process_postprocess_row(
    row: dict[str, object],
    tokenizer_model: str,
    tokenizer_max_length: int,
    seed: int,
) -> dict[str, object]:
    source_path = Path(str(row["source_path"]))
    cleaned_path = Path(str(row["cleaned_path"]))
    cache_path = Path(str(row["cache_path"]))
    if not bool(row.get("skip_cleaning")) or not cleaned_path.exists():
        anonymize_session_pcap(source_path, cleaned_path, seed=seed)

    payload_bytes = read_session_bytes(cleaned_path)
    normalized = normalize_session_bytes(payload_bytes, size=784).tobytes()
    image = bytes_to_rgb_image(normalized, size=784)
    tokenizer = _load_worker_tokenizer(tokenizer_model)
    token_text, input_ids, attention_mask = tokenize_session_bytes(
        normalized,
        tokenizer=tokenizer,
        max_length=tokenizer_max_length,
    )
    write_cached_sample(
        cache_path=cache_path,
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        label=int(row["label_id"]),
        token_text=token_text,
    )
    updated = dict(row)
    updated["cleaned_path"] = str(cleaned_path)
    updated["cache_path"] = str(cache_path)
    return updated


def run_postprocess_tasks(
    pending_rows: list[dict[str, object]],
    *,
    tokenizer_model: str,
    tokenizer_max_length: int,
    seed: int,
    num_workers: int,
    progress_every: int,
    log_fn: Callable[[str], None] = print,
    worker_fn: Callable[[dict[str, object], str, int, int], dict[str, object]] = process_postprocess_row,
) -> list[dict[str, object]]:
    if not pending_rows:
        return []

    total = len(pending_rows)
    results: list[dict[str, object]] = []
    worker = partial(
        worker_fn,
        tokenizer_model=tokenizer_model,
        tokenizer_max_length=tokenizer_max_length,
        seed=seed,
    )

    def consume(iterator) -> None:
        for completed, row in enumerate(iterator, start=1):
            results.append(row)
            if progress_every > 0 and (completed % progress_every == 0 or completed == total):
                log_fn(f"[cache] completed {completed}/{total}")

    worker_count = max(1, num_workers)
    if worker_count == 1:
        consume(map(worker, pending_rows))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            consume(executor.map(worker, pending_rows))

    results.sort(key=lambda row: int(row["order"]))
    return results


def build_cached_manifest(
    session_paths: list[Path],
    *,
    task: str,
    output_root: Path,
    tokenizer_model: str,
    tokenizer_max_length: int,
    seed: int,
    num_workers: int,
    progress_every: int,
) -> pd.DataFrame:
    ready_rows, pending_rows, stats = prepare_cached_rows(
        session_paths,
        task=task,
        output_root=output_root,
        num_workers=num_workers,
        progress_every=progress_every,
    )
    print(
        "[cache] planned "
        f"ready={len(ready_rows)} pending={len(pending_rows)} "
        f"cache_hits={stats['cache_hits']} clean_hits={stats['clean_hits']} "
        f"duplicates={stats['duplicates']} empty={stats['empty']}"
    )
    processed_rows = run_postprocess_tasks(
        pending_rows,
        tokenizer_model=tokenizer_model,
        tokenizer_max_length=tokenizer_max_length,
        seed=seed,
        num_workers=num_workers,
        progress_every=progress_every,
    )
    rows = ready_rows + processed_rows
    rows.sort(key=lambda row: int(row["order"]))
    materialized_rows = [
        {key: value for key, value in row.items() if key not in {"order", "skip_cleaning"}}
        for row in rows
    ]
    return pd.DataFrame(materialized_rows)


def write_split_manifests(
    frame: pd.DataFrame,
    *,
    task: str,
    output_root: Path,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> None:
    if frame.empty:
        raise ValueError("No usable samples were produced.")

    records = [
        SessionRecord(
            sample_id=str(row.sample_id),
            dataset=str(row.dataset),
            family=str(row.family),
            source_path=str(row.source_path),
            label_name=str(row.label_name),
            label_id=int(row.label_id),
        )
        for row in frame.itertuples()
    ]
    split = stratified_split_records(
        records,
        train_size=train_split,
        val_size=val_split,
        test_size=test_split,
        seed=seed,
    )

    task_root = _task_root(output_root, task)
    task_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(task_root / "manifest.csv", index=False)
    for subset_name, subset_records in split.items():
        subset_ids = {record.sample_id for record in subset_records}
        subset_frame = frame[frame["sample_id"].isin(subset_ids)].copy()
        subset_frame["subset"] = subset_name
        subset_frame.to_csv(task_root / f"{subset_name}.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()
    if not args.source_root.exists():
        print(f"SourceData directory not found at {args.source_root}; nothing to prepare.")
        return

    splitcap_launcher = None if args.splitcap_launcher == "" else args.splitcap_launcher.split()
    if not args.skip_splitcap and splitcap_launcher:
        if which(splitcap_launcher[0]) is None:
            raise RuntimeError(f"SplitCap launcher '{splitcap_launcher[0]}' was not found in PATH.")

    raw_paths = discover_capture_files(args.source_root, args.task, include_paths=args.include_path)
    if not raw_paths:
        print(f"No capture files found for task {args.task} under {args.source_root}")
        return

    session_paths = collect_session_paths(
        raw_paths,
        task=args.task,
        output_root=args.output_root,
        splitcap_exe=args.splitcap_exe,
        splitcap_launcher=splitcap_launcher,
        editcap_path=args.editcap_path,
        skip_splitcap=args.skip_splitcap,
        resume_splitcap=args.resume_splitcap,
    )
    frame = build_cached_manifest(
        session_paths,
        task=args.task,
        output_root=args.output_root,
        tokenizer_model=args.tokenizer_model,
        tokenizer_max_length=args.tokenizer_max_length,
        seed=args.seed,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
    )
    write_split_manifests(
        frame,
        task=args.task,
        output_root=args.output_root,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    print(
        f"Prepared {len(frame)} cached samples for task {args.task} under "
        f"{_task_root(args.output_root, args.task)}"
    )


if __name__ == "__main__":
    main()
