from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from tqdm import tqdm

from src.common.structured_logging import format_log_line
from src.data.build_dataset import build_output_paths, ensure_output_dirs, make_manifest_row
from src.data.dataset_inventory import (
    detect_capture_leakage,
    detect_session_leakage,
    scan_source_pcaps,
    split_by_capture,
    split_by_session,
    write_split_artifacts,
)
from src.data.feature_encoder import save_feature_shards
from src.data.pcap_sessionizer import classify_pcap_sessions
from src.data.session_splitcap import cleanup_session_pcaps, split_pcap_to_session_pcaps


SESSION_MANIFEST_COLUMNS = [
    "session_id",
    "dataset",
    "family",
    "capture_id",
    "split",
    "policy",
    "flow_stats",
    "is_tls_ssl",
    "tls_ssl_reason",
]
TLS_MANIFEST_COLUMNS = [
    "session_id",
    "dataset",
    "family",
    "capture_id",
    "split",
    "policy",
    "packet_count",
    "byte_count",
]
DROPPED_MANIFEST_COLUMNS = [
    "session_id",
    "dataset",
    "family",
    "capture_id",
    "split",
    "policy",
    "drop_reason",
]


def preprocess_source(
    source_root: Path | str,
    output_root: Path | str,
    policy: str = "strict",
    filter_mode: str | None = None,
    datasets: Sequence[str] | None = None,
    seed: int = 42,
    cleanup_sessions: bool = True,
    preview_per_family: int = 20,
    show_progress: bool = True,
    resume: bool = False,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    if filter_mode is None:
        filter_mode = policy if policy in {"strict", "relaxed", "session_full"} else "strict"

    log_fn(
        format_log_line(
            level="info",
            module="data",
            event="preprocess_start",
            kv={
                "source_root": str(source_root),
                "policy": policy,
                "datasets": ",".join(datasets) if datasets else "all",
                "resume": resume,
            },
        )
    )

    records = scan_source_pcaps(source_root)
    if datasets:
        keep = {d.strip() for d in datasets if d.strip()}
        records = [r for r in records if r["dataset"] in keep]
    split_rows: List[Dict[str, Any]] = []
    leakage_report: Dict[str, Any] = {}
    split_artifacts: Dict[str, Path] = {}
    split_map = {}
    if policy != "session_full":
        split_rows = split_by_capture(records, seed=seed)
        leakage_report = detect_capture_leakage(split_rows)
        split_artifacts = write_split_artifacts(split_rows, leakage_report, output_root / "manifest")
        split_map = {_split_identity_key(r): r["split"] for r in split_rows}

    by_dataset: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        by_dataset[rec["dataset"]].append(rec)

    total_accepted = 0
    total_dropped = 0
    total_tmp_removed = 0
    session_level_split_rows: List[Dict[str, Any]] = []
    for dataset, dataset_records in sorted(by_dataset.items()):
        paths = build_output_paths(str(output_root), dataset, policy)
        ensure_output_dirs(paths.values())
        run_state = _prepare_dataset_artifacts(paths=paths, resume=resume)

        family_names = sorted({rec["family"] for rec in dataset_records})
        family_to_idx = {name: i for i, name in enumerate(family_names)}

        run_accepted = 0
        run_dropped = 0
        iterator = tqdm(
            enumerate(dataset_records),
            total=len(dataset_records),
            disable=not show_progress,
            desc=f"{dataset} pcap",
            unit="pcap",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.1,
            leave=True,
        )
        for rec_index, rec in iterator:
            capture_stem = _capture_stem(dataset=dataset, family=rec["family"], capture_id=rec["capture_id"])
            done_marker = run_state["checkpoint_dir"] / f"{capture_stem}.done.json"
            session_chunk = run_state["session_chunk_dir"] / f"{capture_stem}.csv"
            tls_chunk = run_state["tls_chunk_dir"] / f"{capture_stem}.csv"
            dropped_chunk = run_state["dropped_chunk_dir"] / f"{capture_stem}.csv"
            rgb_shard = paths["rgb_shard"].parent / f"rgb_shard_{capture_stem}.npz"
            etbert_shard = paths["etbert_shard"].parent / f"etbert_shard_{capture_stem}.npz"

            if resume and done_marker.exists():
                if _capture_outputs_exist(
                    session_chunk=session_chunk,
                    tls_chunk=tls_chunk,
                    dropped_chunk=dropped_chunk,
                    rgb_shard=rgb_shard,
                    etbert_shard=etbert_shard,
                ):
                    continue
                done_marker.unlink(missing_ok=True)

            tmp_capture_dir: Path | None = None
            if policy == "session_full":
                tmp_root = output_root / rec["dataset"] / policy / "tmp_sessions"
                tmp_capture_dir = tmp_root / Path(rec["capture_id"]).stem
                pcap_inputs = split_pcap_to_session_pcaps(rec["pcap_path"], tmp_capture_dir, include_udp=True)
            else:
                pcap_inputs = [Path(rec["pcap_path"])]

            accepted: List[Dict[str, Any]] = []
            dropped: List[Dict[str, Any]] = []
            for input_pcap in pcap_inputs:
                a_rows, d_rows = classify_pcap_sessions(input_pcap, mode=filter_mode)
                accepted.extend(a_rows)
                dropped.extend(d_rows)

            capture_tmp_removed = 0
            if policy == "session_full" and cleanup_sessions:
                capture_tmp_removed = cleanup_session_pcaps(pcap_inputs)
                if tmp_capture_dir is not None:
                    _try_rmdir(tmp_capture_dir)
                    _try_rmdir(tmp_capture_dir.parent)

            split = split_map.get(_split_identity_key(rec), "train")
            if policy == "session_full":
                split = "pending"
            capture_manifest_rows: List[Dict[str, Any]] = []
            capture_tls_rows: List[Dict[str, Any]] = []
            capture_dropped_rows: List[Dict[str, Any]] = []
            capture_samples: List[Dict[str, Any]] = []

            for item in accepted:
                global_session_id = _global_session_id(
                    dataset=rec["dataset"],
                    capture_id=rec["capture_id"],
                    raw_session_id=str(item.get("session_id", "")),
                )
                manifest_row = make_manifest_row(
                    session_id=global_session_id,
                    dataset=rec["dataset"],
                    family=rec["family"],
                    capture_id=rec["capture_id"],
                    split=split,
                    policy=policy,
                    flow_stats={
                        "packet_count": item.get("packet_count", 0),
                        "byte_count": item.get("byte_count", 0),
                    },
                    is_tls_ssl=(
                        bool(item.get("is_tls_ssl", True)) if policy == "session_full" else None
                    ),
                    tls_ssl_reason=(
                        str(item.get("tls_ssl_reason", item.get("tls_reason", "tls")))
                        if policy == "session_full"
                        else None
                    ),
                )
                capture_manifest_rows.append(manifest_row)
                is_tls_flag = bool(item.get("is_tls_ssl", True))
                if is_tls_flag:
                    capture_tls_rows.append(
                        {
                            "session_id": global_session_id,
                            "dataset": rec["dataset"],
                            "family": rec["family"],
                            "capture_id": rec["capture_id"],
                            "split": split,
                            "policy": policy,
                            "packet_count": item.get("packet_count", 0),
                            "byte_count": item.get("byte_count", 0),
                        }
                    )
                capture_samples.append(
                    {
                        **item,
                        "session_id": global_session_id,
                        "dataset": rec["dataset"],
                        "family": rec["family"],
                        "capture_id": rec["capture_id"],
                        "split": split,
                        "policy": policy,
                    }
                )

            for item in dropped:
                global_session_id = _global_session_id(
                    dataset=rec["dataset"],
                    capture_id=rec["capture_id"],
                    raw_session_id=str(item.get("session_id", "")),
                )
                capture_dropped_rows.append(
                    {
                        "session_id": global_session_id,
                        "dataset": rec["dataset"],
                        "family": rec["family"],
                        "capture_id": rec["capture_id"],
                        "split": split,
                        "policy": policy,
                        "drop_reason": item.get("drop_reason", "unknown"),
                    }
                )

            _write_table_csv(capture_manifest_rows, session_chunk, SESSION_MANIFEST_COLUMNS)
            _write_table_csv(capture_tls_rows, tls_chunk, TLS_MANIFEST_COLUMNS)
            _write_table_csv(capture_dropped_rows, dropped_chunk, DROPPED_MANIFEST_COLUMNS)
            save_feature_shards(
                sessions=capture_samples,
                family_to_idx=family_to_idx,
                rgb_path=rgb_shard,
                seq_path=etbert_shard,
                token_max_len=256,
                preview_dir=paths["debug_preview_dir"] if policy == "session_full" else None,
                preview_per_family=preview_per_family,
            )
            _write_done_marker(
                done_marker,
                {
                    "dataset": dataset,
                    "capture_id": rec["capture_id"],
                    "capture_index": rec_index,
                    "accepted_count": len(accepted),
                    "dropped_count": len(dropped),
                    "tmp_removed": capture_tmp_removed,
                    "timestamp": _utc_now_iso(),
                },
            )

            run_accepted += len(accepted)
            run_dropped += len(dropped)
            iterator.set_postfix(
                accepted_tls=run_accepted,
                dropped_non_tls=run_dropped,
                drop_ratio=f"{(run_dropped / max(1, run_accepted + run_dropped)):.2f}",
            )

        session_actual = _consolidate_csv_chunks(
            run_state["session_chunk_dir"],
            paths["manifest"].with_suffix(".csv"),
            SESSION_MANIFEST_COLUMNS,
        )
        tls_actual = _consolidate_csv_chunks(
            run_state["tls_chunk_dir"],
            paths["tls_manifest"].with_suffix(".csv"),
            TLS_MANIFEST_COLUMNS,
        )
        dropped_actual = _consolidate_csv_chunks(
            run_state["dropped_chunk_dir"],
            paths["non_tls_manifest"].with_suffix(".csv"),
            DROPPED_MANIFEST_COLUMNS,
        )
        if policy == "session_full":
            dataset_split_rows = _finalize_session_full_dataset_splits(
                session_csv=session_actual,
                tls_csv=tls_actual,
                dropped_csv=dropped_actual,
                seed=seed,
            )
            session_level_split_rows.extend(dataset_split_rows)

        dataset_accepted, dataset_dropped, dataset_tmp_removed = _sum_done_marker_stats(
            run_state["checkpoint_dir"]
        )
        total_accepted += dataset_accepted
        total_dropped += dataset_dropped
        total_tmp_removed += dataset_tmp_removed

        log_fn(
            format_log_line(
                level="success",
                module="save",
                event="dataset_preprocess_saved",
                kv={
                    "dataset": dataset,
                    "session_manifest": str(session_actual),
                    "tls_manifest": str(tls_actual),
                    "non_tls_manifest": str(dropped_actual),
                    "rgb_shard_dir": str(paths["rgb_shard"].parent),
                    "etbert_shard_dir": str(paths["etbert_shard"].parent),
                },
            )
        )

    if policy == "session_full":
        split_rows = session_level_split_rows
        leakage_report = detect_session_leakage(split_rows)
        split_artifacts = write_split_artifacts(split_rows, leakage_report, output_root / "manifest")

    summary = {
        "total_pcaps": len(records),
        "accepted_sessions": total_accepted,
        "dropped_sessions": total_dropped,
        "split_manifest": str(split_artifacts["split_manifest"]),
        "leakage_report": str(split_artifacts["leakage_report"]),
        "has_leakage": leakage_report["has_leakage"],
        "policy": policy,
        "filter_mode": filter_mode,
        "datasets": ",".join(datasets) if datasets else "all",
        "tmp_session_pcaps_removed": total_tmp_removed,
        "resume": resume,
    }
    log_fn(
        format_log_line(
            level="success",
            module="metric",
            event="preprocess_done",
            kv=summary,
        )
    )
    return summary


def _finalize_session_full_dataset_splits(
    session_csv: Path,
    tls_csv: Path,
    dropped_csv: Path,
    seed: int,
) -> List[Dict[str, Any]]:
    session_df = pd.read_csv(session_csv)
    if session_df.empty:
        _rewrite_split_column(session_df, session_csv, split_map={})
        _rewrite_split_column(pd.read_csv(tls_csv), tls_csv, split_map={})
        _rewrite_split_column(pd.read_csv(dropped_csv), dropped_csv, split_map={})
        return []

    split_input = session_df[
        ["session_id", "dataset", "family", "capture_id"]
    ].to_dict(orient="records")
    split_rows = split_by_session(split_input, seed=seed)
    split_map = {str(row["session_id"]): str(row["split"]) for row in split_rows}

    _rewrite_split_column(session_df, session_csv, split_map=split_map)
    _rewrite_split_column(pd.read_csv(tls_csv), tls_csv, split_map=split_map)
    _rewrite_split_column(pd.read_csv(dropped_csv), dropped_csv, split_map=split_map)
    return split_rows


def _rewrite_split_column(df: pd.DataFrame, target_csv: Path, split_map: Dict[str, str]) -> None:
    if "split" not in df.columns:
        return
    if "session_id" in df.columns and not df.empty:
        df = df.copy()
        df["split"] = [split_map.get(str(sid), "train") for sid in df["session_id"].tolist()]
    df.to_csv(target_csv, index=False)


def _prepare_dataset_artifacts(paths: Dict[str, Path], resume: bool) -> Dict[str, Path]:
    policy_root = paths["manifest"].parent.parent
    manifest_dir = paths["manifest"].parent
    chunk_root = manifest_dir / "chunks"
    session_chunk_dir = chunk_root / "session_manifest"
    tls_chunk_dir = chunk_root / "tls_sessions"
    dropped_chunk_dir = chunk_root / "non_tls_dropped"
    checkpoint_dir = policy_root / "checkpoints" / "preprocess"
    tmp_sessions_dir = policy_root / "tmp_sessions"

    if not resume:
        _remove_tree(chunk_root)
        _remove_tree(checkpoint_dir)
        _remove_tree(tmp_sessions_dir)
        _remove_tree(paths["debug_preview_dir"])
        _unlink_glob(paths["rgb_shard"].parent, "rgb_shard_*.npz")
        _unlink_glob(paths["etbert_shard"].parent, "etbert_shard_*.npz")
        _unlink_file(paths["manifest"])
        _unlink_file(paths["manifest"].with_suffix(".csv"))
        _unlink_file(paths["tls_manifest"])
        _unlink_file(paths["tls_manifest"].with_suffix(".csv"))
        _unlink_file(paths["non_tls_manifest"])
        _unlink_file(paths["non_tls_manifest"].with_suffix(".csv"))

    session_chunk_dir.mkdir(parents=True, exist_ok=True)
    tls_chunk_dir.mkdir(parents=True, exist_ok=True)
    dropped_chunk_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "session_chunk_dir": session_chunk_dir,
        "tls_chunk_dir": tls_chunk_dir,
        "dropped_chunk_dir": dropped_chunk_dir,
        "checkpoint_dir": checkpoint_dir,
    }


def _capture_outputs_exist(
    session_chunk: Path,
    tls_chunk: Path,
    dropped_chunk: Path,
    rgb_shard: Path,
    etbert_shard: Path,
) -> bool:
    return (
        session_chunk.exists()
        and tls_chunk.exists()
        and dropped_chunk.exists()
        and rgb_shard.exists()
        and etbert_shard.exists()
    )


def _split_identity_key(row: Dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset", ""))
    pcap_path = str(row.get("pcap_path", "") or "")
    if pcap_path:
        return dataset, pcap_path
    family = str(row.get("family", ""))
    capture_id = str(row.get("capture_id", ""))
    return dataset, f"{family}/{capture_id}"


def _global_session_id(dataset: str, capture_id: str, raw_session_id: str) -> str:
    if not raw_session_id:
        return ""
    text = f"{dataset}::{capture_id}::{raw_session_id}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def _capture_stem(dataset: str, family: str, capture_id: str) -> str:
    safe_family = _slugify(family)
    safe_capture = _slugify(Path(capture_id).stem)
    stable_key = f"{dataset}::{family}::{capture_id}"
    stable_hash = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:8]
    return f"{safe_family}_{safe_capture}_{stable_hash}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return slug or "unknown"


def _write_table_csv(rows: Sequence[Dict[str, Any]], target_csv: Path, columns: Sequence[str]) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows), columns=list(columns)).to_csv(target_csv, index=False)


def _consolidate_csv_chunks(chunk_dir: Path, target_csv: Path, columns: Sequence[str]) -> Path:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    parts = sorted(chunk_dir.glob("*.csv"))
    if not parts:
        pd.DataFrame(columns=list(columns)).to_csv(target_csv, index=False)
        return target_csv

    wrote_header = False
    with target_csv.open("w", encoding="utf-8", newline="") as out_fp:
        for part in parts:
            with part.open("r", encoding="utf-8", newline="") as in_fp:
                header = in_fp.readline()
                if not header:
                    continue
                if not wrote_header:
                    out_fp.write(header)
                    wrote_header = True
                for line in in_fp:
                    out_fp.write(line)

    if not wrote_header:
        pd.DataFrame(columns=list(columns)).to_csv(target_csv, index=False)
    return target_csv


def _sum_done_marker_stats(checkpoint_dir: Path) -> Tuple[int, int, int]:
    accepted = 0
    dropped = 0
    tmp_removed = 0
    for marker in sorted(checkpoint_dir.glob("*.done.json")):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        accepted += int(data.get("accepted_count", 0))
        dropped += int(data.get("dropped_count", 0))
        tmp_removed += int(data.get("tmp_removed", 0))
    return accepted, dropped, tmp_removed


def _write_done_marker(marker_path: Path, payload: Dict[str, Any]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _unlink_glob(directory: Path, pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        _unlink_file(path)


def _unlink_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except IsADirectoryError:
        return


def _try_rmdir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TLS preprocessing pipeline")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policy", default="strict")
    parser.add_argument("--filter-mode", choices=["strict", "relaxed", "session_full"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cleanup-sessions",
        dest="cleanup_sessions",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--keep-sessions",
        dest="cleanup_sessions",
        action="store_false",
    )
    parser.add_argument("--preview-per-family", type=int, default=20)
    parser.add_argument("--resume", dest="resume", action="store_true", default=False)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    preprocess_source(
        source_root=args.source_root,
        output_root=args.output_root,
        policy=args.policy,
        filter_mode=args.filter_mode,
        datasets=args.datasets,
        seed=args.seed,
        cleanup_sessions=args.cleanup_sessions,
        preview_per_family=args.preview_per_family,
        show_progress=not args.no_progress,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
