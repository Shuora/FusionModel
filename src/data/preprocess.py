from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from tqdm import tqdm

from src.common.structured_logging import format_log_line
from src.data.build_dataset import build_output_paths, ensure_output_dirs, make_manifest_row
from src.data.dataset_inventory import (
    detect_capture_leakage,
    scan_source_pcaps,
    split_by_capture,
    write_split_artifacts,
)
from src.data.feature_encoder import save_feature_shards
from src.data.pcap_sessionizer import classify_pcap_sessions
from src.data.session_splitcap import cleanup_session_pcaps, split_pcap_to_session_pcaps


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
            },
        )
    )

    records = scan_source_pcaps(source_root)
    if datasets:
        keep = {d.strip() for d in datasets if d.strip()}
        records = [r for r in records if r["dataset"] in keep]
    split_rows = split_by_capture(records, seed=seed)
    leakage_report = detect_capture_leakage(split_rows)
    split_artifacts = write_split_artifacts(split_rows, leakage_report, output_root / "manifest")

    split_map = {(r["dataset"], r["capture_id"]): r["split"] for r in split_rows}
    by_dataset: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        by_dataset[rec["dataset"]].append(rec)

    total_accepted = 0
    total_dropped = 0
    total_tmp_removed = 0
    for dataset, dataset_records in sorted(by_dataset.items()):
        paths = build_output_paths(str(output_root), dataset, policy)
        ensure_output_dirs(paths.values())

        session_manifest_rows: List[Dict[str, Any]] = []
        tls_rows: List[Dict[str, Any]] = []
        dropped_rows: List[Dict[str, Any]] = []
        accepted_samples: List[Dict[str, Any]] = []

        iterator = tqdm(
            dataset_records,
            disable=not show_progress,
            desc=f"{dataset} pcap",
            unit="pcap",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.1,
            leave=True,
        )
        for rec in iterator:
            tmp_capture_dir: Path | None = None
            if policy == "session_full":
                tmp_root = output_root / rec["dataset"] / policy / "tmp_sessions"
                tmp_capture_dir = tmp_root / Path(rec["capture_id"]).stem
                pcap_inputs = split_pcap_to_session_pcaps(rec["pcap_path"], tmp_capture_dir)
            else:
                pcap_inputs = [Path(rec["pcap_path"])]

            accepted: List[Dict[str, Any]] = []
            dropped: List[Dict[str, Any]] = []
            for input_pcap in pcap_inputs:
                a_rows, d_rows = classify_pcap_sessions(input_pcap, mode=filter_mode)
                accepted.extend(a_rows)
                dropped.extend(d_rows)

            if policy == "session_full" and cleanup_sessions:
                removed = cleanup_session_pcaps(pcap_inputs)
                total_tmp_removed += removed
                if tmp_capture_dir is not None:
                    _try_rmdir(tmp_capture_dir)
                    _try_rmdir(tmp_capture_dir.parent)

            split = split_map.get((rec["dataset"], rec["capture_id"]), "train")

            for item in accepted:
                manifest_row = make_manifest_row(
                    session_id=item["session_id"],
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
                session_manifest_rows.append(manifest_row)
                is_tls_flag = bool(item.get("is_tls_ssl", True))
                if is_tls_flag:
                    tls_rows.append(
                        {
                            "session_id": item["session_id"],
                            "dataset": rec["dataset"],
                            "family": rec["family"],
                            "capture_id": rec["capture_id"],
                            "split": split,
                            "policy": policy,
                            "packet_count": item.get("packet_count", 0),
                            "byte_count": item.get("byte_count", 0),
                        }
                    )
                accepted_samples.append(
                    {
                        **item,
                        "dataset": rec["dataset"],
                        "family": rec["family"],
                        "capture_id": rec["capture_id"],
                        "split": split,
                        "policy": policy,
                    }
                )

            for item in dropped:
                dropped_rows.append(
                    {
                        "session_id": item.get("session_id", ""),
                        "dataset": rec["dataset"],
                        "family": rec["family"],
                        "capture_id": rec["capture_id"],
                        "split": split,
                        "policy": policy,
                        "drop_reason": item.get("drop_reason", "unknown"),
                    }
                )

            total_accepted += len(accepted)
            total_dropped += len(dropped)
            iterator.set_postfix(
                accepted_tls=total_accepted,
                dropped_non_tls=total_dropped,
                drop_ratio=f"{(total_dropped / max(1, total_accepted + total_dropped)):.2f}",
            )

        session_actual = _write_table_with_fallback(session_manifest_rows, paths["manifest"])
        tls_actual = _write_table_with_fallback(tls_rows, paths["tls_manifest"])
        dropped_actual = _write_table_with_fallback(dropped_rows, paths["non_tls_manifest"])
        family_names = sorted({s["family"] for s in accepted_samples})
        family_to_idx = {name: i for i, name in enumerate(family_names)}
        save_feature_shards(
            sessions=accepted_samples,
            family_to_idx=family_to_idx,
            rgb_path=paths["rgb_shard"],
            seq_path=paths["seq_shard"],
            token_max_len=256,
            preview_dir=paths["debug_preview_dir"] if policy == "session_full" else None,
            preview_per_family=preview_per_family,
        )

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
                    "rgb_shard": str(paths["rgb_shard"]),
                    "seq_shard": str(paths["seq_shard"]),
                },
            )
        )

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


def _write_table_with_fallback(rows: Sequence[Dict[str, Any]], target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    try:
        df.to_parquet(target_path, index=False)
        return target_path
    except Exception:
        csv_path = target_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
