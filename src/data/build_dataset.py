from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.data.tls_filter import classify_session_as_tls


def build_output_paths(base_output_dir: str, dataset: str, policy: str) -> Dict[str, Path]:
    root = Path(base_output_dir) / dataset / policy
    return {
        "rgb_shard": root / "rgb" / "rgb_shard_00000.npz",
        "seq_shard": root / "seq" / "seq_shard_00000.npz",
        "manifest": root / "manifest" / "session_manifest.parquet",
        "tls_manifest": root / "manifest" / "tls_sessions.parquet",
        "non_tls_manifest": root / "manifest" / "non_tls_dropped.parquet",
        "debug_preview_dir": root / "debug" / "preview_png",
    }


def make_manifest_row(
    session_id: str,
    dataset: str,
    family: str,
    capture_id: str,
    split: str,
    policy: str,
    flow_stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "dataset": dataset,
        "family": family,
        "capture_id": capture_id,
        "split": split,
        "policy": policy,
        "flow_stats": flow_stats or {},
    }


def split_tls_and_non_tls(
    sessions: Sequence[Dict[str, Any]], mode: str = "strict"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for session in sessions:
        ok, reason = classify_session_as_tls(
            payload_chunks=session.get("payload_chunks", []),
            protocol=session.get("protocol", "TCP"),
            mode=mode,
        )
        if ok:
            accepted.append({**session, "tls_reason": reason})
        else:
            dropped.append(
                {
                    "session_id": session.get("session_id", ""),
                    "drop_reason": reason,
                }
            )
    return accepted, dropped


def ensure_output_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
