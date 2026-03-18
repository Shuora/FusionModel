from __future__ import annotations

import json
from itertools import chain
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


SUPPORTED_PCAP_SUFFIXES = (".pcap", ".pcapng")
DATASET_FAMILY_ALIASES: Dict[str, Dict[str, str]] = {
    "MFCP": {
        "PUA": "PUA",
        "PUA-1": "PUA",
        "PUA-2": "PUA",
    },
    "MTA": {
        "IcedID": "IcedID",
        "IcedID_1": "IcedID",
        "IcedID_2": "IcedID",
    },
}


def normalize_family_name(dataset: str, family: str) -> str:
    alias_map = DATASET_FAMILY_ALIASES.get(str(dataset), {})
    return alias_map.get(str(family), str(family))


def scan_source_pcaps(source_root: Path | str) -> List[Dict[str, str]]:
    root = Path(source_root)
    records: List[Dict[str, str]] = []

    for pcap_path in chain.from_iterable(root.rglob(f"*{suffix}") for suffix in SUPPORTED_PCAP_SUFFIXES):
        if ":Zone.Identifier" in pcap_path.name:
            continue
        rel = pcap_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            continue

        dataset = parts[0]
        family = parts[1] if len(parts) >= 3 else pcap_path.stem
        family = normalize_family_name(dataset=dataset, family=family)
        records.append(
            {
                "dataset": dataset,
                "family": family,
                "capture_id": pcap_path.name,
                "pcap_path": str(pcap_path),
            }
        )

    return sorted(records, key=lambda x: (x["dataset"], x["family"], x["capture_id"]))


def split_by_capture(
    records: Sequence[Dict[str, str]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = {}
    for item in records:
        key = (item["dataset"], item["family"])
        grouped.setdefault(key, []).append(item)

    rng = random.Random(seed)
    out: List[Dict[str, str]] = []

    for _, items in grouped.items():
        captures = sorted(items, key=lambda x: x["capture_id"])
        rng.shuffle(captures)
        n = len(captures)
        if n == 1:
            split_plan = ["train"]
        else:
            n_train = max(1, int(round(n * train_ratio)))
            if n_train >= n:
                n_train = n - 1
            n_test = n - n_train
            split_plan = ["train"] * n_train + ["test"] * n_test

        for item, split in zip(captures, split_plan):
            out.append({**item, "split": split})

    return sorted(out, key=lambda x: (x["dataset"], x["family"], x["capture_id"]))


def split_by_session(
    records: Sequence[Dict[str, str]],
    seed: int = 42,
    train_ratio: float = 0.8,
) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = {}
    for item in records:
        key = (item["dataset"], item["family"])
        grouped.setdefault(key, []).append(item)

    rng = random.Random(seed)
    out: List[Dict[str, str]] = []

    for _, items in grouped.items():
        sessions = sorted(items, key=lambda x: str(x.get("session_id", "")))
        rng.shuffle(sessions)
        n = len(sessions)
        if n == 1:
            split_plan = ["train"]
        else:
            n_train = max(1, int(round(n * train_ratio)))
            if n_train >= n:
                n_train = n - 1
            n_test = n - n_train
            split_plan = ["train"] * n_train + ["test"] * n_test

        for item, split in zip(sessions, split_plan):
            out.append({**item, "split": split})

    return sorted(
        out,
        key=lambda x: (x["dataset"], x["family"], str(x.get("session_id", "")), x.get("capture_id", "")),
    )


def detect_capture_leakage(split_rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    capture_to_splits: Dict[str, set] = {}
    for row in split_rows:
        identity = str(row.get("pcap_path") or f"{row.get('family', '')}::{row.get('capture_id', '')}")
        capture_key = f"{row['dataset']}::{identity}"
        capture_to_splits.setdefault(capture_key, set()).add(row["split"])

    leaked = sorted([k for k, splits in capture_to_splits.items() if len(splits) > 1])
    split_counts: Dict[str, int] = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    return {
        "has_leakage": len(leaked) > 0,
        "leaked_capture_count": len(leaked),
        "leaked_captures": leaked,
        "split_counts": split_counts,
        "total_rows": len(split_rows),
    }


def detect_session_leakage(split_rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    session_to_splits: Dict[str, set] = {}
    for row in split_rows:
        session_id = str(row.get("session_id", ""))
        dataset = str(row.get("dataset", ""))
        key = f"{dataset}::{session_id}"
        session_to_splits.setdefault(key, set()).add(row["split"])

    leaked = sorted([k for k, splits in session_to_splits.items() if len(splits) > 1])
    split_counts: Dict[str, int] = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    return {
        "has_leakage": len(leaked) > 0,
        "leaked_session_count": len(leaked),
        "leaked_sessions": leaked,
        "split_counts": split_counts,
        "total_rows": len(split_rows),
        "split_granularity": "session",
    }


def write_split_artifacts(
    split_rows: Sequence[Dict[str, str]],
    leakage_report: Dict[str, object],
    output_dir: Path | str,
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "split_manifest.parquet"
    manifest_actual = _write_table_with_fallback(split_rows, manifest_path)

    leakage_path = out_dir / "leakage_report.json"
    leakage_path.write_text(json.dumps(leakage_report, ensure_ascii=False, indent=2))

    return {
        "split_manifest": manifest_actual,
        "leakage_report": leakage_path,
    }


def _write_table_with_fallback(rows: Sequence[Dict[str, str]], target_path: Path) -> Path:
    df = pd.DataFrame(list(rows))
    try:
        df.to_parquet(target_path, index=False)
        return target_path
    except Exception:
        csv_path = target_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
