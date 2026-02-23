from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


def scan_source_pcaps(source_root: Path | str) -> List[Dict[str, str]]:
    root = Path(source_root)
    records: List[Dict[str, str]] = []

    for pcap_path in root.rglob("*.pcap"):
        if ":Zone.Identifier" in pcap_path.name:
            continue
        rel = pcap_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            continue

        dataset = parts[0]
        family = parts[1] if len(parts) >= 3 else pcap_path.stem
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
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
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
        elif n == 2:
            split_plan = ["train", "test"]
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = int(round(n * val_ratio))
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_test = 1
                if n_val > 0:
                    n_val -= 1
                else:
                    n_train -= 1
            split_plan = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

        for item, split in zip(captures, split_plan):
            out.append({**item, "split": split})

    return sorted(out, key=lambda x: (x["dataset"], x["family"], x["capture_id"]))


def detect_capture_leakage(split_rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    capture_to_splits: Dict[str, set] = {}
    for row in split_rows:
        capture_key = f"{row['dataset']}::{row['capture_id']}"
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
