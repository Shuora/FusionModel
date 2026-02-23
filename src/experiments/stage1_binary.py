from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


REQUIRED_STAGE1_DATASETS = ("ISCX", "MFCP", "MTA", "USTC-TFC2016")


def _load_session_manifest(dataset_dir: Path, policy: str) -> pd.DataFrame:
    manifest_dir = dataset_dir / policy / "manifest"
    csv_path = manifest_dir / "session_manifest.csv"
    parquet_path = manifest_dir / "session_manifest.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"missing manifest for dataset={dataset_dir.name} policy={policy}")


def build_stage1_manifest(
    processed_root: Path | str,
    policy: str = "session_full",
    required_datasets: Sequence[str] = REQUIRED_STAGE1_DATASETS,
) -> pd.DataFrame:
    processed_root = Path(processed_root)
    missing: List[str] = []
    frames: List[pd.DataFrame] = []

    for dataset in required_datasets:
        dataset_dir = processed_root / dataset
        if not dataset_dir.exists():
            missing.append(dataset)
            continue
        try:
            df = _load_session_manifest(dataset_dir, policy)
        except FileNotFoundError:
            missing.append(dataset)
            continue
        if "dataset" not in df.columns:
            df["dataset"] = dataset
        frames.append(df)

    if missing:
        raise FileNotFoundError(f"stage1 missing datasets: {sorted(set(missing))}")
    if not frames:
        raise FileNotFoundError("stage1 manifest empty: no datasets loaded")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged["label_binary"] = np.where(merged["dataset"] == "ISCX", 0, 1).astype(np.int64)
    merged["label_text"] = np.where(merged["label_binary"] == 0, "normal", "malicious")
    return merged


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build stage1 mixed binary manifest")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--policy", default="session_full")
    parser.add_argument("--output", default="outputs/stage1_binary_manifest.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = build_stage1_manifest(processed_root=args.processed_root, policy=args.policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

