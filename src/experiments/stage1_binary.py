from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_STAGE1_DATASETS = ("ISCX", "MFCP", "MTA", "USTC-TFC2016")
DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ISCX": ("ISCX", "ISCX-VPN-NonVPN-2016"),
    "MFCP": ("MFCP",),
    "MTA": ("MTA",),
    "USTC-TFC2016": ("USTC-TFC2016",),
}


def _load_session_manifest(dataset_dir: Path, policy: str) -> pd.DataFrame:
    manifest_dir = dataset_dir / policy / "manifest"
    csv_path = manifest_dir / "session_manifest.csv"
    parquet_path = manifest_dir / "session_manifest.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"missing manifest for dataset={dataset_dir.name} policy={policy}")


def _resolve_dataset_manifest(processed_root: Path, dataset: str, policy: str) -> pd.DataFrame:
    aliases = DATASET_ALIASES.get(dataset, (dataset,))
    for alias in aliases:
        dataset_dir = processed_root / alias
        if not dataset_dir.exists():
            continue
        try:
            df = _load_session_manifest(dataset_dir, policy)
        except FileNotFoundError:
            continue
        raw_name = alias
        if "dataset" in df.columns and not df.empty:
            raw_name = str(df["dataset"].iloc[0])
        df["dataset_raw"] = raw_name
        df["dataset"] = dataset
        return df
    raise FileNotFoundError(f"missing manifest for dataset={dataset} aliases={aliases} policy={policy}")


def build_stage1_manifest(
    processed_root: Path | str,
    policy: str = "session_full",
    required_datasets: Sequence[str] = REQUIRED_STAGE1_DATASETS,
) -> pd.DataFrame:
    processed_root = Path(processed_root)
    missing: List[str] = []
    frames: List[pd.DataFrame] = []

    for dataset in required_datasets:
        try:
            df = _resolve_dataset_manifest(processed_root=processed_root, dataset=dataset, policy=policy)
        except FileNotFoundError:
            missing.append(dataset)
            continue
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
