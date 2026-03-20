from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluate import main as evaluate_main
from src.report import main as report_main
from src.train import main as train_main


REQUIRED_STAGE1_DATASETS = ("ISCX", "MFCP", "MTA")
PAPER_STAGE1_ISCX_SPECS = [
    {"name": "vpn_facebook_chat", "capture_prefixes": ("vpn_facebook_chat",), "train": 927, "test": 232},
    {"name": "vpn_file_transfer", "capture_prefixes": ("vpn_ftps", "vpn_sftp", "vpn_skype_files"), "train": 805, "test": 201},
    {"name": "vpn_hangouts_audio", "capture_prefixes": ("vpn_hangouts_audio",), "train": 2538, "test": 634},
    {"name": "vpn_voipbuster", "capture_prefixes": ("vpn_voipbuster",), "train": 1294, "test": 324},
    {"name": "email_nonvpn", "capture_prefixes": ("email",), "train": 2798, "test": 699},
    {"name": "hangouts_audio_nonvpn", "capture_prefixes": ("hangouts_audio",), "train": 1384, "test": 346},
    {"name": "skype_chat_nonvpn", "capture_prefixes": ("skype_chat",), "train": 3542, "test": 886},
    {"name": "torrent_nonvpn", "capture_prefixes": ("torrent",), "train": 836, "test": 209},
    {"name": "voipbuster_nonvpn", "capture_prefixes": ("voipbuster",), "train": 1420, "test": 355},
]
PAPER_STAGE1_MTA_SPECS = [
    {"family": "Dridex", "train": 492, "test": 123},
    {"family": "Emotet", "train": 3368, "test": 842},
    {"family": "Hancitor", "train": 13452, "test": 3363},
    {"family": "IcedID", "train": 1454, "test": 364},
    {"family": "Qakbot", "train": 3350, "test": 838},
    {"family": "Trickbot", "train": 1794, "test": 448},
    {"family": "Ursnif", "train": 506, "test": 127},
]
PAPER_STAGE1_MFCP_SPECS = [
    {"family": "Artemis", "train": 6000, "test": 1500},
    {"family": "Cobalt", "train": 1501, "test": 375},
    {"family": "Dridex", "train": 6000, "test": 1500},
    {"family": "PUA", "train": 5614, "test": 1403},
    {"family": "TrickBot", "train": 6000, "test": 1500},
    {"family": "Ursnif", "train": 6000, "test": 1500},
]
DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ISCX": ("ISCX", "ISCX-VPN-NonVPN-2016"),
    "MFCP": ("MFCP",),
    "MTA": ("MTA",),
    "USTC-TFC2016": ("USTC-TFC2016",),
}


def _log(message: str) -> None:
    print(f"[Stage1Binary] {message}", flush=True)


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
            _log(f"Skip alias={alias}: path not found")
            continue
        try:
            df = _load_session_manifest(dataset_dir, policy)
        except FileNotFoundError:
            _log(f"Skip alias={alias}: manifest missing under policy={policy}")
            continue
        if df.empty:
            _log(f"Skip alias={alias}: manifest is empty")
            continue
        raw_name = alias
        if "dataset" in df.columns and not df.empty:
            raw_name = str(df["dataset"].iloc[0])
        df = df.copy()
        df["dataset_raw"] = raw_name
        df["dataset"] = dataset
        _log(f"Loaded dataset={dataset} via alias={alias}, rows={len(df)}")
        return df
    raise FileNotFoundError(f"missing manifest for dataset={dataset} aliases={aliases} policy={policy}")


def build_stage1_manifest(
    processed_root: Path | str,
    policy: str = "session_full",
    required_datasets: Sequence[str] = REQUIRED_STAGE1_DATASETS,
) -> pd.DataFrame:
    processed_root = Path(processed_root)
    _log(f"Build manifest start: processed_root={processed_root}, policy={policy}")
    missing: List[str] = []
    loaded_frames: List[Tuple[str, pd.DataFrame]] = []
    frames: List[pd.DataFrame] = []

    for dataset in required_datasets:
        try:
            df = _resolve_dataset_manifest(processed_root=processed_root, dataset=dataset, policy=policy)
        except FileNotFoundError:
            missing.append(dataset)
            continue
        loaded_frames.append((dataset, df))

    if missing:
        raise FileNotFoundError(f"stage1 missing datasets: {sorted(set(missing))}")
    if not frames:
        if not loaded_frames:
            raise FileNotFoundError("stage1 manifest empty: no datasets loaded")

    for dataset, df in loaded_frames:
        frames.append(_build_stage1_paper_subset(df, dataset=dataset))

    merged = pd.concat(frames, axis=0, ignore_index=True)
    if merged.empty:
        raise ValueError("stage1 manifest empty: required datasets contain no samples")
    merged["label_binary"] = np.where(merged["dataset"] == "ISCX", 0, 1).astype(np.int64)
    merged["label_text"] = np.where(merged["label_binary"] == 0, "normal", "malicious")
    _log(f"Build manifest done: total_rows={len(merged)}")
    return merged


def _build_stage1_paper_subset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"stage1 paper subset empty for dataset={dataset}")

    if dataset == "ISCX":
        frames = []
        capture_series = (
            df["capture_id"].astype(str).map(lambda value: Path(value).stem.lower())
            if "capture_id" in df.columns
            else pd.Series("", index=df.index)
        )
        for spec in PAPER_STAGE1_ISCX_SPECS:
            keep_mask = capture_series.map(lambda stem: any(stem.startswith(prefix) for prefix in spec["capture_prefixes"]))
            frames.append(
                _select_split_quota(
                    df.loc[keep_mask].reset_index(drop=True),
                    dataset=dataset,
                    group_name=str(spec["name"]),
                    train_required=int(spec["train"]),
                    test_required=int(spec["test"]),
                )
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    if dataset == "MTA":
        frames = []
        for spec in PAPER_STAGE1_MTA_SPECS:
            family_norm = _normalize_family(str(spec["family"]))
            keep_mask = df["family"].astype(str).map(_normalize_family) == family_norm
            frames.append(
                _select_split_quota(
                    df.loc[keep_mask].reset_index(drop=True),
                    dataset=dataset,
                    group_name=str(spec["family"]),
                    train_required=int(spec["train"]),
                    test_required=int(spec["test"]),
                )
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    if dataset == "MFCP":
        frames = []
        for spec in PAPER_STAGE1_MFCP_SPECS:
            family_norm = _normalize_family(str(spec["family"]))
            keep_mask = df["family"].astype(str).map(_normalize_family) == family_norm
            frames.append(
                _select_split_quota(
                    df.loc[keep_mask].reset_index(drop=True),
                    dataset=dataset,
                    group_name=str(spec["family"]),
                    train_required=int(spec["train"]),
                    test_required=int(spec["test"]),
                )
            )
        return pd.concat(frames, axis=0, ignore_index=True)

    return df.reset_index(drop=True)


def _normalize_family(value: str) -> str:
    return str(value).strip().lower()


def _stable_sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_capture_sort"] = out.get("capture_id", pd.Series("", index=out.index)).astype(str)
    out["_session_sort"] = out.get("session_id", pd.Series("", index=out.index)).astype(str)
    out = out.sort_values(by=["_capture_sort", "_session_sort"], kind="stable").reset_index(drop=True)
    return out.drop(columns=["_capture_sort", "_session_sort"])


def _select_split_quota(
    df: pd.DataFrame,
    dataset: str,
    group_name: str,
    train_required: int,
    test_required: int,
) -> pd.DataFrame:
    train_df = _stable_sort_rows(df.loc[df["split"].astype(str) == "train"].reset_index(drop=True))
    test_df = _stable_sort_rows(df.loc[df["split"].astype(str) == "test"].reset_index(drop=True))

    if len(train_df) < train_required or len(test_df) < test_required:
        raise ValueError(
            "stage1 paper quota unavailable: "
            f"dataset={dataset} group={group_name} "
            f"required_train={train_required} available_train={len(train_df)} "
            f"required_test={test_required} available_test={len(test_df)}"
        )

    selected = pd.concat(
        [
            train_df.iloc[:train_required].copy(),
            test_df.iloc[:test_required].copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    return selected.reset_index(drop=True)


def _run_stage_report(run_dir: Path, stage: str, device: str) -> int:
    if stage in {"warmup", "fusion"}:
        _log(f"Evaluate step start: run_dir={run_dir}, split=test")
        eval_code = evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--device", device])
        if eval_code != 0:
            _log(f"Evaluate step failed: exit_code={eval_code}")
            return eval_code
        _log("Evaluate step done")
    else:
        _log(f"Skip evaluate for stage={stage}; report will use stage artifacts directly")

    _log(f"Report step start: run_dir={run_dir}")
    report_code = report_main(["--run-dir", str(run_dir)])
    if report_code != 0:
        _log(f"Report step failed: exit_code={report_code}")
        return report_code
    _log("Report step done")
    return 0


def run_stage1_protocol(
    processed_root: Path,
    policy: str,
    output_manifest: Path,
    run_root: Path,
    run_id: str,
    stage: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    num_workers: int,
) -> int:
    _log("Protocol execute mode enabled")
    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    _log(f"Manifest saved: {output_manifest} (rows={len(manifest)})")

    _log("Train step start")
    train_code = train_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            policy,
            "--stage",
            stage,
            "--run-root",
            str(run_root),
            "--run-id",
            run_id,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--seed",
            str(seed),
            "--device",
            device,
            "--num-workers",
            str(num_workers),
            "--datasets",
            *list(REQUIRED_STAGE1_DATASETS),
            "--session-filter-manifest",
            str(output_manifest),
            "--label-mode",
            "binary",
            "--num-classes",
            "2",
        ]
    )
    if train_code != 0:
        _log(f"Train step failed: exit_code={train_code}")
        return train_code
    _log("Train step done")

    run_dir = run_root / run_id
    return _run_stage_report(run_dir=run_dir, stage=stage, device=device)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build stage1 mixed binary manifest")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--policy", default="session_full")
    parser.add_argument("--output", default="outputs/protocol/stage1_binary_manifest.csv")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-id", default="stage1-binary")
    parser.add_argument("--stage", default="fusion", choices=["warmup", "fusion", "stacking", "moe"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(list(argv) if argv is not None else None)

    processed_root = Path(args.processed_root)
    output = Path(args.output)

    _log(
        f"Start: processed_root={processed_root}, policy={args.policy}, output={output}, execute={args.execute}"
    )
    if args.execute:
        return run_stage1_protocol(
            processed_root=processed_root,
            policy=args.policy,
            output_manifest=output,
            run_root=Path(args.run_root),
            run_id=args.run_id,
            stage=args.stage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=args.policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    _log(f"Manifest saved: {output} (rows={len(manifest)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
