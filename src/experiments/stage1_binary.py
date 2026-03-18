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
STAGE1_ISCX_ALLOWED_CAPTURE_PREFIXES = (
    "vpn_facebook_chat",
    "vpn_ftps",
    "vpn_sftp",
    "vpn_skype_files",
    "vpn_hangouts_audio",
    "vpn_voipbuster",
    "email",
    "hangouts_audio",
    "skype_chat",
    "torrent",
    "voipbuster",
)
STAGE1_MALICIOUS_ALLOWED_FAMILIES = {
    "Artemis",
    "Cobalt",
    "Dridex",
    "Emotet",
    "Hancitor",
    "IcedID",
    "Icedid",
    "Qakbot",
    "Trickbot",
    "TrickBot",
    "Ursnif",
}
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
    frames: List[pd.DataFrame] = []

    for dataset in required_datasets:
        try:
            df = _resolve_dataset_manifest(processed_root=processed_root, dataset=dataset, policy=policy)
        except FileNotFoundError:
            missing.append(dataset)
            continue
        filtered = _filter_stage1_paper_subset(df, dataset=dataset)
        if filtered.empty and not df.empty:
            _log(f"Fallback to unfiltered dataset={dataset}: paper subset matched zero rows")
            filtered = df
        if filtered.empty:
            missing.append(dataset)
            continue
        frames.append(filtered)

    if missing:
        raise FileNotFoundError(f"stage1 missing datasets: {sorted(set(missing))}")
    if not frames:
        raise FileNotFoundError("stage1 manifest empty: no datasets loaded")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    if merged.empty:
        raise ValueError("stage1 manifest empty: required datasets contain no samples")
    merged["label_binary"] = np.where(merged["dataset"] == "ISCX", 0, 1).astype(np.int64)
    merged["label_text"] = np.where(merged["label_binary"] == 0, "normal", "malicious")
    _log(f"Build manifest done: total_rows={len(merged)}")
    return merged


def _filter_stage1_paper_subset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if dataset == "ISCX":
        capture_series = out["capture_id"].astype(str).str.lower() if "capture_id" in out.columns else pd.Series("", index=out.index)
        keep_mask = capture_series.map(_iscx_capture_allowed)
        return out.loc[keep_mask].reset_index(drop=True)
    if dataset in {"MFCP", "MTA"}:
        family_series = out["family"].astype(str) if "family" in out.columns else pd.Series("", index=out.index)
        keep_mask = family_series.isin(STAGE1_MALICIOUS_ALLOWED_FAMILIES)
        return out.loc[keep_mask].reset_index(drop=True)
    return out


def _iscx_capture_allowed(capture_id: str) -> bool:
    stem = Path(str(capture_id)).stem.lower()
    return any(stem.startswith(prefix) for prefix in STAGE1_ISCX_ALLOWED_CAPTURE_PREFIXES)


def _run_stage_report(run_dir: Path, stage: str) -> int:
    if stage in {"warmup", "fusion"}:
        _log(f"Evaluate step start: run_dir={run_dir}, split=test")
        eval_code = evaluate_main(["--run-dir", str(run_dir), "--split", "test"])
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
    return _run_stage_report(run_dir=run_dir, stage=stage)


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
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=args.policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    _log(f"Manifest saved: {output} (rows={len(manifest)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
