from __future__ import annotations

import argparse
import math
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
    print(f"[Stage1Binary][阶段1协议] {message}", flush=True)


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
            _log(f"跳过 alias={alias}: 路径不存在")
            continue
        try:
            df = _load_session_manifest(dataset_dir, policy)
        except FileNotFoundError:
            _log(f"跳过 alias={alias}: policy={policy} 下未找到 manifest")
            continue
        if df.empty:
            _log(f"跳过 alias={alias}: manifest 为空")
            continue
        raw_name = alias
        if "dataset" in df.columns and not df.empty:
            raw_name = str(df["dataset"].iloc[0])
        df = df.copy()
        df["dataset_raw"] = raw_name
        df["dataset"] = dataset
        _log(f"已加载 dataset={dataset}（alias={alias}）, rows={len(df)}")
        return df
    raise FileNotFoundError(f"missing manifest for dataset={dataset} aliases={aliases} policy={policy}")


def build_stage1_manifest(
    processed_root: Path | str,
    policy: str = "session_full",
    required_datasets: Sequence[str] = REQUIRED_STAGE1_DATASETS,
    protocol_mode: str = "paper_balanced",
) -> pd.DataFrame:
    processed_root = Path(processed_root)
    _log(f"开始构建 manifest: processed_root={processed_root}, policy={policy}")
    if protocol_mode not in {"paper_strict", "paper_balanced"}:
        raise ValueError(f"unsupported stage1 protocol_mode: {protocol_mode}")
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
        frames.append(_build_stage1_paper_subset(df, dataset=dataset, protocol_mode=protocol_mode))

    merged = pd.concat(frames, axis=0, ignore_index=True)
    if merged.empty:
        raise ValueError("stage1 manifest empty: required datasets contain no samples")
    merged["label_binary"] = np.where(merged["dataset"] == "ISCX", 0, 1).astype(np.int64)
    merged["label_text"] = np.where(merged["label_binary"] == 0, "normal", "malicious")
    _log(f"manifest 构建完成: total_rows={len(merged)}")
    return merged


def _build_stage1_paper_subset(df: pd.DataFrame, dataset: str, protocol_mode: str) -> pd.DataFrame:
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
            selected = _select_subset_by_mode(
                df.loc[keep_mask].reset_index(drop=True),
                dataset=dataset,
                group_name=str(spec["name"]),
                train_required=int(spec["train"]),
                test_required=int(spec["test"]),
                protocol_mode=protocol_mode,
            )
            if not selected.empty:
                frames.append(selected)
        if not frames:
            raise ValueError(f"stage1 {protocol_mode} subset empty for dataset={dataset}")
        return pd.concat(frames, axis=0, ignore_index=True)

    if dataset == "MTA":
        frames = []
        for spec in PAPER_STAGE1_MTA_SPECS:
            family_norm = _normalize_family(str(spec["family"]))
            keep_mask = df["family"].astype(str).map(_normalize_family) == family_norm
            selected = _select_subset_by_mode(
                df.loc[keep_mask].reset_index(drop=True),
                dataset=dataset,
                group_name=str(spec["family"]),
                train_required=int(spec["train"]),
                test_required=int(spec["test"]),
                protocol_mode=protocol_mode,
            )
            if not selected.empty:
                frames.append(selected)
        if not frames:
            raise ValueError(f"stage1 {protocol_mode} subset empty for dataset={dataset}")
        return pd.concat(frames, axis=0, ignore_index=True)

    if dataset == "MFCP":
        frames = []
        for spec in PAPER_STAGE1_MFCP_SPECS:
            family_norm = _normalize_family(str(spec["family"]))
            keep_mask = df["family"].astype(str).map(_normalize_family) == family_norm
            selected = _select_subset_by_mode(
                df.loc[keep_mask].reset_index(drop=True),
                dataset=dataset,
                group_name=str(spec["family"]),
                train_required=int(spec["train"]),
                test_required=int(spec["test"]),
                protocol_mode=protocol_mode,
            )
            if not selected.empty:
                frames.append(selected)
        if not frames:
            raise ValueError(f"stage1 {protocol_mode} subset empty for dataset={dataset}")
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


def _select_subset_by_mode(
    df: pd.DataFrame,
    dataset: str,
    group_name: str,
    train_required: int,
    test_required: int,
    protocol_mode: str,
) -> pd.DataFrame:
    if protocol_mode == "paper_strict":
        selected = _select_split_quota(
            df=df,
            dataset=dataset,
            group_name=group_name,
            train_required=train_required,
            test_required=test_required,
        )
        _log_subset_summary(
            dataset=dataset,
            group_name=group_name,
            paper_train=train_required,
            paper_test=test_required,
            available_train=train_required,
            available_test=test_required,
            selected_train=train_required,
            selected_test=test_required,
            status="matched",
            protocol_mode=protocol_mode,
        )
        return selected

    train_df = _stable_sort_rows(df.loc[df["split"].astype(str) == "train"].reset_index(drop=True))
    test_df = _stable_sort_rows(df.loc[df["split"].astype(str) == "test"].reset_index(drop=True))
    cap_train = int(math.ceil(train_required * 1.2))
    cap_test = int(math.ceil(test_required * 1.2))
    selected_train_n = min(len(train_df), cap_train)
    selected_test_n = min(len(test_df), cap_test)
    selected = pd.concat(
        [
            train_df.iloc[:selected_train_n].copy(),
            test_df.iloc[:selected_test_n].copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    if selected_train_n == 0 and selected_test_n == 0:
        status = "missing"
    elif selected_train_n < train_required or selected_test_n < test_required:
        status = "undersupplied"
    elif selected_train_n < len(train_df) or selected_test_n < len(test_df):
        status = "capped"
    else:
        status = "matched"
    _log_subset_summary(
        dataset=dataset,
        group_name=group_name,
        paper_train=train_required,
        paper_test=test_required,
        available_train=len(train_df),
        available_test=len(test_df),
        selected_train=selected_train_n,
        selected_test=selected_test_n,
        status=status,
        protocol_mode=protocol_mode,
    )
    return selected.reset_index(drop=True)


def _log_subset_summary(
    dataset: str,
    group_name: str,
    paper_train: int,
    paper_test: int,
    available_train: int,
    available_test: int,
    selected_train: int,
    selected_test: int,
    status: str,
    protocol_mode: str,
) -> None:
    _log(
        "子集统计 "
        f"mode={protocol_mode} dataset={dataset} group={group_name} "
        f"paper_train={paper_train} paper_test={paper_test} "
        f"available_train={available_train} available_test={available_test} "
        f"selected_train={selected_train} selected_test={selected_test} "
        f"status={status}"
    )


def _run_stage_report(run_dir: Path, stage: str, device: str) -> int:
    if stage in {"warmup", "fusion"}:
        _log(f"评估步骤开始: run_dir={run_dir}, split=test")
        eval_code = evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--device", device])
        if eval_code != 0:
            _log(f"评估步骤失败: exit_code={eval_code}")
            return eval_code
        _log("评估步骤完成")
    else:
        _log(f"跳过评估：stage={stage}；report 将直接使用阶段产物")

    _log(f"报告步骤开始: run_dir={run_dir}")
    report_code = report_main(["--run-dir", str(run_dir)])
    if report_code != 0:
        _log(f"报告步骤失败: exit_code={report_code}")
        return report_code
    _log("报告步骤完成")
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
    best_metric: str,
    protocol_mode: str,
    hidden_dim: int,
    fusion_layers: int,
    fusion_heads: int,
    fusion_dropout: float,
    alpha: float,
    beta: float,
    val_fraction: float,
    train_max_samples: int | None,
) -> int:
    _log("协议执行模式已启用")
    manifest = build_stage1_manifest(processed_root=processed_root, policy=policy, protocol_mode=protocol_mode)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    _log(f"Manifest 已保存: {output_manifest} (rows={len(manifest)})")

    _log("训练步骤开始")
    train_args = [
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
        "--best-metric",
        str(best_metric),
        "--hidden-dim",
        str(hidden_dim),
        "--fusion-layers",
        str(fusion_layers),
        "--fusion-heads",
        str(fusion_heads),
        "--fusion-dropout",
        str(fusion_dropout),
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--val-fraction",
        str(val_fraction),
        "--datasets",
        *list(REQUIRED_STAGE1_DATASETS),
        "--session-filter-manifest",
        str(output_manifest),
        "--label-mode",
        "binary",
        "--num-classes",
        "2",
    ]
    if train_max_samples is not None:
        train_args.extend(["--train-max-samples", str(train_max_samples)])
    train_code = train_main(train_args)
    if train_code != 0:
        _log(f"训练步骤失败: exit_code={train_code}")
        return train_code
    _log("训练步骤完成")

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
    parser.add_argument("--best-metric", default="val_macro_f1", choices=["val_macro_f1", "val_acc"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", "--num-heads", dest="fusion_heads", type=int, default=4)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--protocol-mode", default="paper_balanced", choices=["paper_strict", "paper_balanced"])
    args = parser.parse_args(list(argv) if argv is not None else None)

    processed_root = Path(args.processed_root)
    output = Path(args.output)

    _log(
        f"启动: processed_root={processed_root}, policy={args.policy}, output={output}, execute={args.execute}"
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
            best_metric=args.best_metric,
            protocol_mode=args.protocol_mode,
            hidden_dim=args.hidden_dim,
            fusion_layers=args.fusion_layers,
            fusion_heads=args.fusion_heads,
            fusion_dropout=args.fusion_dropout,
            alpha=args.alpha,
            beta=args.beta,
            val_fraction=args.val_fraction,
            train_max_samples=args.train_max_samples,
        )

    manifest = build_stage1_manifest(processed_root=processed_root, policy=args.policy, protocol_mode=args.protocol_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    _log(f"Manifest 已保存: {output} (rows={len(manifest)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
