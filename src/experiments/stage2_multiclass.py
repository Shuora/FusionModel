from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold

from src.evaluate import main as evaluate_main
from src.meta_features import STAGE2_META_SCHEMA_VERSION, flatten_meta_feature_blocks
from src.models.fusion_model import MobileViTETBertFusionClassifier
from src.pipeline_data import load_policy_multimodal_data
from src.report import main as report_main
from src.report import resolve_canonical_final_metric_source_and_path
from src.run_dir import current_run_date_partition
from src.runtime_device import resolve_runtime_device
from src.stage2_registry import STAGE2_DATASET_ORDER, build_stage2_run_layout
from src.stage2_trainer import run_stage_b_dataset_finetune
from src.stacking import main as stacking_main
from src.train import main as train_main
from src.train import run_stage2_shared_stage_a as train_run_stage2_shared_stage_a


STAGE2_TASKS = [
    {"dataset": "MTA", "num_classes": 7},
    {"dataset": "MFCP", "num_classes": 6},
    {"dataset": "USTC-TFC2016", "num_classes": 10},
]


def build_stage2_tasks() -> List[dict]:
    return [dict(item) for item in STAGE2_TASKS]


def _run_stage_report(run_dir: Path, stage: str, device: str) -> int:
    if stage in {"warmup", "fusion"}:
        eval_code = evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--device", device])
        if eval_code != 0:
            return eval_code
    report_code = report_main(["--run-dir", str(run_dir)])
    return report_code


def _build_train_args(
    *,
    processed_root: Path,
    policy: str,
    run_root: Path,
    run_id: str,
    dataset: str,
    num_classes: int,
    stage: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    num_workers: int,
    hidden_dim: int,
    fusion_layers: int,
    fusion_heads: int,
    fusion_dropout: float,
    alpha: float,
    beta: float,
    val_fraction: float,
    best_metric: str,
    checkpoint_selection: str,
    early_stopping_patience: int,
    class_weight_mode: str,
    scheduler: str,
    freeze_image_backbone_epochs: int,
    fusion_mode: str,
    text_shortcut_scale: float,
    max_grad_norm: float,
    grad_explode_threshold: float,
    warmup_checkpoint: str | None = None,
    train_max_samples: int | None = None,
    session_filter_manifest: Path | None = None,
) -> list[str]:
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
        "--best-metric",
        str(best_metric),
        "--checkpoint-selection",
        str(checkpoint_selection),
        "--early-stopping-patience",
        str(early_stopping_patience),
        "--class-weight-mode",
        str(class_weight_mode),
        "--scheduler",
        str(scheduler),
        "--freeze-image-backbone-epochs",
        str(freeze_image_backbone_epochs),
        "--fusion-mode",
        str(fusion_mode),
        "--text-shortcut-scale",
        str(text_shortcut_scale),
        "--max-grad-norm",
        str(max_grad_norm),
        "--grad-explode-threshold",
        str(grad_explode_threshold),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
        "--datasets",
        dataset,
        "--label-mode",
        "multiclass",
        "--num-classes",
        str(num_classes),
    ]
    if warmup_checkpoint:
        train_args.extend(["--warmup-checkpoint", str(warmup_checkpoint)])
    if train_max_samples is not None:
        train_args.extend(["--train-max-samples", str(train_max_samples)])
    if session_filter_manifest is not None:
        train_args.extend(["--session-filter-manifest", str(session_filter_manifest)])
    return train_args


def _run_stage2_task(
    processed_root: Path,
    policy: str,
    dated_run_root: Path,
    dataset: str,
    num_classes: int,
    stage: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    num_workers: int,
    hidden_dim: int,
    fusion_layers: int,
    fusion_heads: int,
    fusion_dropout: float,
    alpha: float,
    beta: float,
    val_fraction: float,
    best_metric: str,
    checkpoint_selection: str,
    early_stopping_patience: int,
    class_weight_mode: str,
    scheduler: str,
    freeze_image_backbone_epochs: int,
    fusion_mode: str,
    text_shortcut_scale: float,
    max_grad_norm: float,
    grad_explode_threshold: float,
    warmup_checkpoint: str | None = None,
    train_max_samples: int | None = None,
    run_id_suffix: str = "",
    run_id_override: str | None = None,
    run_dir: Path | None = None,
    shared_checkpoint: str | None = None,
) -> int:
    computed_run_id = str(run_id_override) if run_id_override is not None else f"stage2-{dataset.lower()}{run_id_suffix}"
    resolved_run_dir = Path(run_dir) if run_dir is not None else dated_run_root / computed_run_id
    effective_warmup_checkpoint = warmup_checkpoint or shared_checkpoint
    train_args = _build_train_args(
        processed_root=processed_root,
        policy=policy,
        run_root=resolved_run_dir.parent,
        run_id=resolved_run_dir.name,
        dataset=dataset,
        num_classes=num_classes,
        stage=stage,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        num_workers=num_workers,
        hidden_dim=hidden_dim,
        fusion_layers=fusion_layers,
        fusion_heads=fusion_heads,
        fusion_dropout=fusion_dropout,
        alpha=alpha,
        beta=beta,
        val_fraction=val_fraction,
        best_metric=best_metric,
        checkpoint_selection=checkpoint_selection,
        early_stopping_patience=early_stopping_patience,
        class_weight_mode=class_weight_mode,
        scheduler=scheduler,
        freeze_image_backbone_epochs=freeze_image_backbone_epochs,
        fusion_mode=fusion_mode,
        text_shortcut_scale=text_shortcut_scale,
        max_grad_norm=max_grad_norm,
        grad_explode_threshold=grad_explode_threshold,
        warmup_checkpoint=effective_warmup_checkpoint,
        train_max_samples=train_max_samples,
    )
    train_code = train_main(train_args)
    if train_code != 0:
        return train_code
    return _run_stage_report(run_dir=resolved_run_dir, stage=stage, device=device)


def _write_runner_manifest(
    path: Path,
    *,
    session_ids: Sequence[str],
    dataset: str,
    train_positions: np.ndarray,
    val_positions: np.ndarray,
) -> None:
    train_set = {int(x) for x in np.asarray(train_positions, dtype=np.int64).tolist()}
    val_set = {int(x) for x in np.asarray(val_positions, dtype=np.int64).tolist()}
    if train_set & val_set:
        raise ValueError("runner fold manifest has overlapping train/val indices")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "dataset", "split"])
        writer.writeheader()
        for pos, sid in enumerate(session_ids):
            if pos in train_set:
                split = "train"
            elif pos in val_set:
                split = "val"
            else:
                continue
            writer.writerow({"session_id": str(sid), "dataset": dataset, "split": split})


def _load_level1_model(run_dir: Path, device: torch.device) -> tuple[MobileViTETBertFusionClassifier, bool]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint under {run_dir / 'checkpoints'}")

    model = MobileViTETBertFusionClassifier(
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        vocab_size=int(cfg.get("vocab_size", 30522)),
        fusion_layers=int(cfg.get("fusion_layers", 2)),
        fusion_heads=int(cfg.get("fusion_heads", cfg.get("num_heads", 4))),
        dropout=float(cfg.get("fusion_dropout", 0.1)),
        fusion_mode=str(cfg.get("fusion_mode", "legacy")),
        text_shortcut_scale=float(cfg.get("text_shortcut_scale", 0.0)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    use_fusion = str(cfg.get("stage", "fusion")) != "warmup"
    return model, use_fusion


def _infer_meta_features(
    *,
    model: MobileViTETBertFusionClassifier,
    use_fusion: bool,
    rgb: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    token_type_ids: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str], dict]:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        schema = {"version": STAGE2_META_SCHEMA_VERSION, "dim": 0, "feature_names": []}
        return np.zeros((0, 0), dtype=np.float32), [], schema

    feature_names: list[str] | None = None
    feature_schema: dict | None = None
    chunks: list[np.ndarray] = []
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            batch_rows = rows[start : start + batch_size]
            out = model(
                torch.from_numpy(rgb[batch_rows]).float().to(device),
                torch.from_numpy(input_ids[batch_rows]).long().to(device),
                torch.from_numpy(attention_mask[batch_rows]).long().to(device),
                torch.from_numpy(token_type_ids[batch_rows]).long().to(device),
                use_fusion=use_fusion,
                return_summary=True,
            )
            flat, names, schema = flatten_meta_feature_blocks(out)
            if feature_names is None:
                feature_names = list(names)
                feature_schema = dict(schema)
                feature_schema["version"] = STAGE2_META_SCHEMA_VERSION
            else:
                if list(names) != feature_names:
                    raise ValueError("runner meta feature names mismatch across batches")
                if int(schema.get("dim", -1)) != int(feature_schema.get("dim", -1)):
                    raise ValueError("runner meta feature dim mismatch across batches")
            chunks.append(flat.astype(np.float32, copy=False))

    assert feature_names is not None
    assert feature_schema is not None
    x = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, int(feature_schema["dim"])), dtype=np.float32)
    return x, feature_names, feature_schema


def _write_meta_artifact(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    feature_schema: dict,
    fold_ids: np.ndarray,
    split_provenance: dict,
) -> None:
    schema = dict(feature_schema)
    schema["version"] = STAGE2_META_SCHEMA_VERSION
    schema["dim"] = int(np.asarray(x).shape[1] if np.asarray(x).ndim == 2 else 0)
    schema["feature_names"] = [str(name) for name in feature_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int32),
        feature_names=np.asarray(schema["feature_names"], dtype=np.str_),
        feature_schema=np.array(json.dumps(schema, ensure_ascii=False, sort_keys=True), dtype=np.str_),
        feature_schema_version=np.array(STAGE2_META_SCHEMA_VERSION, dtype=np.str_),
        fold_ids=np.asarray(fold_ids, dtype=np.int32),
        split_provenance=np.array(json.dumps(split_provenance, ensure_ascii=False, sort_keys=True), dtype=np.str_),
    )


def _generate_level2_meta_artifacts(
    *,
    run_dir: Path,
    processed_root: Path,
    policy: str,
    dataset: str,
    num_classes: int,
    lr: float,
    seed: int,
    device: str,
    num_workers: int,
    batch_size: int,
    hidden_dim: int,
    fusion_layers: int,
    fusion_heads: int,
    fusion_dropout: float,
    alpha: float,
    beta: float,
    val_fraction: float,
    best_metric: str,
    checkpoint_selection: str,
    early_stopping_patience: int,
    class_weight_mode: str,
    scheduler: str,
    freeze_image_backbone_epochs: int,
    fusion_mode: str,
    text_shortcut_scale: float,
    max_grad_norm: float,
    grad_explode_threshold: float,
    n_splits: int,
    oof_epochs: int,
    train_max_samples: int | None = None,
) -> int:
    data = load_policy_multimodal_data(
        processed_root=processed_root,
        policy=policy,
        datasets=[dataset],
        label_mode="multiclass",
    )
    if int(data["y"].shape[0]) == 0:
        raise ValueError(f"empty dataset for level2 artifact generation: {dataset}")

    split = np.asarray(data["split"]).astype(str)
    train_val_mask = np.isin(split, np.asarray(["train", "val"]))
    train_val_rows = np.where(train_val_mask)[0]
    if train_val_rows.size < 2:
        raise ValueError("runner_kfold_oof requires at least 2 train/val samples")

    requested_splits = max(2, int(n_splits))
    effective_splits = min(requested_splits, int(train_val_rows.size))
    if effective_splits < 2:
        raise ValueError("runner_kfold_oof requires effective n_splits >= 2")

    requested_device, resolved_device, _ = resolve_runtime_device(device)
    _ = requested_device
    torch_device = torch.device(resolved_device)
    session_ids_train_val = [str(x) for x in np.asarray(data["session_id"])[train_val_rows].tolist()]

    folds_root = run_dir / "level2_oof_folds"
    manifests_root = run_dir / "meta_features" / "_runner_manifests"
    kfold = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    oof_x: np.ndarray | None = None
    oof_feature_names: list[str] | None = None
    oof_schema: dict | None = None
    fold_ids = np.full((train_val_rows.size,), -1, dtype=np.int32)
    y_full = np.asarray(data["y"], dtype=np.int32)

    for fold_id, (train_pos, val_pos) in enumerate(kfold.split(train_val_rows)):
        manifest_path = manifests_root / f"fold_{fold_id:02d}.csv"
        _write_runner_manifest(
            manifest_path,
            session_ids=session_ids_train_val,
            dataset=dataset,
            train_positions=train_pos,
            val_positions=val_pos,
        )
        fold_run_id = f"{run_dir.name}-l2-fold{fold_id:02d}"
        fold_train_args = _build_train_args(
            processed_root=processed_root,
            policy=policy,
            run_root=folds_root,
            run_id=fold_run_id,
            dataset=dataset,
            num_classes=num_classes,
            stage="fusion",
            epochs=oof_epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + fold_id + 1,
            device=device,
            num_workers=num_workers,
            hidden_dim=hidden_dim,
            fusion_layers=fusion_layers,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            alpha=alpha,
            beta=beta,
            val_fraction=val_fraction,
            best_metric=best_metric,
            checkpoint_selection=checkpoint_selection,
            early_stopping_patience=early_stopping_patience,
            class_weight_mode=class_weight_mode,
            scheduler=scheduler,
            freeze_image_backbone_epochs=freeze_image_backbone_epochs,
            fusion_mode=fusion_mode,
            text_shortcut_scale=text_shortcut_scale,
            max_grad_norm=max_grad_norm,
            grad_explode_threshold=grad_explode_threshold,
            train_max_samples=train_max_samples,
            session_filter_manifest=manifest_path,
        )
        code = train_main(fold_train_args)
        if code != 0:
            return int(code)

        fold_run_dir = folds_root / fold_run_id
        fold_model, fold_use_fusion = _load_level1_model(fold_run_dir, device=torch_device)
        val_rows = train_val_rows[np.asarray(val_pos, dtype=np.int64)]
        fold_x, fold_feature_names, fold_schema = _infer_meta_features(
            model=fold_model,
            use_fusion=fold_use_fusion,
            rgb=np.asarray(data["rgb"]),
            input_ids=np.asarray(data["input_ids"]),
            attention_mask=np.asarray(data["attention_mask"]),
            token_type_ids=np.asarray(data["token_type_ids"]),
            indices=val_rows,
            batch_size=batch_size,
            device=torch_device,
        )
        if oof_x is None:
            oof_x = np.zeros((train_val_rows.size, fold_x.shape[1]), dtype=np.float32)
            oof_feature_names = list(fold_feature_names)
            oof_schema = dict(fold_schema)
            oof_schema["version"] = STAGE2_META_SCHEMA_VERSION
        else:
            if fold_x.shape[1] != oof_x.shape[1]:
                raise ValueError("OOF meta feature dim mismatch between folds")
            if list(fold_feature_names) != oof_feature_names:
                raise ValueError("OOF meta feature names mismatch between folds")
        oof_x[np.asarray(val_pos, dtype=np.int64)] = fold_x
        fold_ids[np.asarray(val_pos, dtype=np.int64)] = int(fold_id)

    if oof_x is None or oof_feature_names is None or oof_schema is None:
        raise ValueError("runner_kfold_oof produced no OOF features")
    if np.any(fold_ids < 0):
        raise ValueError("runner_kfold_oof failed to assign fold_ids for all train/val samples")

    level1_model, level1_use_fusion = _load_level1_model(run_dir, device=torch_device)
    eval_rows = np.where(split == "test")[0]
    eval_split = "test"
    eval_source = "manifest:test"
    if eval_rows.size == 0:
        eval_rows = np.where(~train_val_mask)[0]
        eval_split = "holdout"
        eval_source = "manifest:holdout"
    if eval_rows.size == 0:
        raise ValueError("missing evaluation split for level2 meta export")

    eval_x, eval_feature_names, eval_schema = _infer_meta_features(
        model=level1_model,
        use_fusion=level1_use_fusion,
        rgb=np.asarray(data["rgb"]),
        input_ids=np.asarray(data["input_ids"]),
        attention_mask=np.asarray(data["attention_mask"]),
        token_type_ids=np.asarray(data["token_type_ids"]),
        indices=eval_rows,
        batch_size=batch_size,
        device=torch_device,
    )
    if list(eval_feature_names) != oof_feature_names:
        raise ValueError("level2 eval meta feature names mismatch against OOF features")
    if int(eval_schema.get("dim", -1)) != int(oof_schema.get("dim", -1)):
        raise ValueError("level2 eval meta feature dim mismatch against OOF features")

    meta_dir = run_dir / "meta_features"
    _write_meta_artifact(
        meta_dir / "oof_meta_train.npz",
        x=oof_x,
        y=y_full[train_val_rows],
        feature_names=oof_feature_names,
        feature_schema=oof_schema,
        fold_ids=fold_ids,
        split_provenance={
            "generator": "runner_kfold_oof",
            "split": "train_val",
            "n_splits": int(effective_splits),
            "owner": "src.experiments.stage2_multiclass",
            "level1_run_dir": str(run_dir),
        },
    )
    _write_meta_artifact(
        meta_dir / "meta_test.npz",
        x=eval_x,
        y=y_full[eval_rows],
        feature_names=oof_feature_names,
        feature_schema=oof_schema,
        fold_ids=np.full((eval_rows.size,), -1, dtype=np.int32),
        split_provenance={
            "generator": "runner_holdout_export",
            "split": eval_split,
            "source": eval_source,
            "owner": "src.experiments.stage2_multiclass",
            "level1_run_dir": str(run_dir),
        },
    )
    return 0


def _run_level2_stacking(
    *,
    run_dir: Path,
    n_splits: int,
    oof_epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    num_workers: int,
) -> int:
    meta_artifacts_dir = run_dir / "meta_features"
    return stacking_main(
        [
            "--run-dir",
            str(run_dir),
            "--meta-artifacts-dir",
            str(meta_artifacts_dir),
            "--n-splits",
            str(n_splits),
            "--oof-epochs",
            str(oof_epochs),
            "--batch-size",
            str(batch_size),
            "--device",
            str(device),
            "--num-workers",
            str(num_workers),
            "--seed",
            str(seed),
        ]
    )


def _resolve_final_metric_source(run_dir: Path) -> Path:
    _, metric_path = resolve_canonical_final_metric_source_and_path(run_dir)
    return metric_path


def run_stage2_shared_stage_a(*, processed_root: Path, policy: str, layout, args) -> dict:
    return train_run_stage2_shared_stage_a(
        processed_root=processed_root,
        policy=policy,
        layout=layout,
        args=args,
        dataset_to_indices={name: [] for name in STAGE2_DATASET_ORDER},
        batch_size=args.batch_size,
        current={name: 1.0 for name in STAGE2_DATASET_ORDER},
        reference={name: 1.0 for name in STAGE2_DATASET_ORDER},
    )


def run_stage2_stage_b(*, dataset: str, num_classes: int, layout, args, shared_checkpoint: str) -> dict:
    run_dir = layout.stage_b_run_dirs[dataset]
    recipe = {
        "processed_root": Path(args.processed_root),
        "policy": args.policy,
        "dated_run_root": layout.root_dir,
        "run_dir": run_dir,
        "shared_checkpoint": shared_checkpoint,
        "stage": "fusion",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "hidden_dim": args.hidden_dim,
        "fusion_layers": args.fusion_layers,
        "fusion_heads": args.fusion_heads,
        "fusion_dropout": args.fusion_dropout,
        "alpha": args.alpha,
        "beta": args.beta,
        "val_fraction": args.val_fraction,
        "best_metric": args.best_metric,
        "checkpoint_selection": args.checkpoint_selection,
        "early_stopping_patience": args.early_stopping_patience,
        "class_weight_mode": args.class_weight_mode,
        "scheduler": args.scheduler,
        "freeze_image_backbone_epochs": args.freeze_image_backbone_epochs,
        "fusion_mode": args.fusion_mode,
        "text_shortcut_scale": args.text_shortcut_scale,
        "max_grad_norm": args.max_grad_norm,
        "grad_explode_threshold": args.grad_explode_threshold,
        "run_id_override": run_dir.name,
    }

    return run_stage_b_dataset_finetune(
        dataset=dataset,
        num_classes=num_classes,
        run_dir=run_dir,
        shared_checkpoint=shared_checkpoint,
        recipe=recipe,
        runner=_run_stage2_task,
    )


def main(argv: Iterable[str] | None = None) -> int:
    arg_list = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(description="Build stage2 multiclass task list")
    parser.add_argument("--output", default="outputs/protocol/stage2_tasks.json")
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--processed-root")
    parser.add_argument("--policy", default="session_full")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--stage", default="fusion", choices=["warmup", "fusion", "stacking", "moe"])
    parser.add_argument("--meta-classifier", default="none", choices=["none", "stacking"])
    parser.add_argument("--level2-impl", default="runner_kfold_oof", choices=["runner_kfold_oof"])
    parser.add_argument("--level2-n-splits", type=int, default=3)
    parser.add_argument("--level2-oof-epochs", type=int, default=2)
    parser.add_argument("--level3-router", default="none", choices=["none", "moe"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", "--num-heads", dest="fusion_heads", type=int, default=4)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--best-metric", default="val_macro_f1", choices=["val_macro_f1", "val_acc"])
    parser.add_argument("--checkpoint-selection", default="best_metric", choices=["best_metric", "score_optimized"])
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--class-weight-mode", default="none", choices=["none", "balanced"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    parser.add_argument("--freeze-image-backbone-epochs", type=int, default=0)
    parser.add_argument("--fusion-mode", default="legacy", choices=["legacy", "residual_enhancer"])
    parser.add_argument("--text-shortcut-scale", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--grad-explode-threshold", type=float, default=1e4)
    parser.add_argument("--ustc-train-limits", nargs="+", type=int, default=[4000, 3000, 2000])
    parser.add_argument("--skip-ustc-limited", action="store_true", default=False)
    args = parser.parse_args(arg_list)
    if args.level2_n_splits < 2:
        raise ValueError("--level2-n-splits must be >= 2")
    if args.level2_oof_epochs < 1:
        raise ValueError("--level2-oof-epochs must be >= 1")
    if args.meta_classifier != "none" and args.stage != "fusion":
        raise ValueError("--meta-classifier is only supported together with --stage fusion")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0")

    fusion_mode_flag_set = bool(arg_list is not None and "--fusion-mode" in arg_list)
    text_shortcut_flag_set = bool(arg_list is not None and "--text-shortcut-scale" in arg_list)
    early_stopping_flag_set = bool(arg_list is not None and "--early-stopping-patience" in arg_list)
    if args.execute and args.stage == "fusion":
        if not fusion_mode_flag_set and args.fusion_mode == "legacy":
            args.fusion_mode = "residual_enhancer"
        if not text_shortcut_flag_set and args.text_shortcut_scale == 0.0:
            args.text_shortcut_scale = 0.5
        if not early_stopping_flag_set and args.early_stopping_patience == 0:
            args.early_stopping_patience = 5

    tasks = build_stage2_tasks()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.execute:
        return 0
    if not args.processed_root:
        raise ValueError("--processed-root is required when --execute is enabled")

    processed_root = Path(args.processed_root)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_date = current_run_date_partition()
    layout = build_stage2_run_layout(run_root=run_root, run_date=run_date)
    layout.root_dir.mkdir(parents=True, exist_ok=True)

    shared_result = run_stage2_shared_stage_a(processed_root=processed_root, policy=args.policy, layout=layout, args=args)
    shared_checkpoint = str(shared_result.get("best_checkpoint", layout.shared_run_dir / "checkpoints" / "best.ckpt"))

    acceptance = []
    for task in tasks:
        result = run_stage2_stage_b(
            dataset=str(task["dataset"]),
            num_classes=int(task["num_classes"]),
            layout=layout,
            args=args,
            shared_checkpoint=shared_checkpoint,
        )
        acceptance.append(result)
        if int(result["code"]) != 0:
            break

    layout.acceptance_path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(int(item.get("code", 1)) == 0 for item in acceptance) else int(
        next(item["code"] for item in acceptance if int(item.get("code", 0)) != 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())
