from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

from src.common.structured_logging import format_log_line
from src.models.fusion_model import MobileViTETBertFusionClassifier
from src.pipeline_data import load_policy_multimodal_data
from src.run_dir import build_timestamped_run_identity
from src.runtime_device import resolve_runtime_device


def _build_run_identity(run_root: Path) -> tuple[str, Path]:
    return build_timestamped_run_identity(run_root=run_root, now=datetime.now())


def _sha8(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()[:8]


def _git_commit_short(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = out.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def _class_dist_text(y: np.ndarray, num_classes: int) -> str:
    if y.size == 0:
        return "none"
    counts = np.bincount(y.astype(np.int64, copy=False), minlength=max(1, num_classes))
    return ",".join(f"{i}:{int(c)}" for i, c in enumerate(counts))


def _loss_and_logits(
    out: dict,
    y: torch.Tensor,
    stage: str,
    ce: nn.Module,
    alpha: float,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stage == "warmup":
        loss = 0.5 * ce(out["logits_img"], y) + 0.5 * ce(out["logits_tls"], y)
        pred_logits = 0.5 * (out["logits_img"] + out["logits_tls"])
    else:
        loss = ce(out["logits_fuse"], y) + alpha * ce(out["logits_img"], y) + beta * ce(out["logits_tls"], y)
        pred_logits = out["logits_fuse"]
    return loss, pred_logits


def _confidence_mean(out: dict, stage: str) -> float:
    if stage == "warmup":
        logits = 0.5 * (out["logits_img"] + out["logits_tls"])
    else:
        logits = out["logits_fuse"]
    probs = torch.softmax(logits, dim=1)
    return float(probs.max(dim=1).values.mean().item())


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage: str,
    alpha: float,
    beta: float,
    show_progress: bool,
    epoch: int,
    total_epochs: int,
    lr: float,
) -> tuple[float, float, float, float, float | None]:
    if len(loader) == 0:
        return 0.0, 0.0, 0.0, 0.0, None

    model.eval()
    ce = nn.CrossEntropyLoss()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    fuse_conf_means: List[float] = []
    positive_probs: List[np.ndarray] = []
    seen_samples = 0
    correct_samples = 0

    iterator = tqdm(
        loader,
        disable=not show_progress,
        desc=f"val {epoch}/{total_epochs}",
        unit="batch",
        position=1,
        leave=False,
    )
    with torch.no_grad():
        for rgb_b, input_ids_b, attn_b, token_type_b, y_b in iterator:
            rgb_b = rgb_b.to(device)
            input_ids_b = input_ids_b.to(device)
            attn_b = attn_b.to(device)
            token_type_b = token_type_b.to(device)
            y_b = y_b.to(device)
            out = model(rgb_b, input_ids_b, attn_b, token_type_b, use_fusion=stage != "warmup")
            loss, pred_logits = _loss_and_logits(out, y_b, stage, ce, alpha, beta)
            pred = pred_logits.argmax(dim=1)
            batch_acc = float((pred == y_b).float().mean().item())
            batch_size = int(y_b.shape[0])

            losses.append(float(loss.item()))
            preds.append(pred.cpu().numpy())
            labels.append(y_b.cpu().numpy())
            fuse_conf_means.append(_confidence_mean(out, stage=stage))
            seen_samples += batch_size
            correct_samples += int((pred == y_b).sum().item())
            if pred_logits.shape[1] == 2:
                probs = torch.softmax(pred_logits, dim=1)[:, 1]
                positive_probs.append(probs.detach().cpu().numpy())
            running_loss = float(np.mean(losses)) if losses else float(loss.item())
            running_acc = float(correct_samples / seen_samples) if seen_samples > 0 else batch_acc
            iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", lr=f"{lr:.2e}")

    y_true = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.int64)
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0, None

    val_loss = float(np.mean(losses))
    val_acc = float(np.mean(y_true == y_pred))
    val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    val_fuse_conf_mean = float(np.mean(fuse_conf_means)) if fuse_conf_means else 0.0
    decision_threshold = None
    if positive_probs:
        threshold, _ = choose_best_binary_threshold(
            positive_probs=np.concatenate(positive_probs, axis=0),
            y_true=y_true,
        )
        decision_threshold = threshold
    return val_loss, val_acc, val_macro_f1, val_fuse_conf_mean, decision_threshold


def _derive_validation_mask_from_train(
    train_mask: np.ndarray,
    y: np.ndarray,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(train_mask)[0]
    if idx.size < 2:
        raise ValueError("cannot derive validation split from fewer than 2 training samples")
    rng = np.random.default_rng(seed)
    labels = y[idx]
    val_idx_parts: List[np.ndarray] = []

    for cls in np.unique(labels):
        cls_idx = idx[labels == cls].copy()
        rng.shuffle(cls_idx)
        if cls_idx.size < 2:
            continue
        cls_val_n = int(round(float(cls_idx.size) * float(val_fraction)))
        cls_val_n = max(1, cls_val_n)
        cls_val_n = min(cls_val_n, cls_idx.size - 1)
        if cls_val_n > 0:
            val_idx_parts.append(cls_idx[:cls_val_n])

    if val_idx_parts:
        val_idx = np.concatenate(val_idx_parts, axis=0)
    else:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_val = int(round(float(idx.size) * float(val_fraction)))
        n_val = max(1, n_val)
        n_val = min(n_val, idx.size - 1)
        val_idx = shuffled[:n_val]

    new_train_mask = train_mask.copy()
    new_train_mask[val_idx] = False
    val_mask = np.zeros_like(train_mask, dtype=bool)
    val_mask[val_idx] = True
    return new_train_mask, val_mask


def choose_best_binary_threshold(positive_probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    probs = np.asarray(positive_probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if probs.size == 0 or labels.size == 0 or probs.size != labels.size:
        return 0.5, 0.0
    if np.unique(labels).size < 2:
        pred = (probs >= 0.5).astype(np.int64)
        return 0.5, float(np.mean(pred == labels))

    candidates = sorted({0.5, *[float(x) for x in np.unique(probs).tolist()]})
    best_threshold = 0.5
    best_acc = -1.0
    best_distance = float("inf")
    for threshold in candidates:
        pred = (probs >= threshold).astype(np.int64)
        acc = float(np.mean(pred == labels))
        distance = abs(float(threshold) - 0.5)
        if acc > best_acc or (acc == best_acc and distance < best_distance):
            best_threshold = float(threshold)
            best_acc = acc
            best_distance = distance
    return best_threshold, best_acc


def _select_best_metric_value(best_metric: str, val_acc: float, val_macro_f1: float) -> float:
    if best_metric == "val_acc":
        return float(val_acc)
    if best_metric == "val_macro_f1":
        return float(val_macro_f1)
    raise ValueError(f"unsupported best metric: {best_metric}")


def _select_checkpoint_tuple(
    checkpoint_selection: str,
    best_metric: str,
    val_acc: float,
    val_macro_f1: float,
    decision_threshold: float | None,
) -> tuple[float, float, float]:
    if checkpoint_selection == "best_metric":
        primary = _select_best_metric_value(best_metric=best_metric, val_acc=val_acc, val_macro_f1=val_macro_f1)
        secondary = float(val_macro_f1) if best_metric == "val_acc" else float(val_acc)
        return (primary, secondary, 0.0)
    if checkpoint_selection == "score_optimized":
        threshold_stability = -abs(float(decision_threshold) - 0.5) if decision_threshold is not None else -1.0
        return (float(val_acc), float(val_macro_f1), threshold_stability)
    raise ValueError(f"unsupported checkpoint selection: {checkpoint_selection}")


def _limit_training_samples(
    train_mask: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_samples: int,
) -> np.ndarray:
    idx = np.where(train_mask)[0]
    if idx.size <= max_samples:
        return train_mask
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")

    rng = np.random.default_rng(seed)
    labels = y[idx]
    selected: List[int] = []
    classes = np.unique(labels)
    per_class_target = max(1, max_samples // max(1, len(classes)))

    for cls in classes:
        cls_idx = idx[labels == cls]
        if cls_idx.size == 0:
            continue
        take = min(cls_idx.size, per_class_target)
        picked = rng.choice(cls_idx, size=take, replace=False)
        selected.extend(int(x) for x in picked.tolist())

    if len(selected) > max_samples:
        selected = rng.choice(np.asarray(selected, dtype=np.int64), size=max_samples, replace=False).tolist()
    elif len(selected) < max_samples:
        selected_set = set(selected)
        remaining = [int(i) for i in idx.tolist() if int(i) not in selected_set]
        need = max_samples - len(selected)
        if need > 0 and remaining:
            add = rng.choice(np.asarray(remaining, dtype=np.int64), size=min(need, len(remaining)), replace=False)
            selected.extend(int(x) for x in add.tolist())

    new_mask = np.zeros_like(train_mask, dtype=bool)
    new_mask[np.asarray(selected, dtype=np.int64)] = True
    return new_mask


def _dispatch_stage(
    stage: str,
    run_dir: Path,
    args: argparse.Namespace,
    log: Callable[[str, str, str, dict], None],
) -> int:
    if stage == "stacking":
        from src.stacking import main as stacking_main

        log("info", "model", "stage_dispatch_start", {"target": "stacking", "run_dir": str(run_dir)})
        code = stacking_main(
            [
                "--run-dir",
                str(run_dir),
                "--n-splits",
                str(args.stacking_n_splits),
                "--oof-epochs",
                str(args.stacking_oof_epochs),
                "--batch-size",
                str(args.batch_size),
                "--device",
                str(args.device),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
            ]
        )
        if code != 0:
            log("error", "model", "stage_dispatch_failed", {"target": "stacking", "code": code})
            return code
        log("success", "model", "stage_dispatch_done", {"target": "stacking", "code": code})
        return 0

    if stage == "moe":
        from src.moe import main as moe_main

        log("info", "model", "stage_dispatch_start", {"target": "moe", "run_dir": str(run_dir)})
        code = moe_main(
            [
                "--run-dir",
                str(run_dir),
                "--epochs",
                str(args.moe_epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                str(args.device),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
            ]
        )
        if code != 0:
            log("error", "model", "stage_dispatch_failed", {"target": "moe", "code": code})
            return code
        log("success", "model", "stage_dispatch_done", {"target": "moe", "code": code})
        return 0

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train TLS family fusion model")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--policy", default="strict")
    parser.add_argument("--stage", default="warmup", choices=["warmup", "fusion", "stacking", "moe"])
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--session-filter-manifest", default=None)
    parser.add_argument("--label-mode", default="multiclass", choices=["multiclass", "binary"])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", "--num-heads", dest="fusion_heads", type=int, default=4)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-mode", default="legacy", choices=["legacy", "residual_enhancer"])
    parser.add_argument("--text-shortcut-scale", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--grad-explode-threshold", type=float, default=1e4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--best-metric", default="val_macro_f1", choices=["val_macro_f1", "val_acc"])
    parser.add_argument("--checkpoint-selection", default="best_metric", choices=["best_metric", "score_optimized"])
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--warmup-checkpoint", default=None)
    parser.add_argument("--class-weight-mode", default="none", choices=["none", "balanced"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    parser.add_argument("--freeze-image-backbone-epochs", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--stacking-n-splits", type=int, default=3)
    parser.add_argument("--stacking-oof-epochs", type=int, default=2)
    parser.add_argument("--moe-epochs", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0")

    requested_device, resolved_device, device_fallback = resolve_runtime_device(args.device)
    device = torch.device(resolved_device)

    run_root = Path(args.run_root)
    if args.run_id:
        run_id = args.run_id
        run_dir = run_root / run_id
    else:
        run_id, run_dir = _build_run_identity(run_root)
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.csv"
    cfg_path = run_dir / "config.yaml"
    if log_path.exists():
        log_path.unlink()

    def log(level: str, module: str, event: str, kv: dict) -> None:
        line = format_log_line(level=level, module=module, event=event, kv=kv)
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    data = load_policy_multimodal_data(
        args.processed_root,
        args.policy,
        datasets=args.datasets,
        label_mode=args.label_mode,
        session_filter_manifest=args.session_filter_manifest,
    )
    rgb = data["rgb"]
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    y = data["y"]
    split = data["split"]
    if rgb.shape[0] == 0:
        log("error", "data", "empty_dataset", {"processed_root": args.processed_root, "policy": args.policy})
        return 2

    inferred_classes = int(np.max(y)) + 1
    if args.num_classes is not None and args.num_classes < inferred_classes:
        log("error", "data", "invalid_num_classes", {"configured": args.num_classes, "required": inferred_classes})
        return 2
    num_classes = int(args.num_classes) if args.num_classes is not None else inferred_classes
    vocab_size = int(max(30522, int(input_ids.max()) + 1)) if input_ids.size > 0 else 30522
    cfg = {
        "run_id": run_id,
        "processed_root": args.processed_root,
        "policy": args.policy,
        "datasets": list(args.datasets) if args.datasets else [],
        "session_filter_manifest": args.session_filter_manifest,
        "label_mode": args.label_mode,
        "stage": args.stage,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "model_type": "MobileViTETBertFusionClassifier",
        "hidden_dim": args.hidden_dim,
        "fusion_layers": args.fusion_layers,
        "fusion_heads": args.fusion_heads,
        "fusion_dropout": args.fusion_dropout,
        "fusion_mode": args.fusion_mode,
        "text_shortcut_scale": args.text_shortcut_scale,
        "num_heads": args.fusion_heads,
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "alpha": args.alpha,
        "beta": args.beta,
        "device_requested": requested_device,
        "device": resolved_device,
        "device_fallback": device_fallback,
        "num_workers": args.num_workers,
        "best_metric": args.best_metric,
        "checkpoint_selection": args.checkpoint_selection,
        "early_stopping_patience": args.early_stopping_patience,
        "warmup_checkpoint": args.warmup_checkpoint,
        "class_weight_mode": args.class_weight_mode,
        "scheduler": args.scheduler,
        "freeze_image_backbone_epochs": args.freeze_image_backbone_epochs,
        "max_grad_norm": args.max_grad_norm,
        "grad_explode_threshold": args.grad_explode_threshold,
        "val_fraction": args.val_fraction,
        "train_max_samples": args.train_max_samples,
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    git_commit = _git_commit_short(Path(__file__).resolve().parents[1])
    log("info", "time", "run_bootstrap", {"run_id": run_id, "git_commit": git_commit})
    log(
        "info",
        "model",
        "config_summary",
        {
            "stage": args.stage,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "fusion_layers": args.fusion_layers,
            "fusion_heads": args.fusion_heads,
            "fusion_dropout": args.fusion_dropout,
            "fusion_mode": args.fusion_mode,
            "text_shortcut_scale": args.text_shortcut_scale,
            "alpha": args.alpha,
            "beta": args.beta,
            "device_requested": requested_device,
            "device": resolved_device,
            "device_fallback": device_fallback,
            "num_workers": args.num_workers,
            "best_metric": args.best_metric,
            "checkpoint_selection": args.checkpoint_selection,
            "early_stopping_patience": args.early_stopping_patience,
            "class_weight_mode": args.class_weight_mode,
            "scheduler": args.scheduler,
            "freeze_image_backbone_epochs": args.freeze_image_backbone_epochs,
            "max_grad_norm": args.max_grad_norm,
        },
    )
    log(
        "info",
        "data",
        "dataset_stats",
        {
            "samples": int(rgb.shape[0]),
            "families": int(len(np.unique(y))),
            "class_dist": _class_dist_text(y, num_classes),
            "label_mode": args.label_mode,
            "datasets": ",".join(args.datasets) if args.datasets else "all",
        },
    )

    train_mask = split == "train"
    val_mask = split == "val"
    if not np.any(train_mask):
        log("error", "data", "empty_train_split", {"policy": args.policy, "datasets": cfg["datasets"]})
        return 2
    if args.train_max_samples is not None:
        try:
            limited_train_mask = _limit_training_samples(
                train_mask=train_mask,
                y=y,
                seed=args.seed,
                max_samples=args.train_max_samples,
            )
        except ValueError as exc:
            log("error", "data", "invalid_train_max_samples", {"error": str(exc)})
            return 2
        if int(limited_train_mask.sum()) < int(train_mask.sum()):
            log(
                "info",
                "data",
                "train_samples_limited",
                {"before": int(train_mask.sum()), "after": int(limited_train_mask.sum())},
            )
        train_mask = limited_train_mask
    if not np.any(val_mask):
        try:
            train_mask, val_mask = _derive_validation_mask_from_train(
                train_mask=train_mask,
                y=y,
                seed=args.seed,
                val_fraction=args.val_fraction,
            )
        except ValueError as exc:
            log("error", "data", "empty_val_split", {"error": str(exc)})
            return 2
        log(
            "warning",
            "data",
            "val_split_derived_from_train",
            {
                "val_fraction": args.val_fraction,
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
            },
        )

    rgb_train = torch.from_numpy(rgb[train_mask]).float()
    input_train = torch.from_numpy(input_ids[train_mask]).long()
    attn_train = torch.from_numpy(attention_mask[train_mask]).long()
    token_type_train = torch.from_numpy(token_type_ids[train_mask]).long()
    y_train = torch.from_numpy(y[train_mask]).long()

    rgb_val = torch.from_numpy(rgb[val_mask]).float()
    input_val = torch.from_numpy(input_ids[val_mask]).long()
    attn_val = torch.from_numpy(attention_mask[val_mask]).long()
    token_type_val = torch.from_numpy(token_type_ids[val_mask]).long()
    y_val = torch.from_numpy(y[val_mask]).long()

    if rgb_train.shape[0] == 0:
        log("error", "data", "empty_train_dataset", {"policy": args.policy, "stage": args.stage})
        return 2

    train_dataset = TensorDataset(rgb_train, input_train, attn_train, token_type_train, y_train)
    pin_memory = resolved_device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_dataset = TensorDataset(rgb_val, input_val, attn_val, token_type_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = MobileViTETBertFusionClassifier(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        dropout=args.fusion_dropout,
        fusion_mode=args.fusion_mode,
        text_shortcut_scale=args.text_shortcut_scale,
    ).to(device)
    if args.warmup_checkpoint:
        warmup_ckpt = torch.load(args.warmup_checkpoint, map_location="cpu")
        model.load_state_dict(warmup_ckpt["model_state"], strict=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    ce_weight = None
    if args.class_weight_mode == "balanced" and y_train.numel() > 0:
        classes = np.unique(y_train.numpy())
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.numpy())
        full_weights = np.ones((num_classes,), dtype=np.float32)
        for cls, weight in zip(classes.tolist(), weights.tolist()):
            full_weights[int(cls)] = float(weight)
        ce_weight = torch.tensor(full_weights, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=ce_weight)

    log(
        "info",
        "data",
        "train_start",
        {
            "run_id": run_id,
            "train_samples": int(rgb_train.shape[0]),
            "val_samples": int(rgb_val.shape[0]),
            "classes": num_classes,
            "stage": args.stage,
            "device": resolved_device,
            "num_workers": args.num_workers,
        },
    )

    rows: List[dict] = []
    best_value = -1.0
    best_tuple: tuple[float, float, float] | None = None
    show_progress = not args.no_progress
    loss_stage = "warmup" if args.stage == "warmup" else "fusion"
    best_decision_threshold: float | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        freeze_image = epoch <= args.freeze_image_backbone_epochs
        image_backbone = getattr(model, "image_backbone", None)
        if image_backbone is not None:
            for p in image_backbone.parameters():
                p.requires_grad = not freeze_image
        model.train()

        train_losses: List[float] = []
        train_preds: List[np.ndarray] = []
        train_targets: List[np.ndarray] = []
        train_fuse_confidences: List[float] = []
        clipped_steps = 0
        train_seen_samples = 0
        train_correct_samples = 0

        iterator = tqdm(
            train_loader,
            disable=not show_progress,
            desc=f"train {epoch}/{args.epochs}",
            unit="batch",
            position=0,
            leave=False,
        )
        for rgb_b, input_ids_b, attn_b, token_type_b, y_b in iterator:
            rgb_b = rgb_b.to(device)
            input_ids_b = input_ids_b.to(device)
            attn_b = attn_b.to(device)
            token_type_b = token_type_b.to(device)
            y_b = y_b.to(device)
            optim.zero_grad()
            out = model(rgb_b, input_ids_b, attn_b, token_type_b, use_fusion=loss_stage != "warmup")
            loss, pred_logits = _loss_and_logits(out, y_b, loss_stage, ce, args.alpha, args.beta)
            if not torch.isfinite(loss):
                log("error", "model", "nan_loss", {"epoch": epoch, "stage": args.stage})
                return 3

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm))
            if not np.isfinite(grad_norm):
                log("error", "model", "invalid_grad_norm", {"epoch": epoch, "stage": args.stage})
                return 3
            if grad_norm > args.grad_explode_threshold:
                log(
                    "error",
                    "model",
                    "gradient_explosion",
                    {"epoch": epoch, "grad_norm": f"{grad_norm:.4f}", "threshold": args.grad_explode_threshold},
                )
                return 3
            if grad_norm > args.max_grad_norm:
                clipped_steps += 1

            optim.step()
            pred = pred_logits.argmax(dim=1)
            batch_acc = float((pred == y_b).float().mean().item())
            batch_size = int(y_b.shape[0])

            train_losses.append(float(loss.item()))
            train_preds.append(pred.cpu().numpy())
            train_targets.append(y_b.cpu().numpy())
            train_fuse_confidences.append(_confidence_mean(out, stage=loss_stage))
            train_seen_samples += batch_size
            train_correct_samples += int((pred == y_b).sum().item())
            running_loss = float(np.mean(train_losses)) if train_losses else float(loss.item())
            running_acc = float(train_correct_samples / train_seen_samples) if train_seen_samples > 0 else batch_acc
            iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", lr=f"{args.lr:.2e}")

        if clipped_steps > 0:
            log(
                "warning",
                "model",
                "grad_clipped",
                {
                    "epoch": epoch,
                    "steps": clipped_steps,
                    "max_grad_norm": args.max_grad_norm,
                },
            )
        if scheduler is not None:
            scheduler.step()

        y_true = np.concatenate(train_targets, axis=0) if train_targets else np.zeros((0,), dtype=np.int64)
        y_pred = np.concatenate(train_preds, axis=0) if train_preds else np.zeros((0,), dtype=np.int64)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = float(np.mean(y_true == y_pred)) if y_true.size > 0 else 0.0
        train_macro_f1 = (
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size > 0 else 0.0
        )
        train_fuse_conf_mean = float(np.mean(train_fuse_confidences)) if train_fuse_confidences else 0.0

        val_loss, val_acc, val_macro_f1, val_fuse_conf_mean, val_decision_threshold = _evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            stage=loss_stage,
            alpha=args.alpha,
            beta=args.beta,
            show_progress=show_progress,
            epoch=epoch,
            total_epochs=args.epochs,
            lr=args.lr,
        )
        epoch_time = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_macro_f1,
            "train_fuse_conf_mean": train_fuse_conf_mean,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_fuse_conf_mean": val_fuse_conf_mean,
            "val_decision_threshold": val_decision_threshold,
            "lr": args.lr,
            "epoch_time": epoch_time,
        }
        rows.append(row)

        log(
            "success",
            "metric",
            "epoch_done",
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "train_macroF1": f"{train_macro_f1:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "val_macroF1": f"{val_macro_f1:.4f}",
                "val_decision_threshold": "none" if val_decision_threshold is None else f"{val_decision_threshold:.4f}",
                "fuse_conf_mean": f"{val_fuse_conf_mean:.4f}",
                "lr": args.lr,
                "time_s": f"{epoch_time:.2f}",
            },
        )

        last_path = ckpt_dir / "last.ckpt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "decision_threshold": val_decision_threshold,
            },
            last_path,
        )
        log("success", "save", "checkpoint_saved", {"path": str(last_path), "sha8": _sha8(last_path)})

        current_best_value = _select_best_metric_value(
            best_metric=args.best_metric,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1,
        )
        current_best_tuple = _select_checkpoint_tuple(
            checkpoint_selection=args.checkpoint_selection,
            best_metric=args.best_metric,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1,
            decision_threshold=val_decision_threshold,
        )
        if best_tuple is None or current_best_tuple > best_tuple:
            best_value = current_best_value
            best_tuple = current_best_tuple
            best_decision_threshold = val_decision_threshold
            cfg["decision_threshold"] = best_decision_threshold
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            best_path = ckpt_dir / "best.ckpt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "best_metric": args.best_metric,
                    "best_metric_value": best_value,
                    "best_val_macro_f1": val_macro_f1,
                    "decision_threshold": best_decision_threshold,
                },
                best_path,
            )
            log(
                "success",
                "save",
                "best_checkpoint_saved",
                {
                    "path": str(best_path),
                    "best_metric": args.best_metric,
                    "best_metric_value": f"{best_value:.4f}",
                    "best_val_macro_f1": f"{val_macro_f1:.4f}",
                    "sha8": _sha8(best_path),
                },
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                log(
                    "warning",
                    "model",
                    "early_stopping_triggered",
                    {
                        "epoch": epoch,
                        "best_metric": args.best_metric,
                        "best_metric_value": f"{best_value:.4f}",
                        "patience": args.early_stopping_patience,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                )
                break

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    log("success", "save", "metrics_saved", {"path": str(metrics_path)})

    if args.stage in {"stacking", "moe"}:
        stage_code = _dispatch_stage(stage=args.stage, run_dir=run_dir, args=args, log=log)
        if stage_code != 0:
            return stage_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
