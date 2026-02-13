"""Common training utilities for fusion and stacking experiments."""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from fusion_dataset import FusionDataset, resolve_dataset_dirs

LOGGER = logging.getLogger(__name__)


@dataclass
class RunPaths:
    run_id: str
    root: Path
    logs_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    reports_dir: Path
    figures_dir: Path
    stacking_dir: Path


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(device: str) -> torch.device:
    if str(device).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def init_run_dirs(output_dir: str, run_name: str = "") -> RunPaths:
    root = Path(output_dir).resolve()
    run_id = run_name.strip() if run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / run_id

    logs = run_root / "logs"
    ckpt = run_root / "checkpoints"
    metrics = run_root / "metrics"
    reports = run_root / "reports"
    figures = run_root / "figures"
    stacking = run_root / "stacking"

    for p in (logs, ckpt, metrics, reports, figures, stacking):
        p.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        root=run_root,
        logs_dir=logs,
        checkpoints_dir=ckpt,
        metrics_dir=metrics,
        reports_dir=reports,
        figures_dir=figures,
        stacking_dir=stacking,
    )


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def compute_class_weights(class_counts: Sequence[int], beta: float = 0.9999) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    weights = np.zeros_like(counts)
    valid = counts > 0
    if np.any(valid):
        effective_num = 1.0 - np.power(beta, counts[valid])
        effective_num = np.clip(effective_num, 1e-12, None)
        weights[valid] = (1.0 - beta) / effective_num
        s = weights[valid].sum()
        if s > 0:
            weights[valid] = weights[valid] * (valid.sum() / s)
    return torch.tensor(weights, dtype=torch.float32)


def create_data_loaders(
    *,
    dataset_root: str,
    dataset_name: str,
    image_size: int,
    char_seq_len: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    class_balance: str,
    use_index_cache: bool,
    rebuild_index_cache: bool,
    r_only: bool,
) -> Tuple[DataLoader, DataLoader, List[str], str]:
    train_img, train_pcap, test_img, test_pcap, resolved_name = resolve_dataset_dirs(dataset_root, dataset_name)

    train_ds = FusionDataset(
        train_img,
        train_pcap,
        image_size=image_size,
        char_seq_len=char_seq_len,
        r_only=r_only,
        use_index_cache=use_index_cache,
        rebuild_index_cache=rebuild_index_cache,
    )
    val_ds = FusionDataset(
        test_img,
        test_pcap,
        image_size=image_size,
        char_seq_len=char_seq_len,
        r_only=r_only,
        use_index_cache=use_index_cache,
        rebuild_index_cache=rebuild_index_cache,
    )

    dl_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        dl_kwargs["persistent_workers"] = True

    sampler = None
    shuffle = True
    if class_balance in ("weighted_sampler", "weighted_sampler_loss"):
        class_weights = compute_class_weights(train_ds.class_counts)
        sample_weights = [float(class_weights[t]) for t in train_ds.targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(train_ds, shuffle=shuffle if sampler is None else False, sampler=sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader, train_ds.classes, resolved_name


def create_criterion(
    *,
    loss_type: str,
    class_balance: str,
    class_counts: Sequence[int],
    device: torch.device,
    focal_gamma: float,
) -> nn.Module:
    class_weight = None
    if class_balance in ("weighted_loss", "weighted_sampler_loss"):
        class_weight = compute_class_weights(class_counts).to(device)

    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weight)
    return nn.CrossEntropyLoss(weight=class_weight)


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)

    class _NoOp:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NoOp()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
) -> Tuple[float, float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for images, tokens, labels in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            tokens = tokens.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            with _autocast(device, use_amp):
                logits = model(images, tokens)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = criterion(logits, labels)

            total_loss += float(loss.item()) * images.size(0)
            pred = torch.argmax(logits, dim=1)
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(pred.detach().cpu().tolist())

    denom = max(len(loader.dataset), 1)
    val_loss = total_loss / denom
    val_acc = accuracy_score(all_true, all_pred) if all_true else 0.0
    val_f1 = f1_score(all_true, all_pred, average="macro") if all_true else 0.0
    return val_loss, val_acc, val_f1, all_true, all_pred


def fit_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    early_stop_metric: str,
    use_amp: bool,
    run_paths: RunPaths,
) -> Tuple[List[Dict[str, float]], Path]:
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_score: Optional[float] = None
    best_state: Optional[dict] = None
    best_epoch = 0
    wait = 0

    history: List[Dict[str, float]] = []

    metric_mode = "max" if early_stop_metric in ("val_f1", "val_acc") else "min"
    best_ckpt_path = run_paths.checkpoints_dir / "best.pt"
    last_ckpt_path = run_paths.checkpoints_dir / "last.pt"

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0

        progress = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for images, tokens, labels in progress:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            tokens = tokens.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(images, tokens)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item()) * images.size(0)
            progress.set_postfix({"loss": f"{float(loss.item()):.4f}"})

        train_loss = train_loss_sum / max(len(train_loader.dataset), 1)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
        }
        history.append(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, last_ckpt_path)

        score = row.get(early_stop_metric)
        if score is None:
            raise ValueError(f"Invalid early_stop_metric: {early_stop_metric}")

        improved = False
        if best_score is None:
            improved = True
        elif metric_mode == "max" and score > best_score:
            improved = True
        elif metric_mode == "min" and score < best_score:
            improved = True

        if improved:
            best_score = score
            best_state = model.state_dict()
            best_epoch = epoch
            wait = 0
            torch.save(checkpoint, best_ckpt_path)
        else:
            wait += 1

        LOGGER.info(
            "Epoch %s/%s - train_loss=%.6f val_loss=%.6f val_acc=%.4f val_f1=%.4f best_%s=%.6f wait=%s",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_acc,
            val_f1,
            early_stop_metric,
            best_score if best_score is not None else float("nan"),
            wait,
        )

        if wait >= int(patience):
            LOGGER.info("Early stopping at epoch=%s (best_epoch=%s)", epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_history_csv(history, run_paths.metrics_dir / "epoch_metrics.csv")
    plot_history(history, run_paths.figures_dir)
    return history, best_ckpt_path


def save_history_csv(history: Sequence[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _plot_curve(xs: Sequence[int], ys: Sequence[float], title: str, ylab: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_history(history: Sequence[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    xs = [int(r["epoch"]) for r in history]
    loss = [float(r["val_loss"]) for r in history]
    acc = [float(r["val_acc"]) for r in history]
    f1 = [float(r["val_f1"]) for r in history]

    _plot_curve(xs, loss, "Validation Loss", "Loss", out_dir / "loss_curve.png")
    _plot_curve(xs, acc, "Validation Accuracy", "Accuracy", out_dir / "acc_curve.png")
    _plot_curve(xs, f1, "Validation Macro-F1", "Macro-F1", out_dir / "f1_curve.png")


def save_reports(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    reports_dir: Path,
    prefix: str = "",
) -> Dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_txt = classification_report(y_true, y_pred, target_names=list(class_names), digits=4)
    txt_name = f"{prefix}classification_report.txt" if prefix else "classification_report.txt"
    with (reports_dir / txt_name).open("w", encoding="utf-8") as f:
        f.write(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    fig_name = f"{prefix}confusion_matrix.png" if prefix else "confusion_matrix.png"
    save_confusion_matrix(cm, class_names, reports_dir / fig_name)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics


def save_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out_path: Path) -> None:
    plt.figure(figsize=(max(8, len(labels) * 0.4), max(6, len(labels) * 0.35)))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_final_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
