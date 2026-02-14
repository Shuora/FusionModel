"""Common training utilities for fusion and stacking experiments."""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from contextlib import nullcontext
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
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(max(label_smoothing, 0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
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


def _format_class_distribution(classes: Sequence[str], counts: Sequence[int]) -> str:
    parts = []
    for name, cnt in zip(classes, counts):
        parts.append(f"{name}:{int(cnt)}")
    return ", ".join(parts)


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
    LOGGER.info("加载数据集: %s", resolved_name)
    LOGGER.info("训练目录 image=%s pcap=%s", train_img, train_pcap)
    LOGGER.info("验证目录 image=%s pcap=%s", test_img, test_pcap)

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
    LOGGER.info("DataLoader 参数: batch_size=%s num_workers=%s pin_memory=%s", batch_size, num_workers, pin_memory)
    LOGGER.info("训练样本=%s 验证样本=%s 类别数=%s", len(train_ds), len(val_ds), len(train_ds.classes))
    LOGGER.info("训练集类别分布: %s", _format_class_distribution(train_ds.classes, train_ds.class_counts))
    LOGGER.info("验证集类别分布: %s", _format_class_distribution(val_ds.classes, val_ds.class_counts))
    if sampler is not None:
        LOGGER.info("启用采样策略: %s", class_balance)
    return train_loader, val_loader, train_ds.classes, resolved_name


def create_criterion(
    *,
    loss_type: str,
    class_balance: str,
    class_counts: Sequence[int],
    device: torch.device,
    focal_gamma: float,
    label_smoothing: float = 0.0,
) -> nn.Module:
    class_weight = None
    if class_balance in ("weighted_loss", "weighted_sampler_loss"):
        class_weight = compute_class_weights(class_counts).to(device)

    if loss_type == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            weight=class_weight,
            label_smoothing=label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(max(label_smoothing, 0.0)))


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    use_amp = bool(enabled)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def _make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


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
    scheduler: Optional[object] = None,
    scheduler_mode: str = "none",
    grad_clip_norm: float = 0.0,
    start_epoch: int = 1,
    history: Optional[List[Dict[str, float]]] = None,
    save_artifacts: bool = True,
    stage_name: str = "",
) -> Tuple[List[Dict[str, float]], Path]:
    use_amp_runtime = bool(use_amp and device.type == "cuda")
    scaler = _make_grad_scaler(device, use_amp_runtime)

    best_score: Optional[float] = None
    best_state: Optional[dict] = None
    best_epoch = 0
    wait = 0

    if history is None:
        history = []

    metric_mode = "max" if early_stop_metric in ("val_f1", "val_acc") else "min"
    best_ckpt_path = run_paths.checkpoints_dir / "best.pt"
    last_ckpt_path = run_paths.checkpoints_dir / "last.pt"
    stage_prefix = f"[{stage_name}] " if stage_name else ""
    LOGGER.info(
        "%s开始训练: epochs=%s start_epoch=%s patience=%s early_stop_metric=%s use_amp=%s scheduler=%s grad_clip_norm=%s",
        stage_prefix,
        epochs,
        start_epoch,
        patience,
        early_stop_metric,
        use_amp_runtime,
        scheduler_mode,
        float(grad_clip_norm),
    )

    for offset in range(int(epochs)):
        epoch = int(start_epoch) + offset
        epoch_t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_true: List[int] = []
        train_pred: List[int] = []
        running_total = 0
        running_correct = 0

        progress = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
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

            if use_amp_runtime:
                scaler.scale(loss).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

            train_loss_sum += float(loss.item()) * images.size(0)
            pred = torch.argmax(logits, dim=1)
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(pred.detach().cpu().tolist())
            running_total += labels.size(0)
            running_correct += int((pred == labels).sum().item())
            running_acc = running_correct / max(running_total, 1)
            progress.set_postfix({"loss": f"{float(loss.item()):.4f}", "acc": f"{running_acc:.4f}"})

        train_loss = train_loss_sum / max(len(train_loader.dataset), 1)
        train_acc = accuracy_score(train_true, train_pred) if train_true else 0.0
        train_f1 = f1_score(train_true, train_pred, average="macro") if train_true else 0.0
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        elapsed = time.perf_counter() - epoch_t0

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
            "epoch_sec": float(elapsed),
        }
        history.append(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
            torch.save(checkpoint, best_ckpt_path)
            LOGGER.info("保存最佳模型: %s (epoch=%s, %s=%.6f)", best_ckpt_path, epoch, early_stop_metric, score)
        else:
            wait += 1

        if scheduler is not None:
            if scheduler_mode == "reduce":
                scheduler.step(float(score))
            else:
                scheduler.step()

        LOGGER.info(
            "%sEpoch %s 结果: 训练 Loss=%.6f Acc=%.4f F1=%.4f | 验证 Loss=%.6f Acc=%.4f F1=%.4f | best_%s=%.6f wait=%s/%s 耗时=%.1fs",
            stage_prefix,
            epoch,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
            early_stop_metric,
            best_score if best_score is not None else float("nan"),
            wait,
            patience,
            elapsed,
        )
        LOGGER.info("%s当前学习率: %.8f", stage_prefix, float(optimizer.param_groups[0].get("lr", 0.0)))

        if wait >= int(patience):
            LOGGER.info("%s触发早停: epoch=%s (best_epoch=%s)", stage_prefix, epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_artifacts:
        save_history_csv(history, run_paths.metrics_dir / "epoch_metrics.csv")
        LOGGER.info("已保存训练曲线数据: %s", run_paths.metrics_dir / "epoch_metrics.csv")
        plot_history(history, run_paths.figures_dir)
        LOGGER.info("已保存训练曲线图: %s", run_paths.figures_dir)
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
    train_loss = [float(r["train_loss"]) for r in history]
    val_loss = [float(r["val_loss"]) for r in history]
    train_acc = [float(r.get("train_acc", 0.0)) for r in history]
    val_acc = [float(r["val_acc"]) for r in history]
    train_f1 = [float(r.get("train_f1", 0.0)) for r in history]
    val_f1 = [float(r["val_f1"]) for r in history]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, train_loss, marker="o", label="Train Loss")
    plt.plot(xs, val_loss, marker="o", label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    (out_dir / "loss_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, train_acc, marker="o", label="Train Acc")
    plt.plot(xs, val_acc, marker="o", label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, train_f1, marker="o", label="Train F1")
    plt.plot(xs, val_f1, marker="o", label="Val F1")
    plt.title("Macro-F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_curve.png", dpi=160)
    plt.close()


def save_reports(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    reports_dir: Path,
    prefix: str = "",
) -> Dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(class_names)))

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    txt_name = f"{prefix}classification_report.txt" if prefix else "classification_report.txt"
    txt_path = reports_dir / txt_name
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(report_txt)
    LOGGER.info("已保存分类报告: %s", txt_path)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    fig_name = f"{prefix}confusion_matrix.png" if prefix else "confusion_matrix.png"
    fig_path = reports_dir / fig_name
    save_confusion_matrix(cm, class_names, fig_path)
    LOGGER.info("已保存混淆矩阵: %s", fig_path)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report_file": txt_name,
        "confusion_image": fig_name,
        "report_text": report_txt,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def save_markdown_report(
    *,
    reports_dir: Path,
    file_name: str,
    title: str,
    test_accuracy: float,
    macro_f1: float,
    report_text: str,
    confusion_image: str,
    confusion_matrix_text: str = "",
    metrics_curve_image: str = "",
) -> Path:
    out = reports_dir / file_name
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Test Accuracy:** {test_accuracy:.4f}\n\n")
        f.write(f"**Macro F1:** {macro_f1:.4f}\n\n")
        f.write("**分类报告:**\n\n")
        f.write("```\n")
        f.write(report_text.strip() + "\n")
        f.write("```\n\n")
        if confusion_matrix_text:
            f.write("**混淆矩阵:**\n\n")
            f.write("```\n")
            f.write(confusion_matrix_text.strip() + "\n")
            f.write("```\n\n")
        if confusion_image:
            f.write(f"![Confusion Matrix]({confusion_image})\n")
        if metrics_curve_image:
            f.write(f"![Metrics Curve]({metrics_curve_image})\n")
    LOGGER.info("已保存 Markdown 报告: %s", out)
    return out


def collect_attention_diagnostics(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    out_path: Path,
    max_batches: int = 8,
) -> Optional[Dict[str, float]]:
    model.eval()
    attn_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (images, tokens, _) in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            images = images.to(device, non_blocking=(device.type == "cuda"))
            tokens = tokens.to(device, non_blocking=(device.type == "cuda"))
            with _autocast(device, use_amp):
                try:
                    out = model(images, tokens, return_attention=True)
                except TypeError:
                    return None
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                return None
            attn = out[1]
            if attn is None:
                continue
            if not isinstance(attn, torch.Tensor):
                continue
            if attn.ndim == 1:
                attn = attn.unsqueeze(0)
            if attn.ndim != 2:
                continue
            attn_chunks.append(attn.detach().float().cpu())

    if not attn_chunks:
        return None

    attn_all = torch.cat(attn_chunks, dim=0)
    attn_mean = attn_all.mean(dim=0).numpy()
    eps = 1e-12
    row = attn_all.clamp(min=eps)
    entropy = float((-(row * row.log()).sum(dim=1)).mean().item())
    stats = {
        "mean": float(attn_all.mean().item()),
        "min": float(attn_all.min().item()),
        "max": float(attn_all.max().item()),
        "entropy": entropy,
        "top1": float(torch.topk(attn_all, k=1, dim=1).values.mean().item()),
        "top5": float(torch.topk(attn_all, k=min(5, attn_all.size(1)), dim=1).values.sum(dim=1).mean().item()),
        "top10": float(torch.topk(attn_all, k=min(10, attn_all.size(1)), dim=1).values.sum(dim=1).mean().item()),
    }

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(attn_mean)), attn_mean)
    plt.title("Mean Attention Curve")
    plt.xlabel("Token Index")
    plt.ylabel("Attention Weight")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return stats


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
