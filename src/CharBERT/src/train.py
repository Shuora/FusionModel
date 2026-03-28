import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from config import TrainingConfig
from data import create_dataloaders, SPECIAL_TOKENS
from eval import evaluate_predictions, save_confusion_matrix, save_metric_curve
from logging_utils import setup_logging, progress_bar
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_attention_mask(batch_inputs: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (batch_inputs != pad_id).long()


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device, logger) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = progress_bar(loader, desc="训练中")
    for batch_inputs, batch_labels in pbar:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        attn_mask = compute_attention_mask(batch_inputs, pad_id=SPECIAL_TOKENS["PAD"]).to(device)

        optimizer.zero_grad()
        logits = model(batch_inputs, attn_mask)
        loss = F.cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct = (preds == batch_labels).sum().item()
        total_correct += correct
        total += batch_labels.size(0)
        total_loss += loss.item() * batch_labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / batch_labels.size(0):.4f}"})

    epoch_loss = total_loss / total
    epoch_acc = total_correct / total
    logger.info(f"训练集: 平均损失 {epoch_loss:.4f}, 准确率 {epoch_acc:.4f}")
    return {"loss": epoch_loss, "acc": epoch_acc}


def validate(model: nn.Module, loader: DataLoader, device: torch.device, logger, id_to_label: Dict[int, str]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    pbar = progress_bar(loader, desc="验证中")
    with torch.no_grad():
        for batch_inputs, batch_labels in pbar:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            attn_mask = compute_attention_mask(batch_inputs, pad_id=SPECIAL_TOKENS["PAD"]).to(device)

            logits = model(batch_inputs, attn_mask)
            loss = F.cross_entropy(logits, batch_labels)

            preds = logits.argmax(dim=-1)
            correct = (preds == batch_labels).sum().item()
            total_correct += correct
            total += batch_labels.size(0)
            total_loss += loss.item() * batch_labels.size(0)

            all_true.extend(batch_labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / batch_labels.size(0):.4f}"})

    epoch_loss = total_loss / total
    epoch_acc = total_correct / total
    metrics = evaluate_predictions(all_true, all_pred, [id_to_label[i] for i in range(len(id_to_label))])
    report_text = classification_report(all_true, all_pred, target_names=[id_to_label[i] for i in range(len(id_to_label))], digits=4)

    logger.info(f"验证集: 平均损失 {epoch_loss:.4f}, 准确率 {epoch_acc:.4f}")
    logger.info("验证集分类报告:\n" + report_text)
    logger.info(
        "验证集宏指标: 精确率 {:.4f}, 召回率 {:.4f}, F1 {:.4f}, 准确率 {:.4f}".format(
            metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"], metrics["accuracy"]
        )
    )
    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "all_true": all_true,
        "all_pred": all_pred,
        "cm": metrics["cm"],
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    cfg = TrainingConfig().resolve_paths(base_dir)
    cfg.ensure_dirs()
    logger = setup_logging(Path(cfg.output_dir) / cfg.log_file)
    logger.info("开始训练 CharBERT 模型...")
    logger.info(f"使用设备: {cfg.device}")
    logger.info(f"训练数据目录: {cfg.train_dir}")
    logger.info(f"验证数据目录: {cfg.test_dir}")
    logger.info(f"输出目录: {cfg.output_dir}")
    logger.info(f"仅评估模式: {cfg.eval_only}")
    if cfg.checkpoint:
        logger.info(f"指定模型: {cfg.checkpoint}")

    # 处理 checkpoint 路径
    checkpoint_path = cfg.checkpoint if cfg.checkpoint else Path(cfg.output_dir) / "best_model.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path

    # 如果仅评估且已有 checkpoint，预取标签映射保持一致
    if cfg.eval_only and checkpoint_path.exists():
        ckpt_meta = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if "label_to_id" in ckpt_meta:
            label_to_id = ckpt_meta["label_to_id"]
            cfg.label_list = [lbl for lbl, idx in sorted(label_to_id.items(), key=lambda x: x[1])]

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, label_to_id, id_to_label = create_dataloaders(
        cfg.train_dir, cfg.test_dir, cfg.max_len, cfg.batch_size, cfg.num_workers, cfg.label_list
    )

    model = build_model(cfg, num_labels=len(label_to_id)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 仅评估模式：跳过训练，直接加载 checkpoint 并评估
    if cfg.eval_only:
        if not checkpoint_path.exists():
            logger.error(f"未找到指定的模型文件: {checkpoint_path}")
            return
        logger.info(f"仅评估模式，加载模型: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        if "label_to_id" in ckpt:
            label_to_id = ckpt["label_to_id"]
            id_to_label = {v: k for k, v in label_to_id.items()}
        final_metrics = validate(model, test_loader, device, logger, id_to_label)
        final_cm_path = Path(cfg.output_dir) / "cm_final.png"
        save_confusion_matrix(final_metrics["cm"], [id_to_label[i] for i in range(len(id_to_label))], final_cm_path, "最终模型混淆矩阵")
        logger.info(
            "最终准确率 {:.4f}, 宏精确率 {:.4f}, 宏召回率 {:.4f}, 宏F1 {:.4f}".format(
                final_metrics["acc"], final_metrics["precision_macro"], final_metrics["recall_macro"], final_metrics["f1_macro"]
            )
        )
        return

    history: Dict[str, List[float]] = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "val_f1_macro": []}

    best_f1 = -1.0
    best_state_path: Path = Path(cfg.output_dir) / "best_model.pt"
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        logger.info(f"==== Epoch {epoch}/{cfg.epochs} ====")
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, logger)
        val_metrics = validate(model, test_loader, device, logger, id_to_label)

        history["loss"].append(train_metrics["loss"])
        history["acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        # 保存最好模型
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {"model_state": model.state_dict(), "label_to_id": label_to_id, "cfg": asdict(cfg)}
            torch.save(best_state, best_state_path)
            logger.info(f"新最佳模型已保存: {best_state_path}, F1_macro={best_f1:.4f}, Accuracy={val_metrics['acc']:.4f}")

        # 混淆矩阵
        cm_path = Path(cfg.output_dir) / f"cm_epoch{epoch}.png"
        save_confusion_matrix(val_metrics["cm"], [id_to_label[i] for i in range(len(id_to_label))], cm_path, f"Epoch {epoch} 混淆矩阵")

    # 训练完成后保存曲线
    for metric in ["loss", "acc", "val_loss", "val_acc", "val_f1_macro"]:
        save_metric_curve(history, Path(cfg.output_dir), metric)

    # 最终报告：使用最佳模型在验证集上再次评估并输出详细报告与混淆矩阵
    if best_state_path.exists():
        logger.info("加载最佳模型进行最终评估...")
        checkpoint = torch.load(best_state_path, map_location=device, weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        if "label_to_id" in checkpoint:
            label_to_id = checkpoint["label_to_id"]
            id_to_label = {v: k for k, v in label_to_id.items()}
        final_metrics = validate(model, test_loader, device, logger, id_to_label)
        final_cm_path = Path(cfg.output_dir) / "cm_final.png"
        save_confusion_matrix(final_metrics["cm"], [id_to_label[i] for i in range(len(id_to_label))], final_cm_path, "最终模型混淆矩阵")
        logger.info(
            "最终准确率 {:.4f}, 宏精确率 {:.4f}, 宏召回率 {:.4f}, 宏F1 {:.4f}".format(
                final_metrics["acc"], final_metrics["precision_macro"], final_metrics["recall_macro"], final_metrics["f1_macro"]
            )
        )
    else:
        logger.warning("未找到最佳模型文件，跳过最终评估")

    logger.info("训练完成。最佳宏 F1: {:.4f}".format(best_f1))


if __name__ == "__main__":
    main()

