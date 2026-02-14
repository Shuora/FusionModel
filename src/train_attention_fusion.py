"""Train native CharBERT + MobileViT attention fusion model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from models_charbert_mobilevit import AttentionFusionModel
from train_common import (
    create_criterion,
    create_data_loaders,
    device_from_arg,
    evaluate,
    fit_model,
    init_run_dirs,
    save_final_metrics,
    save_reports,
    set_seed,
    setup_logging,
)

LOGGER = logging.getLogger("train_attention_fusion")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train attention fusion model (CharBERT + native MobileViT)")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_name", default="")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=28)
    p.add_argument("--char_seq_len", type=int, default=786)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--early_stop_metric", choices=["val_loss", "val_acc", "val_f1"], default="val_f1")

    p.add_argument("--class_balance", choices=["none", "weighted_loss", "weighted_sampler", "weighted_sampler_loss"], default="none")
    p.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument("--attention_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--ablation", choices=["none", "r_only"], default="none")
    p.add_argument("--no_index_cache", action="store_true")
    p.add_argument("--rebuild_index_cache", action="store_true")

    p.add_argument("--output_dir", default="outputs/train")
    p.add_argument("--run_name", default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    set_seed(int(args.seed))
    device = device_from_arg(args.device)

    run_paths = init_run_dirs(args.output_dir, args.run_name)
    setup_logging(run_paths.logs_dir / "train.log")

    LOGGER.info("运行 ID: %s", run_paths.run_id)
    LOGGER.info("训练设备: %s", device)

    train_loader, val_loader, classes, resolved_dataset = create_data_loaders(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        image_size=args.image_size,
        char_seq_len=args.char_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory or device.type == "cuda"),
        class_balance=args.class_balance,
        use_index_cache=(not args.no_index_cache),
        rebuild_index_cache=args.rebuild_index_cache,
        r_only=(args.ablation == "r_only"),
    )

    num_classes = len(classes)
    LOGGER.info("数据集=%s 类别数=%s 训练样本=%s 验证样本=%s", resolved_dataset, num_classes, len(train_loader.dataset), len(val_loader.dataset))

    model = AttentionFusionModel(
        num_classes=num_classes,
        attention_dim=args.attention_dim,
        char_seq_len=args.char_seq_len,
        dropout=args.dropout,
    ).to(device)

    criterion = create_criterion(
        loss_type=args.loss_type,
        class_balance=args.class_balance,
        class_counts=getattr(train_loader.dataset, "class_counts", []),
        device=device,
        focal_gamma=args.focal_gamma,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history, best_ckpt = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        use_amp=(not args.no_amp),
        run_paths=run_paths,
    )

    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, val_f1, y_true, y_pred = evaluate(
        model,
        val_loader,
        criterion,
        device,
        use_amp=(not args.no_amp),
    )

    report_metrics = save_reports(
        y_true=y_true,
        y_pred=y_pred,
        class_names=classes,
        reports_dir=run_paths.reports_dir,
    )

    final_metrics = {
        "run_id": run_paths.run_id,
        "dataset": resolved_dataset,
        "num_classes": num_classes,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_f1": float(val_f1),
        "report_metrics": report_metrics,
        "epochs_recorded": len(history),
        "class_names": classes,
        "config": vars(args),
    }

    save_final_metrics(run_paths.metrics_dir / "final_metrics.json", final_metrics)

    LOGGER.info("训练完成，最佳模型: %s", best_ckpt)
    LOGGER.info("最终指标 val_loss=%.6f val_acc=%.4f val_f1=%.4f", val_loss, val_acc, val_f1)
    LOGGER.info("输出目录: %s", run_paths.root)


if __name__ == "__main__":
    main()
