"""Train attention fusion model and fit XGBoost stacking classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from models_charbert_mobilevit import AttentionFusionModel
from train_common import (
    collect_attention_diagnostics,
    create_criterion,
    create_data_loaders,
    device_from_arg,
    evaluate,
    fit_model,
    init_run_dirs,
    save_markdown_report,
    save_final_metrics,
    save_reports,
    set_seed,
    setup_logging,
)

LOGGER = logging.getLogger("train_attention_stacking")

DEFAULT_ARGS = {
    "batch_size": 32,
    "epochs": 32,
    "lr_scheduler": "none",
    "grad_clip_norm": 0.0,
    "label_smoothing": 0.0,
    "class_balance": "none",
    "stage1_epochs": 0,
    "stage2_lr_scale": 0.25,
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attention fusion + XGBoost stacking")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_name", default="")
    p.add_argument(
        "--train_profile",
        choices=["auto", "none", "ustc_strict_base", "ustc_strict_sota"],
        default="auto",
        help="USTC profile presets for stacking pipeline.",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=28)
    p.add_argument("--char_seq_len", type=int, default=786)

    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--early_stop_metric", choices=["val_loss", "val_acc", "val_f1"], default="val_f1")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--lr_scheduler", choices=["none", "reduce", "cosine"], default="none")
    p.add_argument("--lr_patience", type=int, default=2)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)

    p.add_argument("--class_balance", choices=["none", "weighted_loss", "weighted_sampler", "weighted_sampler_loss"], default="none")
    p.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument("--attention_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--stage1_epochs", type=int, default=0, help="Freeze stage epochs; 0 disables two-stage training.")
    p.add_argument("--stage2_lr_scale", type=float, default=0.25, help="Learning-rate scale for stage2 unfreezing.")
    p.add_argument("--freeze_charbert_layers", type=int, default=1, help="Number of front transformer layers to freeze in stage1.")

    p.add_argument("--use_pretrained_mobilevit", action="store_true")
    p.add_argument("--mobilevit_pretrained_dir", default="model/mobilevit-small")
    p.add_argument("--mobilevit_pretrained_input_size", type=int, default=224)
    p.add_argument("--use_pretrained_charbert", action="store_true")
    p.add_argument("--charbert_pretrained_path", default="")

    p.add_argument("--xgb_estimators", type=int, default=300)
    p.add_argument("--xgb_max_depth", type=int, default=6)
    p.add_argument("--xgb_lr", type=float, default=0.05)
    p.add_argument("--stacking_subsample", type=float, default=1.0, help="Subsample ratio for meta-learner training.")

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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    mode: str,
    early_stop_metric: str,
    lr_factor: float,
    lr_patience: int,
    min_lr: float,
    epochs: int,
) -> Optional[object]:
    if mode == "reduce":
        reduce_mode = "min" if early_stop_metric == "val_loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=reduce_mode,
            factor=float(lr_factor),
            patience=int(lr_patience),
            min_lr=float(min_lr),
        )
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(min_lr),
        )
    return None


def _set_if_default(args: argparse.Namespace, key: str, value) -> None:
    if key not in DEFAULT_ARGS:
        return
    if getattr(args, key) == DEFAULT_ARGS[key]:
        setattr(args, key, value)


def apply_profile(args: argparse.Namespace, resolved_dataset: str) -> None:
    profile = str(args.train_profile)
    if profile == "auto":
        profile = "ustc_strict_base" if "USTC" in str(resolved_dataset).upper() else "none"
    if profile == "none":
        return
    if "USTC" not in str(resolved_dataset).upper():
        LOGGER.info("train_profile=%s 仅建议用于 USTC，当前数据集=%s", profile, resolved_dataset)

    if profile == "ustc_strict_base":
        _set_if_default(args, "batch_size", 64)
        _set_if_default(args, "lr_scheduler", "reduce")
        _set_if_default(args, "grad_clip_norm", 1.0)
        _set_if_default(args, "label_smoothing", 0.03)
        _set_if_default(args, "class_balance", "weighted_sampler_loss")
        _set_if_default(args, "stage1_epochs", 10)
    elif profile == "ustc_strict_sota":
        _set_if_default(args, "batch_size", 64)
        _set_if_default(args, "epochs", 40)
        _set_if_default(args, "lr_scheduler", "reduce")
        _set_if_default(args, "grad_clip_norm", 1.0)
        _set_if_default(args, "label_smoothing", 0.02)
        _set_if_default(args, "class_balance", "weighted_sampler_loss")
        _set_if_default(args, "stage1_epochs", 12)

    model_dir = Path(str(args.mobilevit_pretrained_dir)).resolve()
    if model_dir.exists() and model_dir.is_dir() and (not args.use_pretrained_mobilevit):
        args.use_pretrained_mobilevit = True
        LOGGER.info("train_profile=%s 自动启用预训练 MobileViT: %s", profile, model_dir)

    args.train_profile = profile


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(flag)


def apply_stage1_freeze(model: AttentionFusionModel, freeze_charbert_layers: int) -> int:
    _set_requires_grad(model.image_encoder, False)
    _set_requires_grad(model.text_encoder, False)

    if hasattr(model.image_encoder, "head"):
        _set_requires_grad(getattr(model.image_encoder, "head"), True)
    if hasattr(model.image_encoder, "mvit2"):
        _set_requires_grad(getattr(model.image_encoder, "mvit2"), True)
    if hasattr(model.image_encoder, "ir2"):
        _set_requires_grad(getattr(model.image_encoder, "ir2"), True)
    if hasattr(model.image_encoder, "proj"):
        _set_requires_grad(getattr(model.image_encoder, "proj"), True)

    layers = getattr(getattr(model.text_encoder, "encoder", None), "layers", None)
    if isinstance(layers, torch.nn.ModuleList):
        freeze_n = max(int(freeze_charbert_layers), 0)
        for idx, layer in enumerate(layers):
            _set_requires_grad(layer, idx >= freeze_n)
    if hasattr(model.text_encoder, "proj"):
        _set_requires_grad(getattr(model.text_encoder, "proj"), True)

    for m in (model.q_proj, model.k_proj, model.v_proj, model.fused_head, model.image_head, model.text_head):
        _set_requires_grad(m, True)
    return count_trainable_params(model)


def apply_stage2_unfreeze(model: AttentionFusionModel) -> int:
    _set_requires_grad(model, True)
    return count_trainable_params(model)


def build_optimizer(args: argparse.Namespace, model: AttentionFusionModel, lr: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = list(model.parameters())
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)


def collect_meta_features(model: AttentionFusionModel, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for images, tokens, y in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            tokens = tokens.to(device, non_blocking=(device.type == "cuda"))

            img_feat, txt_feat, ctx, _, fused_logits, _ = model.extract_features(images, tokens)
            pf = torch.softmax(fused_logits, dim=1)
            feat = torch.cat([img_feat, txt_feat, ctx, pf], dim=1)

            feats.append(feat.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    if feats:
        X = np.concatenate(feats, axis=0)
        Y = np.concatenate(labels, axis=0)
    else:
        X = np.empty((0, 0), dtype=np.float32)
        Y = np.empty((0,), dtype=np.int64)
    return X, Y


def main() -> None:
    args = build_argparser().parse_args()

    set_seed(int(args.seed))
    device = device_from_arg(args.device)
    apply_profile(args, args.dataset_root)

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
    LOGGER.info("训练参数: %s", json.dumps(vars(args), ensure_ascii=False))

    num_classes = len(classes)
    LOGGER.info("数据集=%s 类别数=%s 训练样本=%s 验证样本=%s", resolved_dataset, num_classes, len(train_loader.dataset), len(val_loader.dataset))

    model = AttentionFusionModel(
        num_classes=num_classes,
        attention_dim=args.attention_dim,
        char_seq_len=args.char_seq_len,
        dropout=args.dropout,
        use_pretrained_mobilevit=bool(args.use_pretrained_mobilevit),
        mobilevit_pretrained_dir=args.mobilevit_pretrained_dir,
        mobilevit_pretrained_input_size=int(args.mobilevit_pretrained_input_size),
        use_pretrained_charbert=bool(args.use_pretrained_charbert),
        charbert_pretrained_path=args.charbert_pretrained_path,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("模型参数量: total=%s trainable=%s", total_params, trainable_params)

    criterion = create_criterion(
        loss_type=args.loss_type,
        class_balance=args.class_balance,
        class_counts=getattr(train_loader.dataset, "class_counts", []),
        device=device,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    total_epochs = int(args.epochs)
    stage1_epochs = max(0, min(int(args.stage1_epochs), total_epochs))
    stage2_epochs = total_epochs - stage1_epochs
    stage_summaries = []

    history = []
    if stage1_epochs > 0:
        trainable_stage1 = apply_stage1_freeze(model, int(args.freeze_charbert_layers))
        LOGGER.info(
            "Stage1 冻结训练: epochs=%s 冻结CharBERT前层=%s trainable_params=%s",
            stage1_epochs,
            args.freeze_charbert_layers,
            trainable_stage1,
        )
        optimizer1 = build_optimizer(args, model, float(args.lr))
        scheduler1 = build_scheduler(
            optimizer1,
            mode=args.lr_scheduler,
            early_stop_metric=args.early_stop_metric,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr,
            epochs=stage1_epochs,
        )
        history, _ = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer1,
            device=device,
            epochs=stage1_epochs,
            patience=args.patience,
            early_stop_metric=args.early_stop_metric,
            use_amp=(not args.no_amp),
            run_paths=run_paths,
            scheduler=scheduler1,
            scheduler_mode=args.lr_scheduler,
            grad_clip_norm=args.grad_clip_norm,
            start_epoch=1,
            history=history,
            save_artifacts=(stage2_epochs <= 0),
            stage_name="stage1",
        )
        stage_summaries.append(
            {
                "stage": "stage1",
                "epochs": stage1_epochs,
                "trainable_params": trainable_stage1,
                "lr": float(args.lr),
            }
        )

    if stage2_epochs > 0:
        if stage1_epochs > 0:
            stage2_lr = float(args.lr) * float(args.stage2_lr_scale)
            trainable_stage2 = apply_stage2_unfreeze(model)
            stage_name = "stage2"
        else:
            stage2_lr = float(args.lr)
            trainable_stage2 = count_trainable_params(model)
            stage_name = "single_stage"
        LOGGER.info(
            "%s 训练: epochs=%s lr=%.6f trainable_params=%s",
            stage_name,
            stage2_epochs,
            stage2_lr,
            trainable_stage2,
        )
        optimizer2 = build_optimizer(args, model, stage2_lr)
        scheduler2 = build_scheduler(
            optimizer2,
            mode=args.lr_scheduler,
            early_stop_metric=args.early_stop_metric,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr,
            epochs=stage2_epochs,
        )
        history, best_ckpt = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer2,
            device=device,
            epochs=stage2_epochs,
            patience=args.patience,
            early_stop_metric=args.early_stop_metric,
            use_amp=(not args.no_amp),
            run_paths=run_paths,
            scheduler=scheduler2,
            scheduler_mode=args.lr_scheduler,
            grad_clip_norm=args.grad_clip_norm,
            start_epoch=len(history) + 1,
            history=history,
            save_artifacts=True,
            stage_name=stage_name,
        )
        stage_summaries.append(
            {
                "stage": stage_name,
                "epochs": stage2_epochs,
                "trainable_params": trainable_stage2,
                "lr": stage2_lr,
            }
        )
    else:
        best_ckpt = run_paths.checkpoints_dir / "best.pt"

    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    base_val_loss, base_val_acc, base_val_f1, base_true, base_pred = evaluate(
        model,
        val_loader,
        criterion,
        device,
        use_amp=(not args.no_amp),
    )

    base_report_metrics = save_reports(
        y_true=base_true,
        y_pred=base_pred,
        class_names=classes,
        reports_dir=run_paths.reports_dir,
    )
    base_report_text = str(base_report_metrics.get("report_text", "")).strip()
    base_cm_arr = np.asarray(base_report_metrics.get("confusion_matrix", []), dtype=np.int64)
    base_cm_text = np.array2string(base_cm_arr, separator=" ")
    if base_report_text:
        LOGGER.info("[base] 分类报告:\n%s", base_report_text)
    if base_cm_arr.size > 0:
        LOGGER.info("[base] 混淆矩阵:\n%s", base_cm_text)

    save_markdown_report(
        reports_dir=run_paths.reports_dir,
        file_name="report_attention_base.md",
        title="融合方式: attention",
        test_accuracy=float(base_val_acc),
        macro_f1=float(base_val_f1),
        report_text=base_report_text,
        confusion_image=str(base_report_metrics.get("confusion_image", "")),
        confusion_matrix_text=base_cm_text,
        metrics_curve_image="../figures/f1_curve.png",
    )

    X_train, y_train = collect_meta_features(model, train_loader, device)
    X_test, y_test = collect_meta_features(model, val_loader, device)
    LOGGER.info("元特征维度: train=%s test=%s", X_train.shape, X_test.shape)

    sub_ratio = float(min(max(args.stacking_subsample, 0.01), 1.0))
    if sub_ratio < 1.0 and len(y_train) > 0:
        rng = np.random.default_rng(int(args.seed))
        take = max(1, int(len(y_train) * sub_ratio))
        idx = rng.choice(len(y_train), size=take, replace=False)
        X_train_meta = X_train[idx]
        y_train_meta = y_train[idx]
        LOGGER.info("stacking_subsample=%.3f, 元学习器训练样本: %s -> %s", sub_ratio, len(y_train), len(y_train_meta))
    else:
        X_train_meta = X_train
        y_train_meta = y_train

    np.save(run_paths.stacking_dir / "meta_features_train.npy", X_train)
    np.save(run_paths.stacking_dir / "meta_features_test.npy", X_test)
    LOGGER.info("已保存元特征: %s", run_paths.stacking_dir / "meta_features_train.npy")
    LOGGER.info("已保存元特征: %s", run_paths.stacking_dir / "meta_features_test.npy")

    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("xgboost is required for stacking. Install with: pip install xgboost") from exc

    if num_classes <= 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    clf = XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=0.9,
        colsample_bytree=0.9,
        objective=objective,
        num_class=(num_classes if num_classes > 2 else None),
        eval_metric=eval_metric,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=0,
    )
    LOGGER.info(
        "XGBoost 参数: n_estimators=%s max_depth=%s learning_rate=%s objective=%s eval_metric=%s",
        args.xgb_estimators,
        args.xgb_max_depth,
        args.xgb_lr,
        objective,
        eval_metric,
    )
    clf.fit(X_train_meta, y_train_meta)

    if num_classes <= 2:
        prob = clf.predict_proba(X_test)
        pred = (prob[:, 1] >= 0.5).astype(np.int64)
    else:
        pred = clf.predict(X_test)

    pred = np.asarray(pred, dtype=np.int64)

    stacking_report_metrics = save_reports(
        y_true=y_test.tolist(),
        y_pred=pred.tolist(),
        class_names=classes,
        reports_dir=run_paths.reports_dir,
        prefix="stacking_",
    )
    label_ids = list(range(len(classes)))
    stacking_report_text = classification_report(
        y_test,
        pred,
        labels=label_ids,
        target_names=list(classes),
        digits=4,
        zero_division=0,
    )
    stacking_cm = confusion_matrix(y_test, pred, labels=label_ids)
    stacking_cm_text = np.array2string(stacking_cm, separator=" ")
    LOGGER.info("[stacking] 分类报告:\n%s", stacking_report_text)
    LOGGER.info("[stacking] 混淆矩阵:\n%s", stacking_cm_text)

    save_markdown_report(
        reports_dir=run_paths.reports_dir,
        file_name="report_attention_stacking_xgboost.md",
        title="融合方式: attention+stacking (xgboost)",
        test_accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        report_text=stacking_report_text,
        confusion_image=str(stacking_report_metrics.get("confusion_image", "")),
        confusion_matrix_text=stacking_cm_text,
        metrics_curve_image="../figures/f1_curve.png",
    )

    model_path = run_paths.stacking_dir / "meta_model_xgboost.json"
    clf.save_model(str(model_path))
    LOGGER.info("已保存 stacking 模型: %s", model_path)

    attention_curve = run_paths.figures_dir / "attention_curve.png"
    attn_stats = collect_attention_diagnostics(
        model=model,
        loader=val_loader,
        device=device,
        use_amp=(not args.no_amp),
        out_path=attention_curve,
    )
    if attn_stats:
        LOGGER.info(
            "[AttentionDiag] mean=%.6f min=%.6f max=%.6f entropy=%.6f top1=%.4f top5=%.4f top10=%.4f",
            float(attn_stats["mean"]),
            float(attn_stats["min"]),
            float(attn_stats["max"]),
            float(attn_stats["entropy"]),
            float(attn_stats["top1"]),
            float(attn_stats["top5"]),
            float(attn_stats["top10"]),
        )
        LOGGER.info("已保存注意力曲线: %s", attention_curve)

    final_metrics = {
        "run_id": run_paths.run_id,
        "dataset": resolved_dataset,
        "num_classes": num_classes,
        "train_profile": args.train_profile,
        "stage_summaries": stage_summaries,
        "image_encoder_type": getattr(model, "image_encoder_type", "native"),
        "base": {
            "val_loss": float(base_val_loss),
            "val_acc": float(base_val_acc),
            "val_f1": float(base_val_f1),
            "report_metrics": base_report_metrics,
        },
        "stacking": {
            "val_acc": float(accuracy_score(y_test, pred)),
            "val_f1": float(f1_score(y_test, pred, average="macro")),
            "report_metrics": stacking_report_metrics,
            "model_path": str(model_path),
        },
        "epochs_recorded": len(history),
        "class_names": classes,
        "config": vars(args),
    }

    save_final_metrics(run_paths.metrics_dir / "final_metrics.json", final_metrics)
    LOGGER.info("已保存最终指标: %s", run_paths.metrics_dir / "final_metrics.json")

    LOGGER.info(
        "Stacking 完成. base_f1=%.4f stacking_f1=%.4f 输出目录=%s",
        float(base_val_f1),
        final_metrics["stacking"]["val_f1"],
        run_paths.root,
    )


if __name__ == "__main__":
    main()
