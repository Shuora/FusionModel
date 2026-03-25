from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.meta_features import STAGE2_META_SCHEMA_VERSION, flatten_meta_feature_blocks
from src.models.fusion_model import MobileViTETBertFusionClassifier
from src.pipeline_data import load_policy_multimodal_data
from src.runtime_device import resolve_runtime_device

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def _empty_meta_schema() -> dict:
    return {
        "version": STAGE2_META_SCHEMA_VERSION,
        "dim": 0,
        "feature_names": [],
    }


def _train_base_model(
    rgb: np.ndarray,
    input_ids: np.ndarray,
    attention: np.ndarray,
    token_type_ids: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    vocab_size: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    hidden_dim: int,
    fusion_layers: int,
    fusion_heads: int,
    fusion_dropout: float,
    lr: float,
    alpha: float,
    beta: float,
) -> MobileViTETBertFusionClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MobileViTETBertFusionClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        fusion_layers=fusion_layers,
        fusion_heads=fusion_heads,
        dropout=fusion_dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(rgb).float(),
            torch.from_numpy(input_ids).long(),
            torch.from_numpy(attention).long(),
            torch.from_numpy(token_type_ids).long(),
            torch.from_numpy(y).long(),
        ),
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    for _ in range(max(1, epochs)):
        model.train()
        for rgb_b, input_b, att_b, type_b, y_b in loader:
            rgb_b = rgb_b.to(device)
            input_b = input_b.to(device)
            att_b = att_b.to(device)
            type_b = type_b.to(device)
            y_b = y_b.to(device)
            optim.zero_grad()
            out = model(rgb_b, input_b, att_b, type_b)
            loss = ce(out["logits_fuse"], y_b) + alpha * ce(out["logits_img"], y_b) + beta * ce(out["logits_tls"], y_b)
            loss.backward()
            optim.step()
    return model


def _predict_meta(
    model: MobileViTETBertFusionClassifier,
    rgb: np.ndarray,
    input_ids: np.ndarray,
    attention: np.ndarray,
    token_type_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, list[str], dict]:
    if rgb.shape[0] == 0:
        schema = _empty_meta_schema()
        return np.zeros((0, 0), dtype=np.float32), [], schema
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(rgb).float(),
        torch.from_numpy(input_ids).long(),
        torch.from_numpy(attention).long(),
        torch.from_numpy(token_type_ids).long(),
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feats: list[np.ndarray] = []
    feature_names: list[str] | None = None
    feature_schema: dict | None = None
    with torch.no_grad():
        for rgb_b, input_b, att_b, type_b in loader:
            rgb_b = rgb_b.to(device)
            input_b = input_b.to(device)
            att_b = att_b.to(device)
            type_b = type_b.to(device)
            out = model(rgb_b, input_b, att_b, type_b, return_summary=True)
            x_b, names_b, schema_b = flatten_meta_feature_blocks(out)
            if feature_names is None:
                feature_names = names_b
                feature_schema = schema_b
            elif names_b != feature_names:
                raise ValueError("inconsistent meta feature schema across batches")
            feats.append(x_b)
    if not feats:
        schema = _empty_meta_schema()
        return np.zeros((0, 0), dtype=np.float32), [], schema
    return np.concatenate(feats, axis=0), feature_names or [], feature_schema or _empty_meta_schema()


def _fit_meta_learner(x_train: np.ndarray, y_train: np.ndarray, num_classes: int):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for stacking but not available")
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    clf = XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective=objective,
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(x_train, y_train)
    return clf


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train stacking meta-learner with OOF features")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--oof-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    _, resolved_device, _ = resolve_runtime_device(args.device)
    device = torch.device(resolved_device)
    data = load_policy_multimodal_data(
        cfg["processed_root"],
        cfg["policy"],
        datasets=cfg.get("datasets") or None,
        label_mode=str(cfg.get("label_mode", "multiclass")),
        session_filter_manifest=cfg.get("session_filter_manifest"),
    )

    rgb = data["rgb"]
    input_ids = data["input_ids"]
    attention = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    y = data["y"]
    split = data["split"]
    if rgb.shape[0] == 0:
        return 2

    train_mask = np.isin(split, ["train", "val"])
    test_mask = split == "test"
    if not np.any(test_mask):
        test_mask = split == "val"
    if not np.any(test_mask):
        test_mask = np.ones_like(train_mask, dtype=bool)

    rgb_tv = rgb[train_mask]
    input_tv = input_ids[train_mask]
    att_tv = attention[train_mask]
    type_tv = token_type_ids[train_mask]
    y_tv = y[train_mask]

    rgb_test = rgb[test_mask]
    input_test = input_ids[test_mask]
    att_test = attention[test_mask]
    type_test = token_type_ids[test_mask]
    y_test = y[test_mask]

    num_classes = int(np.max(y)) + 1
    vocab_size = int(max(30522, int(input_ids.max()) + 1))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    fusion_layers = int(cfg.get("fusion_layers", 2))
    fusion_heads = int(cfg.get("fusion_heads", cfg.get("num_heads", 4)))
    fusion_dropout = float(cfg.get("fusion_dropout", 0.1))
    lr = float(cfg.get("lr", 1e-3))
    alpha = float(cfg.get("alpha", 0.3))
    beta = float(cfg.get("beta", 0.3))

    # OOF meta features
    n_splits = min(args.n_splits, int(np.bincount(y_tv).min()) if len(y_tv) > 0 else 2)
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    oof_x: np.ndarray | None = None
    meta_feature_names: list[str] | None = None
    meta_feature_schema: dict | None = None
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(rgb_tv, y_tv)):
        model = _train_base_model(
            rgb_tv[tr_idx],
            input_tv[tr_idx],
            att_tv[tr_idx],
            type_tv[tr_idx],
            y_tv[tr_idx],
            num_classes=num_classes,
            vocab_size=vocab_size,
            epochs=args.oof_epochs,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            seed=args.seed + fold_id,
            hidden_dim=hidden_dim,
            fusion_layers=fusion_layers,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            lr=lr,
            alpha=alpha,
            beta=beta,
        )
        fold_x, fold_names, fold_schema = _predict_meta(
            model,
            rgb_tv[va_idx],
            input_tv[va_idx],
            att_tv[va_idx],
            type_tv[va_idx],
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        if oof_x is None:
            oof_x = np.zeros((len(y_tv), fold_x.shape[1]), dtype=np.float32)
            meta_feature_names = list(fold_names)
            meta_feature_schema = dict(fold_schema)
        elif fold_names != meta_feature_names:
            raise ValueError("inconsistent meta feature schema across folds")
        oof_x[va_idx] = fold_x

    # Final model for test meta features
    final_model = _train_base_model(
        rgb_tv,
        input_tv,
        att_tv,
        type_tv,
        y_tv,
        num_classes=num_classes,
        vocab_size=vocab_size,
        epochs=max(1, args.oof_epochs),
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed + 999,
        hidden_dim=hidden_dim,
        fusion_layers=fusion_layers,
        fusion_heads=fusion_heads,
        fusion_dropout=fusion_dropout,
        lr=lr,
        alpha=alpha,
        beta=beta,
    )
    test_x, test_feature_names, test_feature_schema = _predict_meta(
        final_model,
        rgb_test,
        input_test,
        att_test,
        type_test,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    if oof_x is None:
        oof_x = np.zeros((len(y_tv), test_x.shape[1]), dtype=np.float32)
        meta_feature_names = list(test_feature_names)
        meta_feature_schema = dict(test_feature_schema)
    if test_feature_names != (meta_feature_names or []):
        raise ValueError("meta feature schema mismatch between OOF and test")

    meta_model = _fit_meta_learner(oof_x, y_tv, num_classes=num_classes)
    pred = meta_model.predict(test_x)
    top1 = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, pred, average="macro", zero_division=0))

    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    schema_version = (meta_feature_schema or {}).get("version", STAGE2_META_SCHEMA_VERSION)
    schema_json = json.dumps(meta_feature_schema or _empty_meta_schema(), ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        stack_dir / "oof_meta_train.npz",
        X=oof_x,
        y=y_tv,
        feature_names=np.asarray(meta_feature_names or [], dtype=np.str_),
        feature_schema=np.array(schema_json, dtype=np.str_),
        feature_schema_version=np.array(schema_version, dtype=np.str_),
    )
    np.savez_compressed(
        stack_dir / "meta_test.npz",
        X=test_x,
        y=y_test,
        feature_names=np.asarray(test_feature_names, dtype=np.str_),
        feature_schema=np.array(json.dumps(test_feature_schema, ensure_ascii=False, sort_keys=True), dtype=np.str_),
        feature_schema_version=np.array(test_feature_schema.get("version", STAGE2_META_SCHEMA_VERSION), dtype=np.str_),
    )
    metrics = {
        "top1": top1,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "n_train_samples": int(len(y_tv)),
        "n_test_samples": int(len(y_test)),
        "n_splits": int(n_splits),
        "meta_schema_version": schema_version,
        "meta_feature_dim": int(oof_x.shape[1]),
        "meta_feature_names": list(meta_feature_names or []),
    }
    (stack_dir / "meta_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta_model.save_model(str(stack_dir / "meta_model.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
