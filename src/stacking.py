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

from src.models.fusion_model import MobileViTETBertFusionClassifier
from src.pipeline_data import load_policy_multimodal_data

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def _compute_meta_features(out: dict) -> np.ndarray:
    logits_img = out["logits_img"]
    logits_tls = out["logits_tls"]
    logits_fuse = out["logits_fuse"]

    p_img = torch.softmax(logits_img, dim=1)
    p_tls = torch.softmax(logits_tls, dim=1)
    p_fuse = torch.softmax(logits_fuse, dim=1)

    def entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1, keepdim=True)

    def margin(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
        if top2.shape[1] < 2:
            return torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device)
        return (top2[:, 0] - top2[:, 1]).unsqueeze(1)

    feats = torch.cat(
        [
            logits_img,
            logits_tls,
            logits_fuse,
            entropy(p_img),
            entropy(p_tls),
            entropy(p_fuse),
            margin(logits_img),
            margin(logits_tls),
            margin(logits_fuse),
            out["gate"],
        ],
        dim=1,
    )
    return feats.detach().cpu().numpy().astype(np.float32, copy=False)


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
    seed: int,
) -> MobileViTETBertFusionClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MobileViTETBertFusionClassifier(
        num_classes=num_classes,
        hidden_dim=128,
        vocab_size=vocab_size,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
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
    )

    for _ in range(max(1, epochs)):
        model.train()
        for rgb_b, input_b, att_b, type_b, y_b in loader:
            optim.zero_grad()
            out = model(rgb_b, input_b, att_b, type_b)
            loss = ce(out["logits_fuse"], y_b) + 0.3 * ce(out["logits_img"], y_b) + 0.3 * ce(
                out["logits_tls"], y_b
            )
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
) -> np.ndarray:
    if rgb.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(rgb).float(),
        torch.from_numpy(input_ids).long(),
        torch.from_numpy(attention).long(),
        torch.from_numpy(token_type_ids).long(),
    )
    loader = DataLoader(ds, batch_size=max(1, batch_size), shuffle=False)
    feats = []
    with torch.no_grad():
        for rgb_b, input_b, att_b, type_b in loader:
            out = model(rgb_b, input_b, att_b, type_b)
            feats.append(_compute_meta_features(out))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
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

    # OOF meta features
    n_splits = min(args.n_splits, int(np.bincount(y_tv).min()) if len(y_tv) > 0 else 2)
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    oof_x = np.zeros((len(y_tv), 3 * num_classes + 7), dtype=np.float32)
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
            seed=args.seed + fold_id,
        )
        oof_x[va_idx] = _predict_meta(
            model,
            rgb_tv[va_idx],
            input_tv[va_idx],
            att_tv[va_idx],
            type_tv[va_idx],
            batch_size=args.batch_size,
        )

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
        seed=args.seed + 999,
    )
    test_x = _predict_meta(
        final_model,
        rgb_test,
        input_test,
        att_test,
        type_test,
        batch_size=args.batch_size,
    )

    meta_model = _fit_meta_learner(oof_x, y_tv, num_classes=num_classes)
    pred = meta_model.predict(test_x)
    top1 = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, pred, average="macro", zero_division=0))

    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stack_dir / "oof_meta_train.npz", X=oof_x, y=y_tv)
    np.savez_compressed(stack_dir / "meta_test.npz", X=test_x, y=y_test)
    metrics = {
        "top1": top1,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "n_train_samples": int(len(y_tv)),
        "n_test_samples": int(len(y_test)),
        "n_splits": int(n_splits),
    }
    (stack_dir / "meta_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta_model.save_model(str(stack_dir / "meta_model.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
