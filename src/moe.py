from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.models.fusion_model import TinyFusionClassifier
from src.pipeline_data import load_policy_multimodal_data


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 16, num_experts: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _expert_probs(out: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_img = torch.softmax(out["logits_img"], dim=1)
    p_tls = torch.softmax(out["logits_tls"], dim=1)
    p_fuse = torch.softmax(out["logits_fuse"], dim=1)
    return p_img, p_tls, p_fuse


def _router_features(out: dict) -> torch.Tensor:
    p_img, p_tls, p_fuse = _expert_probs(out)

    def entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1, keepdim=True)

    f = torch.cat(
        [
            entropy(p_img),
            entropy(p_tls),
            entropy(p_fuse),
            out["gate"],
            p_img.max(dim=1, keepdim=True).values,
            p_tls.max(dim=1, keepdim=True).values,
            p_fuse.max(dim=1, keepdim=True).values,
        ],
        dim=1,
    )
    return f


def _mixture_probs(out: dict, router_logits: torch.Tensor) -> torch.Tensor:
    p_img, p_tls, p_fuse = _expert_probs(out)
    weights = torch.softmax(router_logits, dim=1)
    mix = (
        weights[:, 0:1] * p_img
        + weights[:, 1:2] * p_tls
        + weights[:, 2:3] * p_fuse
    )
    return mix


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train MoE router on top of fusion model")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    data = load_policy_multimodal_data(cfg["processed_root"], cfg["policy"])

    rgb = data["rgb"]
    token_ids = data["token_ids"]
    attention = data["attention_mask"]
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

    rgb_train = torch.from_numpy(rgb[train_mask]).float()
    tok_train = torch.from_numpy(token_ids[train_mask]).long()
    att_train = torch.from_numpy(attention[train_mask]).long()
    y_train = torch.from_numpy(y[train_mask]).long()

    rgb_test = torch.from_numpy(rgb[test_mask]).float()
    tok_test = torch.from_numpy(token_ids[test_mask]).long()
    att_test = torch.from_numpy(attention[test_mask]).long()
    y_test = y[test_mask]

    model = TinyFusionClassifier(
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        vocab_size=int(cfg.get("vocab_size", 8192)),
        num_heads=int(cfg.get("num_heads", 4)),
    )
    best_ckpt = torch.load(run_dir / "checkpoints" / "best.ckpt", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    router = RouterMLP(in_dim=7, hidden_dim=16, num_experts=3)
    opt = torch.optim.Adam(router.parameters(), lr=args.lr)
    ds = TensorDataset(rgb_train, tok_train, att_train, y_train)
    loader = DataLoader(ds, batch_size=max(1, args.batch_size), shuffle=True)

    for _ in range(max(1, args.epochs)):
        router.train()
        for rgb_b, tok_b, att_b, y_b in loader:
            with torch.no_grad():
                out = model(rgb_b, tok_b, att_b)
                f = _router_features(out)
            opt.zero_grad()
            logits = router(f)
            with torch.no_grad():
                out = model(rgb_b, tok_b, att_b)
            mix_p = _mixture_probs(out, logits)
            loss = nn.NLLLoss()(torch.log(mix_p.clamp_min(1e-8)), y_b)
            loss.backward()
            opt.step()

    router.eval()
    with torch.no_grad():
        out_test = model(rgb_test, tok_test, att_test)
        f_test = _router_features(out_test)
        logits_test = router(f_test)
        p_test = _mixture_probs(out_test, logits_test).cpu().numpy()
    pred = p_test.argmax(axis=1)

    metrics = {
        "top1": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "n_test_samples": int(len(y_test)),
    }

    moe_dir = run_dir / "moe"
    moe_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"router_state": router.state_dict(), "config": cfg}, moe_dir / "router.ckpt")
    (moe_dir / "moe_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
