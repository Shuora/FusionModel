from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.common.structured_logging import format_log_line
from src.models.fusion_model import TinyFusionClassifier
from src.pipeline_data import load_policy_multimodal_data


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate TLS fusion model")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="best")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    data = load_policy_multimodal_data(cfg["processed_root"], cfg["policy"])

    rgb = data["rgb"]
    token_ids = data["token_ids"]
    attention_mask = data["attention_mask"]
    y = data["y"]
    split = data["split"]

    mask = split == args.split
    if not np.any(mask):
        mask = split == "val"
    if not np.any(mask):
        mask = np.ones_like(split, dtype=bool)

    rgb_eval = torch.from_numpy(rgb[mask]).float()
    token_eval = torch.from_numpy(token_ids[mask]).long()
    attn_eval = torch.from_numpy(attention_mask[mask]).long()
    y_eval = y[mask]

    ckpt_name = "best.ckpt" if args.checkpoint == "best" else "last.ckpt"
    ckpt = torch.load(run_dir / "checkpoints" / ckpt_name, map_location="cpu")
    model = TinyFusionClassifier(
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        vocab_size=int(cfg.get("vocab_size", 8192)),
        num_heads=int(cfg.get("num_heads", 4)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(rgb_eval, token_eval, attn_eval)
    if cfg.get("stage") == "warmup":
        logits = (out["logits_img"] + out["logits_tls"]) / 2.0
    else:
        logits = out["logits_fuse"]

    logits_np = logits.cpu().numpy()
    pred = np.argmax(logits_np, axis=1)

    top1 = float(accuracy_score(y_eval, pred))
    macro_f1 = float(f1_score(y_eval, pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_eval, pred, average="macro", zero_division=0))
    gate_mean = float(out["gate"].mean().item())
    cm = confusion_matrix(y_eval, pred, labels=list(range(int(cfg["num_classes"]))))

    payload = {
        "run_id": cfg["run_id"],
        "split": args.split,
        "top1": top1,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "gate_mean": gate_mean,
        "num_samples": int(mask.sum()),
        "checkpoint": ckpt_name,
    }
    out_json = run_dir / f"eval_{args.split}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm).to_csv(fig_dir / f"confusion_matrix_{args.split}.csv", index=False)

    print(
        format_log_line(
            level="success",
            module="eval",
            event="eval_done",
            kv={
                "split": args.split,
                "top1": f"{top1:.4f}",
                "macro_f1": f"{macro_f1:.4f}",
                "gate_mean": f"{gate_mean:.4f}",
            },
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
