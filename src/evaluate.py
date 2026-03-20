from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.structured_logging import format_log_line
from src.models.fusion_model import MobileViTETBertFusionClassifier
from src.pipeline_data import load_policy_multimodal_data
from src.runtime_device import resolve_runtime_device


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate TLS fusion model")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--allow-split-fallback", action="store_true", default=False)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    device_preference = args.device or cfg.get("device_requested") or cfg.get("device", "auto")
    requested_device, resolved_device, device_fallback = resolve_runtime_device(device_preference)
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
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    y = data["y"]
    split = data["split"]

    requested_split = str(args.split)
    effective_split = requested_split
    fallback_used = False

    mask = split == requested_split
    if not np.any(mask) and args.allow_split_fallback:
        mask = split == "val"
        if np.any(mask):
            effective_split = "val"
            fallback_used = True
    if not np.any(mask) and args.allow_split_fallback:
        mask = np.ones_like(split, dtype=bool)
        if np.any(mask):
            effective_split = "all"
            fallback_used = True
    if not np.any(mask):
        print(
            format_log_line(
                level="error",
                module="eval",
                event="missing_eval_split",
                kv={"split": requested_split, "fallback_enabled": args.allow_split_fallback},
            )
        )
        return 2

    rgb_eval = torch.from_numpy(rgb[mask]).float().to(device)
    input_eval = torch.from_numpy(input_ids[mask]).long().to(device)
    attn_eval = torch.from_numpy(attention_mask[mask]).long().to(device)
    token_type_eval = torch.from_numpy(token_type_ids[mask]).long().to(device)
    y_eval = y[mask]

    ckpt_name = "best.ckpt" if args.checkpoint == "best" else "last.ckpt"
    ckpt = torch.load(run_dir / "checkpoints" / ckpt_name, map_location="cpu")
    model = MobileViTETBertFusionClassifier(
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        vocab_size=int(cfg.get("vocab_size", 30522)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(rgb_eval, input_eval, attn_eval, token_type_eval)
    if cfg.get("stage") == "warmup":
        logits = (out["logits_img"] + out["logits_tls"]) / 2.0
    else:
        logits = out["logits_fuse"]

    logits_np = logits.cpu().numpy()
    pred = np.argmax(logits_np, axis=1)

    top1 = float(accuracy_score(y_eval, pred))
    macro_precision = float(precision_score(y_eval, pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_eval, pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_eval, pred, average="macro", zero_division=0))
    gate_mean = float(out["gate"].mean().item())
    cm = confusion_matrix(y_eval, pred, labels=list(range(int(cfg["num_classes"]))))

    payload = {
        "run_id": cfg["run_id"],
        "split": effective_split,
        "requested_split": requested_split,
        "effective_split": effective_split,
        "fallback_used": fallback_used,
        "top1": top1,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "gate_mean": gate_mean,
        "num_samples": int(mask.sum()),
        "checkpoint": ckpt_name,
        "device_requested": requested_device,
        "device": resolved_device,
        "device_fallback": device_fallback,
    }
    out_json = run_dir / f"eval_{effective_split}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm).to_csv(fig_dir / f"confusion_matrix_{effective_split}.csv", index=False)
    _save_confusion_png(cm=cm, output_path=fig_dir / f"confusion_matrix_{effective_split}.png")

    print(
        format_log_line(
            level="success",
            module="eval",
            event="eval_done",
            kv={
                "requested_split": requested_split,
                "effective_split": effective_split,
                "fallback_used": fallback_used,
                "top1": f"{top1:.4f}",
                "macro_precision": f"{macro_precision:.4f}",
                "macro_f1": f"{macro_f1:.4f}",
                "gate_mean": f"{gate_mean:.4f}",
            },
        )
    )
    return 0


def _save_confusion_png(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
