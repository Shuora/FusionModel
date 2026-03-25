from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

STAGE2_META_SCHEMA_VERSION = "stage2_meta_v1"
_LOGIT_BRANCHES: Tuple[str, ...] = ("img", "tls", "fuse")
_DEFAULT_SUMMARY_KEYS: Tuple[str, ...] = ("img_pooled_norm", "txt_pooled_norm", "fused_norm")
ROUTER_META_FEATURE_NAMES: Tuple[str, ...] = (
    "entropy_img",
    "entropy_tls",
    "entropy_fuse",
    "agreement_img_tls",
    "max_prob_img",
    "max_prob_tls",
    "max_prob_fuse",
)


def _require_2d_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={value.ndim}")
    return value


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)


def _summary_keys(summary: Mapping[str, Any] | None) -> list[str]:
    if not summary:
        return []
    keys = [k for k in _DEFAULT_SUMMARY_KEYS if k in summary]
    extras = sorted(k for k in summary.keys() if k not in _DEFAULT_SUMMARY_KEYS)
    return keys + extras


def _summary_block(
    summary: Mapping[str, Any] | None,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    keys = _summary_keys(summary)
    if not keys:
        return torch.empty((batch_size, 0), dtype=dtype, device=device), []

    cols = []
    for key in keys:
        raw = summary[key]
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=dtype, device=device)
        else:
            raw = raw.to(device=device, dtype=dtype)
        if raw.ndim == 1:
            raw = raw.unsqueeze(1)
        if raw.ndim != 2 or raw.shape[0] != batch_size or raw.shape[1] != 1:
            raise ValueError(
                f"summary[{key}] must be shaped [batch, 1], got {tuple(raw.shape)}"
            )
        cols.append(raw)
    return torch.cat(cols, dim=1), keys


def build_meta_feature_blocks(level1_output: Mapping[str, Any]) -> Dict[str, Any]:
    logits = {
        "img": _require_2d_tensor("logits_img", level1_output["logits_img"]),
        "tls": _require_2d_tensor("logits_tls", level1_output["logits_tls"]),
        "fuse": _require_2d_tensor("logits_fuse", level1_output["logits_fuse"]),
    }
    batch_size = logits["img"].shape[0]
    num_classes = logits["img"].shape[1]
    for name, tensor in logits.items():
        if tensor.shape != (batch_size, num_classes):
            raise ValueError(f"logits_{name} shape mismatch: expected {(batch_size, num_classes)}, got {tuple(tensor.shape)}")

    probs = {name: torch.softmax(tensor, dim=1) for name, tensor in logits.items()}
    entropy = torch.cat([_entropy(probs[name]) for name in _LOGIT_BRANCHES], dim=1)
    max_prob = torch.cat([probs[name].max(dim=1, keepdim=True).values for name in _LOGIT_BRANCHES], dim=1)
    agreement = (probs["img"] * probs["tls"]).sum(dim=1, keepdim=True)
    summary, _ = _summary_block(
        level1_output.get("summary"),
        batch_size=batch_size,
        dtype=logits["img"].dtype,
        device=logits["img"].device,
    )

    return {
        "logits": logits,
        "confidence": {
            "entropy": entropy,
            "max_prob": max_prob,
        },
        "agreement": agreement,
        "summary": summary,
    }


def flatten_meta_feature_blocks_tensor(level1_output: Mapping[str, Any]) -> tuple[torch.Tensor, list[str], dict]:
    blocks = build_meta_feature_blocks(level1_output)
    summary = level1_output.get("summary")
    summary_keys = _summary_keys(summary) if isinstance(summary, Mapping) else []

    feature_tensors = []
    feature_names: list[str] = []

    num_classes = blocks["logits"]["img"].shape[1]
    for branch in _LOGIT_BRANCHES:
        feature_tensors.append(blocks["logits"][branch])
        feature_names.extend(f"logits_{branch}_c{i}" for i in range(num_classes))

    feature_tensors.append(blocks["confidence"]["entropy"])
    feature_names.extend(f"entropy_{branch}" for branch in _LOGIT_BRANCHES)

    feature_tensors.append(blocks["confidence"]["max_prob"])
    feature_names.extend(f"max_prob_{branch}" for branch in _LOGIT_BRANCHES)

    feature_tensors.append(blocks["agreement"])
    feature_names.append("agreement_img_tls")

    if blocks["summary"].shape[1] > 0:
        feature_tensors.append(blocks["summary"])
        feature_names.extend(f"summary_{name}" for name in summary_keys)

    flat = torch.cat(feature_tensors, dim=1)
    schema = {
        "version": STAGE2_META_SCHEMA_VERSION,
        "dim": int(flat.shape[1]),
        "feature_names": feature_names,
    }
    return flat, feature_names, schema


def _resolve_feature_indices(feature_names: Sequence[str], selected_names: Sequence[str]) -> list[int]:
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in selected_names if name not in index]
    if missing:
        raise ValueError(f"missing selected meta features: {missing}")
    return [index[name] for name in selected_names]


def select_meta_feature_columns_tensor(
    flat_features: torch.Tensor,
    feature_names: Sequence[str],
    selected_names: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
    indices = _resolve_feature_indices(feature_names, selected_names)
    selected = flat_features[:, indices]
    return selected, [feature_names[i] for i in indices]


def build_router_meta_features(level1_output: Mapping[str, Any]) -> tuple[torch.Tensor, list[str], dict]:
    flat, feature_names, schema = flatten_meta_feature_blocks_tensor(level1_output)
    router_x, router_feature_names = select_meta_feature_columns_tensor(
        flat,
        feature_names,
        ROUTER_META_FEATURE_NAMES,
    )
    router_schema = {
        "version": schema["version"],
        "source_dim": schema["dim"],
        "dim": int(router_x.shape[1]),
        "feature_names": router_feature_names,
    }
    return router_x, router_feature_names, router_schema


def flatten_meta_feature_blocks(level1_output: Mapping[str, Any]) -> tuple[np.ndarray, list[str], dict]:
    flat, feature_names, schema = flatten_meta_feature_blocks_tensor(level1_output)
    flat_np = flat.detach().cpu().numpy().astype(np.float32, copy=False)
    return flat_np, feature_names, schema


__all__ = [
    "ROUTER_META_FEATURE_NAMES",
    "STAGE2_META_SCHEMA_VERSION",
    "build_meta_feature_blocks",
    "build_router_meta_features",
    "flatten_meta_feature_blocks",
    "flatten_meta_feature_blocks_tensor",
    "select_meta_feature_columns_tensor",
]
