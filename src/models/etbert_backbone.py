from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


DEFAULT_ETBERT_CONFIG: Dict[str, Any] = {
    "emb_size": 128,
    "feedforward_size": 512,
    "hidden_size": 128,
    "heads_num": 2,
    "layers_num": 2,
    "max_seq_length": 128,
    "dropout": 0.1,
}


def _normalize_heads_num(hidden_size: int, heads_num: int) -> int:
    heads_num = max(1, int(heads_num))
    if hidden_size % heads_num == 0:
        return heads_num
    for candidate in range(min(hidden_size, heads_num), 0, -1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def _is_tensor_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and all(isinstance(v, torch.Tensor) for v in obj.values())


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if _is_tensor_dict(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if _is_tensor_dict(value):
                return value
    raise TypeError("Unsupported ET-BERT checkpoint format")


def _strip_checkpoint_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "bert.", "backbone.", "text_backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _map_external_key(key: str) -> str | None:
    key = _strip_checkpoint_prefixes(key)

    embedding_map = {
        "embeddings.word_embeddings.weight": "token_embed.weight",
        "embeddings.position_embeddings.weight": "pos_embed.weight",
        "embeddings.token_type_embeddings.weight": "type_embed.weight",
        "embedding.word_embedding.weight": "token_embed.weight",
        "embedding.position_embedding.weight": "pos_embed.weight",
        "embedding.segment_embedding.weight": "type_embed.weight",
    }
    if key in embedding_map:
        return embedding_map[key]

    m = re.fullmatch(r"encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)", key)
    if m:
        layer_idx, wb = m.groups()
        return f"encoder.layers.{layer_idx}.self_attn.out_proj.{wb}"
    m = re.fullmatch(r"encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)", key)
    if m:
        layer_idx, wb = m.groups()
        return f"encoder.layers.{layer_idx}.linear1.{wb}"
    m = re.fullmatch(r"encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)", key)
    if m:
        layer_idx, wb = m.groups()
        return f"encoder.layers.{layer_idx}.linear2.{wb}"
    m = re.fullmatch(r"encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.(weight|bias)", key)
    if m:
        layer_idx, wb = m.groups()
        return f"encoder.layers.{layer_idx}.norm1.{wb}"
    m = re.fullmatch(r"encoder\.layer\.(\d+)\.output\.LayerNorm\.(weight|bias)", key)
    if m:
        layer_idx, wb = m.groups()
        return f"encoder.layers.{layer_idx}.norm2.{wb}"

    m = re.fullmatch(r"encoder\.transformer\.(\d+)\.self_attn\.final_linear\.(weight|bias)", key)
    if m:
        layer_idx, wb = m.groups()
        return f"encoder.layers.{layer_idx}.self_attn.out_proj.{wb}"
    m = re.fullmatch(r"encoder\.transformer\.(\d+)\.feed_forward\.linear_([12])\.(weight|bias)", key)
    if m:
        layer_idx, linear_idx, wb = m.groups()
        linear_name = "linear1" if linear_idx == "1" else "linear2"
        return f"encoder.layers.{layer_idx}.{linear_name}.{wb}"
    m = re.fullmatch(r"encoder\.transformer\.(\d+)\.layer_norm_([12])\.(gamma|beta|weight|bias)", key)
    if m:
        layer_idx, norm_idx, gb = m.groups()
        norm_name = "norm1" if norm_idx == "1" else "norm2"
        param_name = "weight" if gb in ("gamma", "weight") else "bias"
        return f"encoder.layers.{layer_idx}.{norm_name}.{param_name}"
    return key


def _normalize_external_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, list[str]], list[Dict[str, Any]]]:
    normalized: Dict[str, torch.Tensor] = {}
    sources: Dict[str, list[str]] = {}
    qkv_weights: Dict[str, Dict[int, torch.Tensor]] = {}
    qkv_biases: Dict[str, Dict[int, torch.Tensor]] = {}
    qkv_weight_sources: Dict[str, Dict[int, str]] = {}
    qkv_bias_sources: Dict[str, Dict[int, str]] = {}
    attn_index_map = {"query": 0, "key": 1, "value": 2}
    attn_name_by_idx = {0: "query", 1: "key", 2: "value"}
    incomplete_qkv_groups: list[Dict[str, Any]] = []

    for raw_key, value in state_dict.items():
        stripped = _strip_checkpoint_prefixes(raw_key)

        m = re.fullmatch(r"encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)", stripped)
        if m:
            layer_idx, part_name, wb = m.groups()
            target_key = f"encoder.layers.{layer_idx}.self_attn.in_proj_{wb}"
            part_idx = attn_index_map[part_name]
            if wb == "weight":
                qkv_weights.setdefault(target_key, {})[part_idx] = value
                qkv_weight_sources.setdefault(target_key, {})[part_idx] = raw_key
            else:
                qkv_biases.setdefault(target_key, {})[part_idx] = value
                qkv_bias_sources.setdefault(target_key, {})[part_idx] = raw_key
            continue

        m = re.fullmatch(r"encoder\.transformer\.(\d+)\.self_attn\.linear_layers\.([012])\.(weight|bias)", stripped)
        if m:
            layer_idx, part_name, wb = m.groups()
            target_key = f"encoder.layers.{layer_idx}.self_attn.in_proj_{wb}"
            part_idx = int(part_name)
            if wb == "weight":
                qkv_weights.setdefault(target_key, {})[part_idx] = value
                qkv_weight_sources.setdefault(target_key, {})[part_idx] = raw_key
            else:
                qkv_biases.setdefault(target_key, {})[part_idx] = value
                qkv_bias_sources.setdefault(target_key, {})[part_idx] = raw_key
            continue

        mapped_key = _map_external_key(raw_key)
        if mapped_key is not None:
            normalized[mapped_key] = value
            sources[mapped_key] = [raw_key]

    for target_key, parts in qkv_weights.items():
        if {0, 1, 2} <= set(parts.keys()):
            normalized[target_key] = torch.cat([parts[0], parts[1], parts[2]], dim=0)
            src_parts = qkv_weight_sources.get(target_key, {})
            sources[target_key] = [src_parts.get(0, ""), src_parts.get(1, ""), src_parts.get(2, "")]
        else:
            src_parts = qkv_weight_sources.get(target_key, {})
            missing = [attn_name_by_idx[idx] for idx in (0, 1, 2) if idx not in parts]
            incomplete_qkv_groups.append(
                {
                    "mapped_key": target_key,
                    "source_keys": [src_parts[idx] for idx in (0, 1, 2) if idx in src_parts],
                    "reason": f"incomplete_qkv_group:missing={','.join(missing)}",
                }
            )
    for target_key, parts in qkv_biases.items():
        if {0, 1, 2} <= set(parts.keys()):
            normalized[target_key] = torch.cat([parts[0], parts[1], parts[2]], dim=0)
            src_parts = qkv_bias_sources.get(target_key, {})
            sources[target_key] = [src_parts.get(0, ""), src_parts.get(1, ""), src_parts.get(2, "")]
        else:
            src_parts = qkv_bias_sources.get(target_key, {})
            missing = [attn_name_by_idx[idx] for idx in (0, 1, 2) if idx not in parts]
            incomplete_qkv_groups.append(
                {
                    "mapped_key": target_key,
                    "source_keys": [src_parts[idx] for idx in (0, 1, 2) if idx in src_parts],
                    "reason": f"incomplete_qkv_group:missing={','.join(missing)}",
                }
            )

    cleaned_sources = {k: [s for s in v if s] for k, v in sources.items()}
    return normalized, cleaned_sources, incomplete_qkv_groups


class ETBertBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int | None = None,
        max_tokens: int | None = None,
        num_layers: int | None = None,
        config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.last_checkpoint_report: Dict[str, Any] = {
            "checkpoint_found": False,
            "checkpoint_path": None,
            "status": ["not_requested"],
            "raw_key_count": 0,
            "normalized_key_count": 0,
            "loaded_key_count": 0,
            "skipped_key_count": 0,
            "skipped_keys": [],
            "missing_keys": [],
            "unexpected_keys": [],
        }
        self.checkpoint_report = self.last_checkpoint_report

        etbert_cfg = dict(DEFAULT_ETBERT_CONFIG)
        if config_path is not None:
            with Path(config_path).open("r", encoding="utf-8") as fp:
                etbert_cfg.update(json.load(fp))
        if config is not None:
            etbert_cfg.update(dict(config))

        resolved_hidden_dim = int(hidden_dim or etbert_cfg.get("hidden_size") or etbert_cfg.get("emb_size", 128))
        resolved_max_tokens = int(max_tokens or etbert_cfg.get("max_seq_length", 128))
        cfg_layers = int(etbert_cfg.get("layers_num", 2))
        requested_layers = cfg_layers if num_layers is None else int(num_layers)
        effective_layers = max(1, min(requested_layers, cfg_layers))
        heads_num = _normalize_heads_num(resolved_hidden_dim, int(etbert_cfg.get("heads_num", 2)))
        feedforward_size = int(etbert_cfg.get("feedforward_size", resolved_hidden_dim * 4))
        dropout = float(etbert_cfg.get("dropout", 0.1))

        self.max_tokens = resolved_max_tokens
        self.num_layers = effective_layers
        self.config_layers_num = cfg_layers
        self.requested_layers = requested_layers
        self.hidden_dim = resolved_hidden_dim
        self.etbert_config = etbert_cfg

        self.token_embed = nn.Embedding(vocab_size, resolved_hidden_dim)
        self.type_embed = nn.Embedding(2, resolved_hidden_dim)
        self.pos_embed = nn.Embedding(resolved_max_tokens, resolved_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resolved_hidden_dim,
            nhead=heads_num,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # Disable the prototype nested-tensor fast path to avoid PyTorch runtime warnings
        # in training/evaluation logs while keeping the same encoder semantics.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=effective_layers,
            enable_nested_tensor=False,
        )
        if checkpoint_path is not None:
            self.last_checkpoint_report = self.load_pretrained_checkpoint(checkpoint_path)
            self.checkpoint_report = self.last_checkpoint_report

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        checkpoint_path_str = str(path)
        if not path.exists():
            report = {
                "checkpoint_found": False,
                "checkpoint_path": checkpoint_path_str,
                "status": ["checkpoint_missing"],
                "raw_key_count": 0,
                "normalized_key_count": 0,
                "loaded_key_count": 0,
                "skipped_key_count": 0,
                "skipped_keys": [],
                "missing_keys": [],
                "unexpected_keys": [],
            }
            self.last_checkpoint_report = report
            self.checkpoint_report = report
            return report

        payload = torch.load(path, map_location="cpu")
        raw_state_dict = _extract_state_dict(payload)
        normalized_state_dict, mapped_sources, incomplete_qkv_groups = _normalize_external_state_dict(raw_state_dict)

        target_state_dict = self.state_dict()
        loadable_state_dict: Dict[str, torch.Tensor] = {}
        filtered_out_keys: list[str] = []
        skipped_keys: list[Dict[str, Any]] = list(incomplete_qkv_groups)
        for key, value in normalized_state_dict.items():
            target = target_state_dict.get(key)
            if target is None:
                filtered_out_keys.append(key)
                skipped_keys.append(
                    {
                        "mapped_key": key,
                        "source_keys": mapped_sources.get(key, []),
                        "reason": "target_key_missing",
                    }
                )
                continue
            if target.shape != value.shape:
                filtered_out_keys.append(key)
                skipped_keys.append(
                    {
                        "mapped_key": key,
                        "source_keys": mapped_sources.get(key, []),
                        "reason": f"shape_mismatch:{tuple(value.shape)}!={tuple(target.shape)}",
                    }
                )
                continue
            loadable_state_dict[key] = value

        incompatible = self.load_state_dict(loadable_state_dict, strict=False)
        unexpected_keys = sorted(set(incompatible.unexpected_keys).union(filtered_out_keys))
        status: list[str] = []
        if loadable_state_dict:
            status.append("loaded")
        else:
            status.append("no_compatible_keys")
        if skipped_keys:
            status.append("partial")

        report = {
            "checkpoint_found": True,
            "checkpoint_path": checkpoint_path_str,
            "status": status,
            "raw_key_count": len(raw_state_dict),
            "normalized_key_count": len(normalized_state_dict),
            "loaded_key_count": len(loadable_state_dict),
            "skipped_key_count": len(skipped_keys),
            "skipped_keys": skipped_keys,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": unexpected_keys,
        }
        self.last_checkpoint_report = report
        self.checkpoint_report = report
        return report

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_len = min(input_ids.size(1), self.max_tokens)
        input_ids = input_ids[:, :token_len]
        attention_mask = attention_mask[:, :token_len]
        token_type_ids = token_type_ids[:, :token_len]

        pos = torch.arange(token_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), token_len)
        x = self.token_embed(input_ids) + self.type_embed(token_type_ids.clamp(min=0, max=1)) + self.pos_embed(pos)
        key_padding_mask = attention_mask <= 0
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        valid = attention_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom
