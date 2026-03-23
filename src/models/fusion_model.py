from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.etbert_backbone import ETBertBackbone
from src.models.mobilevit_backbone import MobileViTBackbone


def _normalize_heads_num(hidden_dim: int, heads_num: int) -> int:
    heads_num = max(1, int(heads_num))
    if hidden_dim % heads_num == 0:
        return heads_num
    for candidate in range(min(hidden_dim, heads_num), 0, -1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (tokens * valid).sum(dim=1) / denom


class BidirectionalFusionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm1 = nn.LayerNorm(hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim)
        self.image_norm1 = nn.LayerNorm(hidden_dim)
        self.image_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_update, _ = self.text_to_image(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        text_tokens = self.text_norm1(text_tokens + self.dropout(text_update))
        text_tokens = self.text_norm2(text_tokens + self.text_ffn(text_tokens))

        image_update, _ = self.image_to_text(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_mask <= 0,
            need_weights=False,
        )
        image_tokens = self.image_norm1(image_tokens + self.dropout(image_update))
        image_tokens = self.image_norm2(image_tokens + self.image_ffn(image_tokens))
        return image_tokens, text_tokens


class BidirectionalFusionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BidirectionalFusionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            image_tokens, text_tokens = layer(image_tokens=image_tokens, text_tokens=text_tokens, text_mask=text_mask)
        return image_tokens, text_tokens


class MobileViTETBertFusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        vocab_size: int = 30522,
        max_tokens: int = 128,
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_backbone = MobileViTBackbone(out_dim=hidden_dim)
        self.text_backbone = ETBertBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, max_tokens=max_tokens)
        normalized_heads = _normalize_heads_num(hidden_dim=hidden_dim, heads_num=fusion_heads)
        self.fusion_encoder = BidirectionalFusionEncoder(
            hidden_dim=hidden_dim,
            num_layers=max(1, fusion_layers),
            num_heads=normalized_heads,
            dropout=dropout,
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.head_fuse = nn.Linear(hidden_dim, num_classes)
        self.head_img = nn.Linear(hidden_dim, num_classes)
        self.head_tls = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        return_features: bool = False,
        use_fusion: bool = True,
    ) -> Dict[str, torch.Tensor]:
        img_features = self.image_backbone.forward_features(rgb)
        txt_features = self.text_backbone.forward_features(input_ids, attention_mask, token_type_ids)
        img_tokens = img_features["tokens"]
        txt_tokens = txt_features["tokens"]
        if use_fusion:
            img_tokens, txt_tokens = self.fusion_encoder(
                image_tokens=img_tokens,
                text_tokens=txt_tokens,
                text_mask=txt_features["mask"],
            )
            img_ctx = img_tokens.mean(dim=1)
            txt_ctx = _masked_mean(txt_tokens, txt_features["mask"])
        else:
            img_ctx = img_features["pooled"]
            txt_ctx = txt_features["pooled"]
        fused = self.fusion_proj(torch.cat([img_ctx, txt_ctx], dim=-1))
        out = {
            "logits_fuse": self.head_fuse(fused),
            "logits_img": self.head_img(img_ctx),
            "logits_tls": self.head_tls(txt_ctx),
        }
        if return_features:
            out["img_tokens"] = img_tokens
            out["txt_tokens"] = txt_tokens
        return out
