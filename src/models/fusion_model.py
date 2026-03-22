from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.etbert_backbone import ETBertBackbone
from src.models.mobilevit_backbone import MobileViTBackbone


class MobileViTETBertFusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        vocab_size: int = 30522,
        max_tokens: int = 128,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.image_backbone = MobileViTBackbone(out_dim=hidden_dim)
        self.text_backbone = ETBertBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, max_tokens=max_tokens)
        self.fusion_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.modality_fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.fusion_norm1 = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm2 = nn.LayerNorm(hidden_dim)
        self.head_fuse = nn.Linear(hidden_dim, num_classes)
        self.head_img = nn.Linear(hidden_dim, num_classes)
        self.head_tls = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        img_feature = self.image_backbone(rgb)
        tls_feature = self.text_backbone(input_ids, attention_mask, token_type_ids)

        # Two modality tokens (image/text) are fused by a learnable attention query.
        modality_tokens = torch.stack([img_feature, tls_feature], dim=1)
        query = self.fusion_query.expand(modality_tokens.shape[0], -1, -1)
        attn_out, attn_weights = self.modality_fusion_attn(query, modality_tokens, modality_tokens, need_weights=True)
        fused = self.fusion_norm1(query + attn_out)
        fused = self.fusion_norm2(fused + self.fusion_mlp(fused))
        fused_feature = fused.squeeze(1)

        # Keep `gate` for downstream compatibility: attention weight assigned to image token.
        gate = attn_weights[:, :, 0]
        return {
            "logits_fuse": self.head_fuse(fused_feature),
            "logits_img": self.head_img(img_feature),
            "logits_tls": self.head_tls(tls_feature),
            "gate": gate,
        }
