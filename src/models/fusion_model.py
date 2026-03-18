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
        vocab_size: int = 30522,
        max_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.image_backbone = MobileViTBackbone(out_dim=hidden_dim)
        self.text_backbone = ETBertBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, max_tokens=max_tokens)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
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
    ) -> Dict[str, torch.Tensor]:
        img_feature = self.image_backbone(rgb)
        tls_feature = self.text_backbone(input_ids, attention_mask, token_type_ids)
        gate = self.gate(torch.cat([img_feature, tls_feature], dim=-1))
        fused = gate * img_feature + (1.0 - gate) * tls_feature
        return {
            "logits_fuse": self.head_fuse(fused),
            "logits_img": self.head_img(img_feature),
            "logits_tls": self.head_tls(tls_feature),
            "gate": gate,
        }


class TinyFusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        vocab_size: int = 8192,
        num_heads: int = 4,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens

        # Image branch
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)

        # TLS token branch
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_tokens, hidden_dim)
        self.tls_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention fusion
        self.cross_img_to_tls = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_tls_to_img = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Heads
        self.head_fuse = nn.Linear(hidden_dim, num_classes)
        self.head_img = nn.Linear(hidden_dim, num_classes)
        self.head_tls = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, rgb: torch.Tensor, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        bsz = rgb.size(0)
        token_len = token_ids.size(1)
        if token_len > self.max_tokens:
            token_ids = token_ids[:, : self.max_tokens]
            attention_mask = attention_mask[:, : self.max_tokens]
            token_len = self.max_tokens

        # Image tokens: [B, C, H, W] -> [B, N_img, D]
        img_feat = self.img_conv(rgb)  # [B, D, 7, 7]
        img_tokens = img_feat.flatten(2).transpose(1, 2)  # [B, 49, D]
        img_tokens = self.img_proj(img_tokens)
        img_pooled = img_tokens.mean(dim=1)

        # TLS tokens: embedding + positional
        pos = torch.arange(token_len, device=token_ids.device).unsqueeze(0).expand(bsz, token_len)
        tls_tokens = self.token_embed(token_ids) + self.pos_embed(pos)
        tls_tokens = self.tls_proj(tls_tokens)

        key_padding_mask = attention_mask <= 0
        # if a sample is fully masked, unmask first token to avoid NaN in attention
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        z_img, _ = self.cross_img_to_tls(
            query=img_tokens,
            key=tls_tokens,
            value=tls_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        z_tls, _ = self.cross_tls_to_img(
            query=tls_tokens,
            key=img_tokens,
            value=img_tokens,
            need_weights=False,
        )

        p_img = z_img.mean(dim=1)
        # masked mean pooling for tls branch
        valid = attention_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        p_tls = (z_tls * valid).sum(dim=1) / denom

        g = self.gate(torch.cat([p_img, p_tls], dim=-1))
        fused = g * p_img + (1.0 - g) * p_tls

        return {
            "logits_fuse": self.head_fuse(fused),
            "logits_img": self.head_img(img_pooled),
            "logits_tls": self.head_tls(p_tls),
            "gate": g,
        }
