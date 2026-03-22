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
