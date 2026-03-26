from __future__ import annotations

import torch
from torch import nn

from fusion_malicious.models.fusion import BidirectionalCrossAttention, GatedFusion
from fusion_malicious.models.image_encoder import ImageEncoder
from fusion_malicious.models.text_encoder import TextEncoder


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        text_backbone: nn.Module,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_backbone)
        self.text_encoder = TextEncoder(text_backbone)
        self.cross_attention = BidirectionalCrossAttention(hidden_dim, num_heads)
        self.gated_fusion = GatedFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image)
        text_tokens = self.text_encoder(input_ids, attention_mask)
        image_attended, text_attended = self.cross_attention(image_tokens, text_tokens)
        image_summary = image_attended.mean(dim=1)
        text_summary = text_attended.mean(dim=1)
        fused = self.gated_fusion(image_summary, text_summary)
        return self.classifier(fused)
