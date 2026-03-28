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
        self.hidden_dim = hidden_dim
        self.image_encoder = ImageEncoder(image_backbone)
        self.text_encoder = TextEncoder(text_backbone)
        self.image_projection: nn.Linear | None = None
        self.text_projection: nn.Linear | None = None
        self.cross_attention = BidirectionalCrossAttention(hidden_dim, num_heads)
        self.gated_fusion = GatedFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image)
        text_tokens = self.text_encoder(input_ids, attention_mask)
        image_tokens = self._project_tokens(image_tokens, "image_projection")
        text_tokens = self._project_tokens(text_tokens, "text_projection")
        text_key_padding_mask = attention_mask == 0
        if text_tokens.size(1) != text_key_padding_mask.size(1):
            mask_len = text_key_padding_mask.size(1)
            padding_len = text_tokens.size(1) - mask_len
            if padding_len < 0:
                text_key_padding_mask = text_key_padding_mask[:, : text_tokens.size(1)]
            else:
                pad = torch.zeros(
                    text_key_padding_mask.size(0),
                    padding_len,
                    dtype=text_key_padding_mask.dtype,
                    device=text_key_padding_mask.device,
                )
                text_key_padding_mask = torch.cat([text_key_padding_mask, pad], dim=1)
        image_attended, text_attended = self.cross_attention(
            image_tokens,
            text_tokens,
            text_key_padding_mask=text_key_padding_mask,
        )
        image_summary = image_attended.mean(dim=1)
        valid_mask = (~text_key_padding_mask).unsqueeze(-1).to(dtype=text_attended.dtype)
        numerator = (text_attended * valid_mask).sum(dim=1)
        denominator = valid_mask.sum(dim=1).clamp(min=1)
        text_summary = numerator / denominator
        fused = self.gated_fusion(image_summary, text_summary)
        return self.classifier(fused)

    def _project_tokens(self, tokens: torch.Tensor, projection_attr: str) -> torch.Tensor:
        feature_dim = tokens.size(-1)
        if feature_dim == self.hidden_dim:
            return tokens

        projection = getattr(self, projection_attr)
        if projection is None or projection.in_features != feature_dim:
            projection = nn.Linear(feature_dim, self.hidden_dim)
            projection.to(device=tokens.device, dtype=tokens.dtype)
            setattr(self, projection_attr, projection)

        return projection(tokens)
