from __future__ import annotations

import torch
from torch import nn


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_attended, _ = self.image_to_text(image_tokens, text_tokens, text_tokens)
        text_attended, _ = self.text_to_image(text_tokens, image_tokens, image_tokens)
        return image_attended, text_attended


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.transform = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, image_summary: torch.Tensor, text_summary: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([image_summary, text_summary], dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        transform = torch.tanh(self.transform(fused))
        return gate * transform
