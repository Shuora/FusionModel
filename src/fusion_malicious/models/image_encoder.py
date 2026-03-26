from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(image)
        if not torch.is_tensor(tokens):
            raise TypeError("Image backbone must return a tensor")

        if tokens.dim() == 3:
            return tokens

        if tokens.dim() == 4:
            batch, channels, height, width = tokens.shape
            flatten = tokens.flatten(2)  # [B, C, H*W]
            return flatten.transpose(1, 2).reshape(batch, height * width, channels)

        if tokens.dim() == 2:
            return tokens.unsqueeze(1)

        raise ValueError(f"Unsupported image backbone output shape: {tokens.shape}")
