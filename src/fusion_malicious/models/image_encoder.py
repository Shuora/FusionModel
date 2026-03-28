from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, output_dim: int | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim
        self.projection: nn.Linear | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(image)
        if not torch.is_tensor(tokens):
            raise TypeError("Image backbone must return a tensor")

        if tokens.dim() == 3:
            pass

        elif tokens.dim() == 4:
            batch, channels, height, width = tokens.shape
            flatten = tokens.flatten(2)  # [B, C, H*W]
            tokens = flatten.transpose(1, 2).reshape(batch, height * width, channels)

        elif tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)

        else:
            raise ValueError(f"Unsupported image backbone output shape: {tokens.shape}")

        return self._project_if_needed(tokens)

    def _project_if_needed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.output_dim is None:
            return tokens

        feature_dim = tokens.size(-1)
        if feature_dim == self.output_dim:
            return tokens

        projection = self.projection
        if projection is None or projection.in_features != feature_dim:
            projection = nn.Linear(feature_dim, self.output_dim)
            projection.to(device=tokens.device, dtype=tokens.dtype)
            self.projection = projection

        return projection(tokens)
