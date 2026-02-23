from __future__ import annotations

import torch
from torch import nn


class ImageBranch(nn.Module):
    """Lightweight image branch for TLS-RGB inputs."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(image_tensor)
