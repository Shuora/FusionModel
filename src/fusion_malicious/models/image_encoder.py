from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)
