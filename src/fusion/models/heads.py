from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class GateHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, image_feat: torch.Tensor, tls_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_feat, tls_feat], dim=-1)
        return torch.sigmoid(self.fc(x))
