from __future__ import annotations

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)
