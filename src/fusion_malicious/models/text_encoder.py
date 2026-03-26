from __future__ import annotations

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(result, dict):
            result = result.get("last_hidden_state", result)
        elif hasattr(result, "last_hidden_state"):
            result = getattr(result, "last_hidden_state")

        if not torch.is_tensor(result):
            raise TypeError("Text backbone must return tensor-like hidden states")

        return result
