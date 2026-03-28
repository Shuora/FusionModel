from __future__ import annotations

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, output_dim: int | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim
        self.projection: nn.Linear | None = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(result, dict):
            result = result.get("last_hidden_state", result)
        elif hasattr(result, "last_hidden_state"):
            result = getattr(result, "last_hidden_state")

        if not torch.is_tensor(result):
            raise TypeError("Text backbone must return tensor-like hidden states")

        if result.dim() == 2:
            result = result.unsqueeze(1)
        if result.dim() != 3:
            raise ValueError(f"Unsupported text backbone output shape: {result.shape}")

        return self._project_if_needed(result)

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
