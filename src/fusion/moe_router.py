from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class MoeRouter(nn.Module):
    """Lightweight MoE router for decision-level fusion experiments."""

    def __init__(self, input_dim: int, num_experts: int, num_classes: int) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")

        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, num_classes) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_logits = self.gate(x)
        gates = torch.softmax(gate_logits, dim=-1)

        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        fused_logits = torch.sum(expert_logits * gates.unsqueeze(-1), dim=1)

        return {
            "logits": fused_logits,
            "gates": gates,
            "expert_logits": expert_logits,
        }
