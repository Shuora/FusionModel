from __future__ import annotations

import torch
from torch import nn


class TlsBertBranch(nn.Module):
    """Embedding + mask-aware pooling as minimal TLS token branch."""

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(token_ids)
        if attn_mask is None:
            return emb.mean(dim=1)

        mask = attn_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (emb * mask).sum(dim=1) / denom
