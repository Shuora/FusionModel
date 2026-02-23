from __future__ import annotations

import torch
from torch import nn

from src.fusion.models.heads import ClassificationHead, GateHead
from src.fusion.models.image_branch import ImageBranch
from src.fusion.models.tls_bert_branch import TlsBertBranch


class FusionModel(nn.Module):
    """Bidirectional cross-attn + gated late fusion."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        vocab_size: int = 4096,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.image_branch = ImageBranch(hidden_dim=hidden_dim)
        self.tls_branch = TlsBertBranch(vocab_size=vocab_size, hidden_dim=hidden_dim)

        self.img_to_tls_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.tls_to_img_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

        self.head_img = ClassificationHead(hidden_dim=hidden_dim, num_classes=num_classes)
        self.head_tls = ClassificationHead(hidden_dim=hidden_dim, num_classes=num_classes)
        self.gate_head = GateHead(hidden_dim=hidden_dim)

    def forward(
        self,
        image_tensor: torch.Tensor,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_feat = self.image_branch(image_tensor)
        tls_feat = self.tls_branch(token_ids, attn_mask)

        image_ctx, _ = self.img_to_tls_attn(image_feat.unsqueeze(1), tls_feat.unsqueeze(1), tls_feat.unsqueeze(1))
        tls_ctx, _ = self.tls_to_img_attn(tls_feat.unsqueeze(1), image_feat.unsqueeze(1), image_feat.unsqueeze(1))

        image_ctx = image_ctx.squeeze(1)
        tls_ctx = tls_ctx.squeeze(1)

        logits_img = self.head_img(image_ctx)
        logits_tls = self.head_tls(tls_ctx)
        gate = self.gate_head(image_ctx, tls_ctx)
        logits_fuse = gate * logits_img + (1.0 - gate) * logits_tls

        return {
            "logits_fuse": logits_fuse,
            "logits_img": logits_img,
            "logits_tls": logits_tls,
            "gate": gate,
        }
