from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn

from src.models.etbert_backbone import ETBertBackbone
from src.models.mobilevit_backbone import MobileViTBackbone


def _normalize_heads_num(hidden_dim: int, heads_num: int) -> int:
    heads_num = max(1, int(heads_num))
    if hidden_dim % heads_num == 0:
        return heads_num
    for candidate in range(min(hidden_dim, heads_num), 0, -1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


class DatasetConditionedHead(nn.Module):
    def __init__(self, hidden_dim: int, dataset_order: tuple[str, ...], output_dims: dict[str, int]) -> None:
        super().__init__()
        self.dataset_order = tuple(dataset_order)
        self.output_dims = {name: int(output_dims[name]) for name in self.dataset_order}
        self.projections = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, self.output_dims[name]) for name in self.dataset_order}
        )

    def forward(self, fused: torch.Tensor, dataset_name: str) -> torch.Tensor:
        return self.projections[str(dataset_name)](fused)


class Stage2UnifiedSummary(TypedDict):
    img_pooled_norm: torch.Tensor
    seq_pooled_norm: torch.Tensor
    fused_norm: torch.Tensor


class Stage2UnifiedForwardOutput(TypedDict, total=False):
    logits: torch.Tensor
    summary: Stage2UnifiedSummary


class Stage2UnifiedClassifier(nn.Module):
    def __init__(
        self,
        dataset_vocab: dict[str, int],
        output_dims: dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        trunk_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_vocab = {str(name): int(idx) for name, idx in dataset_vocab.items()}
        self.output_dims = {str(name): int(dim) for name, dim in output_dims.items()}

        if set(self.output_dims.keys()) != set(self.dataset_vocab.keys()):
            raise ValueError("output_dims keys must exactly match dataset_vocab keys")
        expected_ids = set(range(len(self.dataset_vocab)))
        actual_ids = set(self.dataset_vocab.values())
        if actual_ids != expected_ids:
            raise ValueError("dataset_vocab ids must be contiguous and exactly match range(len(dataset_vocab))")

        self.image_backbone = MobileViTBackbone(out_dim=hidden_dim)
        self.sequence_backbone = ETBertBackbone(vocab_size=30522, hidden_dim=hidden_dim, max_tokens=128)

        normalized_heads = _normalize_heads_num(hidden_dim=hidden_dim, heads_num=num_heads)
        self.dataset_embed = nn.Embedding(len(self.dataset_vocab), hidden_dim)
        self.image_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=normalized_heads,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )
        self.sequence_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=normalized_heads,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )
        self.image_to_sequence = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=normalized_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sequence_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=normalized_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.shared_trunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=normalized_heads,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, int(trunk_layers)),
        )
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = DatasetConditionedHead(
            hidden_dim=hidden_dim,
            dataset_order=tuple(self.dataset_vocab.keys()),
            output_dims=self.output_dims,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        dataset_name: str,
        return_summary: bool = False,
    ) -> Stage2UnifiedForwardOutput:
        img = self.image_backbone.forward_features(rgb)
        seq = self.sequence_backbone.forward_features(input_ids, attention_mask, token_type_ids)

        img_tokens = self.image_self(img["tokens"])
        seq_tokens = self.sequence_self(seq["tokens"])
        img_cross, _ = self.image_to_sequence(
            img_tokens,
            seq_tokens,
            seq_tokens,
            key_padding_mask=seq["mask"] <= 0,
            need_weights=False,
        )
        seq_cross, _ = self.sequence_to_image(
            seq_tokens,
            img_tokens,
            img_tokens,
            need_weights=False,
        )
        shared_tokens = torch.cat([img_tokens + img_cross, seq_tokens + seq_cross], dim=1)
        fused_tokens = self.shared_trunk(shared_tokens)
        fused = fused_tokens.mean(dim=1)

        dataset_idx = int(self.dataset_vocab[str(dataset_name)])
        dataset_embed = self.dataset_embed(
            torch.full((fused.shape[0],), dataset_idx, dtype=torch.long, device=fused.device)
        )
        conditioned = fused + dataset_embed
        pre_logits = self.pre_classifier(torch.cat([conditioned, img["pooled"], seq["pooled"], fused], dim=1))

        out: Stage2UnifiedForwardOutput = {"logits": self.head(pre_logits, dataset_name=str(dataset_name))}
        if return_summary:
            out["summary"] = {
                "img_pooled_norm": torch.linalg.vector_norm(img["pooled"], dim=1, keepdim=True),
                "seq_pooled_norm": torch.linalg.vector_norm(seq["pooled"], dim=1, keepdim=True),
                "fused_norm": torch.linalg.vector_norm(pre_logits, dim=1, keepdim=True),
            }
        return out
