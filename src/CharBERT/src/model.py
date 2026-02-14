import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CharBERT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float, num_labels: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=vocab_size - 3)  # PAD assumed last of specials
        self.pos_encoder = PositionalEncoding(hidden_size, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
        pooled = x[:, 0]  # CLS 位置
        logits = self.classifier(self.dropout(pooled))
        return logits


def build_model(cfg, num_labels: int):
    return CharBERT(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        num_labels=num_labels,
        max_len=cfg.max_len,
    )

