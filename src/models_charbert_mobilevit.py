"""Native CharBERT + MobileViT attention fusion models."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = [
            nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        *,
        depth: int = 2,
        patch_size: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.local_conv = ConvBNAct(in_channels, in_channels, k=3, s=1)
        self.local_proj = ConvBNAct(in_channels, transformer_dim, k=1, s=1, p=0)

        patch_dim = transformer_dim * self.patch_size * self.patch_size
        self.token_embed = nn.Linear(patch_dim, transformer_dim)
        self.token_unembed = nn.Linear(transformer_dim, patch_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=transformer_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.fusion_proj = ConvBNAct(transformer_dim, in_channels, k=1, s=1, p=0)
        self.fusion = ConvBNAct(in_channels * 2, in_channels, k=3, s=1)

    def _pad_to_patch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        ph = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pw = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
        return x, ph, pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        y = self.local_conv(x)
        y = self.local_proj(y)

        y, ph, pw = self._pad_to_patch(y)
        b, c, h, w = y.shape

        patches = F.unfold(y, kernel_size=self.patch_size, stride=self.patch_size)
        # (B, C*P*P, N) -> (B, N, C*P*P)
        tokens = patches.transpose(1, 2)
        tokens = self.token_embed(tokens)
        tokens = self.transformer(tokens)
        tokens = self.token_unembed(tokens)

        # (B, N, C*P*P) -> (B, C*P*P, N)
        patches_rec = tokens.transpose(1, 2)
        y = F.fold(
            patches_rec,
            output_size=(h, w),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        if ph > 0 or pw > 0:
            y = y[:, :, : h - ph, : w - pw]

        y = self.fusion_proj(y)
        y = torch.cat([shortcut, y], dim=1)
        y = self.fusion(y)
        return y


class MobileViTEncoder(nn.Module):
    """Small-input (28x28) native MobileViT-style encoder."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = ConvBNAct(3, 16, k=3, s=2)  # 28 -> 14
        self.ir1 = InvertedResidual(16, 24, stride=1, expand_ratio=2)
        self.mvit1 = MobileViTBlock(24, transformer_dim=48, depth=2, patch_size=2, n_heads=4)

        self.down = ConvBNAct(24, 48, k=3, s=2)  # 14 -> 7
        self.ir2 = InvertedResidual(48, 48, stride=1, expand_ratio=2)
        self.mvit2 = MobileViTBlock(48, transformer_dim=64, depth=2, patch_size=2, n_heads=4)

        self.head = ConvBNAct(48, out_dim, k=1, s=1, p=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.ir1(x)
        x = self.mvit1(x)
        x = self.down(x)
        x = self.ir2(x)
        x = self.mvit2(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CharBERTEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 259,
        pad_id: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 786,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.hidden_size = int(hidden_size)
        self.feature_dim = int(feature_dim)

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=self.pad_id)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, feature_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        pad_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        pooled = x[:, 0]  # CLS
        feat = self.proj(self.dropout(pooled))

        if return_sequence:
            return feat, x, pad_mask
        return feat


class AttentionFusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        attention_dim: int = 128,
        char_seq_len: int = 786,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.attention_dim = int(attention_dim)

        self.image_encoder = MobileViTEncoder(out_dim=128)
        self.text_encoder = CharBERTEncoder(max_len=char_seq_len, feature_dim=128, hidden_size=128)

        self.q_proj = nn.Linear(128, attention_dim)
        self.k_proj = nn.Linear(self.text_encoder.hidden_size, attention_dim)
        self.v_proj = nn.Linear(self.text_encoder.hidden_size, attention_dim)

        self.image_head = nn.Linear(128, num_classes)
        self.text_head = nn.Linear(128, num_classes)

        self.fused_head = nn.Sequential(
            nn.Linear(128 + attention_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _encode(self, images: torch.Tensor, input_ids: torch.Tensor):
        img_feat = self.image_encoder(images)
        txt_feat, txt_seq, pad_mask = self.text_encoder(input_ids, return_sequence=True)
        return img_feat, txt_feat, txt_seq, pad_mask

    def _cross_attention(self, img_feat: torch.Tensor, txt_seq: torch.Tensor, pad_mask: torch.Tensor):
        q = self.q_proj(img_feat).unsqueeze(1)  # (B,1,D)
        k = self.k_proj(txt_seq)                # (B,S,D)
        v = self.v_proj(txt_seq)                # (B,S,D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.attention_dim))
        scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        ctx = torch.matmul(attn, v).squeeze(1)
        return ctx, attn.squeeze(1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, *, return_attention: bool = False):
        img_feat, txt_feat, txt_seq, pad_mask = self._encode(images, input_ids)
        ctx, attn = self._cross_attention(img_feat, txt_seq, pad_mask)

        fused = torch.cat([img_feat, ctx], dim=1)
        logits = self.fused_head(fused)

        if return_attention:
            return logits, attn
        return logits

    def branch_logits(self, images: torch.Tensor, input_ids: torch.Tensor):
        img_feat, txt_feat, txt_seq, pad_mask = self._encode(images, input_ids)
        ctx, _ = self._cross_attention(img_feat, txt_seq, pad_mask)
        fused = torch.cat([img_feat, ctx], dim=1)

        image_logits = self.image_head(img_feat)
        text_logits = self.text_head(txt_feat)
        fused_logits = self.fused_head(fused)
        return image_logits, text_logits, fused_logits
