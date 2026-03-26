from typing import Optional

import torch
from torch import nn

from fusion_malicious.models.image_encoder import ImageEncoder
from fusion_malicious.models.multimodal import MultimodalClassifier
from fusion_malicious.models.text_encoder import TextEncoder


class FakeImageBackbone(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.size(0)
        return torch.ones(batch, 16, 32)


class FakeTextBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = input_ids.size(0)
        return torch.ones(batch, 12, 32)


class DictOutputTextBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = input_ids.size(0)
        tokens = torch.ones(batch, 8, 32)
        return {"last_hidden_state": tokens}


class FourDimImageBackbone(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.size(0)
        tokens = torch.arange(batch * 3 * 4 * 4, dtype=torch.float32).reshape(batch, 3, 4, 4)
        return tokens


class IdentityCrossAttention(nn.Module):
    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return image_tokens, text_tokens


class TextOnlyFusion(nn.Module):
    def forward(self, image_summary: torch.Tensor, text_summary: torch.Tensor) -> torch.Tensor:
        return text_summary


class SequenceTextBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.size()
        values = torch.arange(seq, dtype=torch.float32, device=input_ids.device).reshape(1, seq, 1) + 1
        return values.expand(batch, -1, 1)


def test_multimodal_classifier_returns_logits() -> None:
    model = MultimodalClassifier(
        image_backbone=FakeImageBackbone(),
        text_backbone=FakeTextBackbone(),
        hidden_dim=32,
        num_classes=7,
        num_heads=4,
    )
    logits = model(
        image=torch.randn(4, 3, 112, 112),
        input_ids=torch.ones(4, 32, dtype=torch.long),
        attention_mask=torch.ones(4, 32, dtype=torch.long),
    )
    assert logits.shape == (4, 7)


def test_text_encoder_prefers_last_hidden_state() -> None:
    encoder = TextEncoder(backbone=DictOutputTextBackbone())
    output = encoder(
        input_ids=torch.ones(2, 16, dtype=torch.long),
        attention_mask=torch.ones(2, 16, dtype=torch.long),
    )
    assert output.shape == (2, 8, 32)


def test_image_encoder_flattens_spatial_tokens() -> None:
    encoder = ImageEncoder(backbone=FourDimImageBackbone())
    output = encoder(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 16, 3)


def test_multimodal_masked_pooling_respects_attention_mask() -> None:
    model = MultimodalClassifier(
        image_backbone=FakeImageBackbone(),
        text_backbone=SequenceTextBackbone(),
        hidden_dim=1,
        num_classes=1,
        num_heads=1,
    )
    model.cross_attention = IdentityCrossAttention()
    model.gated_fusion = TextOnlyFusion()
    model.classifier = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.classifier.weight.fill_(1.0)

    batch = 2
    seq = 4
    attention_mask = torch.zeros(batch, seq, dtype=torch.long)
    attention_mask[:, 0] = 1

    logits = model(
        image=torch.randn(batch, 3, 112, 112),
        input_ids=torch.ones(batch, seq, dtype=torch.long),
        attention_mask=attention_mask,
    )
    assert torch.allclose(logits, torch.ones(batch, 1))
