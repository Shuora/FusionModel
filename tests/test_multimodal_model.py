import torch
from torch import nn

from fusion_malicious.models.multimodal import MultimodalClassifier


class FakeImageBackbone(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.size(0)
        return torch.ones(batch, 16, 32)


class FakeTextBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = input_ids.size(0)
        return torch.ones(batch, 12, 32)


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
