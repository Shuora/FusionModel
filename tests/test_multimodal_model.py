from typing import Optional

import torch
from torch import nn
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fusion_malicious.models.factory import build_image_backbone, build_text_backbone
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


class HighDimImageBackbone(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.size(0)
        return torch.ones(batch, 6, self.feature_dim)


class HighDimTextBackbone(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.size()
        return torch.ones(batch, seq, self.feature_dim)


class FakeTimmBackbone(nn.Module):
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = image.size(0)
        return torch.zeros(batch, 2, 2), torch.ones(batch, 2, 2)


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


def test_image_encoder_projects_to_hidden_dim() -> None:
    encoder = ImageEncoder(backbone=HighDimImageBackbone(feature_dim=64), output_dim=8)
    output = encoder(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 6, 8)


def test_text_encoder_projects_to_hidden_dim() -> None:
    encoder = TextEncoder(backbone=HighDimTextBackbone(feature_dim=48), output_dim=6)
    output = encoder(
        input_ids=torch.ones(2, 12, dtype=torch.long),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
    )
    assert output.shape == (2, 12, 6)


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
    model.text_encoder.projection = nn.Identity()
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


def test_multimodal_classifier_handles_mismatched_branch_dims() -> None:
    model = MultimodalClassifier(
        image_backbone=HighDimImageBackbone(feature_dim=48),
        text_backbone=HighDimTextBackbone(feature_dim=96),
        hidden_dim=32,
        num_classes=3,
        num_heads=4,
    )
    logits = model(
        image=torch.randn(2, 3, 112, 112),
        input_ids=torch.ones(2, 11, dtype=torch.long),
        attention_mask=torch.ones(2, 11, dtype=torch.long),
    )
    assert logits.shape == (2, 3)


def test_build_image_backbone_selects_feature_index() -> None:
    create_args: dict[str, object] = {}

    def fake_create_model(architecture: str, pretrained: bool, features_only: bool, **kwargs: object) -> nn.Module:
        create_args["architecture"] = architecture
        create_args["pretrained"] = pretrained
        create_args["features_only"] = features_only
        create_args["kwargs"] = kwargs
        return FakeTimmBackbone()

    fake_timm = SimpleNamespace(create_model=fake_create_model)
    with patch("fusion_malicious.models.factory._import_timm_module", return_value=fake_timm):
        backbone = build_image_backbone("mobilevit_small", pretrained=False, feature_index=0)
        output = backbone(torch.zeros(1, 3, 32, 32))
    assert output.shape == (1, 2, 2)
    assert torch.allclose(output, torch.zeros(1, 2, 2))
    assert create_args == {
        "architecture": "mobilevit_small",
        "pretrained": False,
        "features_only": True,
        "kwargs": {},
    }


def test_build_text_backbone_respects_pretrained_flag() -> None:
    config = MagicMock()
    config.return_dict = False
    pretrained_model = MagicMock()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            return config

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            return pretrained_model

        @staticmethod
        def from_config(*args: object, **kwargs: object) -> MagicMock:
            raise AssertionError("should not build from config when pretrained")

    fake_transformers = SimpleNamespace(AutoConfig=FakeAutoConfig, AutoModel=FakeAutoModel)
    with patch("fusion_malicious.models.factory._import_transformers_module", return_value=fake_transformers):
        result = build_text_backbone("et-bert-small", pretrained=True, trust_remote_code=True)
    assert result is pretrained_model
    assert config.return_dict is True


def test_build_text_backbone_builds_from_config_when_requested() -> None:
    config = MagicMock()
    config.return_dict = False
    config_model = MagicMock()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            return config

    class FakeAutoModel:
        from_config_kwargs: Optional[dict[str, object]] = None

        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            raise AssertionError("should not call from_pretrained when pretrained=False")

        @staticmethod
        def from_config(cfg: MagicMock, *args: object, **kwargs: object) -> MagicMock:
            FakeAutoModel.from_config_kwargs = kwargs
            return config_model

    fake_transformers = SimpleNamespace(AutoConfig=FakeAutoConfig, AutoModel=FakeAutoModel)
    with patch("fusion_malicious.models.factory._import_transformers_module", return_value=fake_transformers):
        result = build_text_backbone(
            "et-bert-small",
            pretrained=False,
            trust_remote_code=True,
            model_kwargs={"extra": 1},
        )

    assert result is config_model
    assert config.return_dict is True
    assert FakeAutoModel.from_config_kwargs == {"trust_remote_code": True, "extra": 1}
