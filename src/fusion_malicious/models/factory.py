from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel
else:
    PretrainedConfig = Any  # type: ignore[assignment]
    PreTrainedModel = Any  # type: ignore[assignment]


class _TimmFeatureBackbone(nn.Module):
    def __init__(self, model: nn.Module, feature_index: int | None = None) -> None:
        super().__init__()
        self.model = model
        self.feature_index = feature_index

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.model(image)

        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("Timm feature backbone returned an empty sequence")
            index = self.feature_index if self.feature_index is not None else -1
            features = features[index]

        if not torch.is_tensor(features):
            raise TypeError("Timm feature backbone must return tensors")

        return features


class _TimmFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(image)
        else:
            features = self.model(image)

        if not torch.is_tensor(features):
            raise TypeError("Timm backbone must return tensors")

        return features


def _import_timm_module() -> ModuleType:
    try:
        return importlib.import_module("timm")
    except ImportError as exc:  # pragma: no cover - rare in dependent env
        raise ImportError("timm is required to build image backbones; install it before calling build_image_backbone") from exc


def _import_transformers_module() -> ModuleType:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - rare in dependent env
        raise ImportError("transformers is required to build text backbones; install it before calling build_text_backbone") from exc


def build_image_backbone(
    architecture: str,
    *,
    pretrained: bool = True,
    features_only: bool = True,
    feature_index: int | None = None,
    **timm_kwargs: Any,
) -> nn.Module:
    timm = _import_timm_module()
    backbone = timm.create_model(
        architecture,
        pretrained=pretrained,
        features_only=features_only,
        **timm_kwargs,
    )

    if features_only:
        return _TimmFeatureBackbone(backbone, feature_index=feature_index)

    return _TimmFeatureExtractor(backbone)


def build_text_backbone(
    model_name: str,
    *,
    pretrained: bool = True,
    config: PretrainedConfig | None = None,
    trust_remote_code: bool = False,
    config_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> PreTrainedModel:
    transformers = _import_transformers_module()
    AutoConfig = getattr(transformers, "AutoConfig")
    AutoModel = getattr(transformers, "AutoModel")

    config_kwargs = dict(config_kwargs or {})
    model_kwargs = dict(model_kwargs or {})

    if config is None:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )

    config.return_dict = True

    if pretrained:
        return AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

    return AutoModel.from_config(
        config,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )


__all__ = [
    "build_image_backbone",
    "build_text_backbone",
]
