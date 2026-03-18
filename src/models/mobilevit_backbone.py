from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import MobileViTConfig, MobileViTForImageClassification


DEFAULT_MOBILEVIT_CHECKPOINT = Path("/tmp/Shuora-MobileViT/malicious_traffic_mobilevit_model.pth")


class MobileViTBackbone(nn.Module):
    def __init__(
        self,
        out_dim: int = 128,
        checkpoint_path: str | Path | None = DEFAULT_MOBILEVIT_CHECKPOINT,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        config = MobileViTConfig(num_labels=num_labels)
        self.model = MobileViTForImageClassification(config)
        self.backbone_dim = int(self.model.classifier.in_features)
        self._load_checkpoint_if_available(checkpoint_path)
        self.proj = nn.Linear(self.backbone_dim, out_dim)

    def _load_checkpoint_if_available(self, checkpoint_path: str | Path | None) -> None:
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            return
        filtered = {k: v for k, v in state.items() if not str(k).startswith("classifier.")}
        self.model.load_state_dict(filtered, strict=False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        restore_training = False
        if self.training and rgb.shape[0] == 1:
            restore_training = True
            self.model.mobilevit.eval()
        outputs = self.model.mobilevit(pixel_values=rgb)
        if restore_training:
            self.model.mobilevit.train()
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=(-1, -2))
        return self.proj(pooled)
