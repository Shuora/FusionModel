from __future__ import annotations

from pathlib import Path
from typing import Dict

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
        self.token_stage_dims = list(self.model.config.neck_hidden_sizes[-4:-1])
        self._load_checkpoint_if_available(checkpoint_path)
        self.token_proj = nn.ModuleList(nn.Linear(dim, out_dim) for dim in self.token_stage_dims)
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

    def _forward_mobilevit(self, rgb: torch.Tensor, output_hidden_states: bool = False):
        restore_training = False
        if self.training and rgb.shape[0] == 1:
            restore_training = True
            self.model.mobilevit.eval()
        outputs = self.model.mobilevit(pixel_values=rgb, output_hidden_states=output_hidden_states)
        if restore_training:
            self.model.mobilevit.train()
        return outputs

    def forward_features(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self._forward_mobilevit(rgb, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states or ())
        token_maps = hidden_states[-len(self.token_proj) :] if hidden_states else [outputs.last_hidden_state]
        token_chunks = []
        for fmap, proj in zip(token_maps, self.token_proj):
            tokens = fmap.flatten(start_dim=2).transpose(1, 2)
            token_chunks.append(proj(tokens))

        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=(-1, -2))
        pooled = self.proj(pooled)
        tokens = torch.cat(token_chunks, dim=1) if token_chunks else pooled.unsqueeze(1)
        return {"tokens": tokens, "pooled": pooled}

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.forward_features(rgb)["pooled"]
