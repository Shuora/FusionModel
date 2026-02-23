from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyFusionDataset(Dataset):
    """Tiny synthetic dataset for smoke training and wiring checks."""

    def __init__(self, size: int = 16, num_classes: int = 2) -> None:
        self.size = size
        self.num_classes = max(1, num_classes)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        label = idx % self.num_classes
        return {
            "image": torch.zeros(3, 28, 28, dtype=torch.float32),
            "token_ids": torch.zeros(32, dtype=torch.long),
            "attn_mask": torch.ones(32, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
