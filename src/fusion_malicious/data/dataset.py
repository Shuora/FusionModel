from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch


class CachedSessionDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._frame.iloc[index]
        cache_path = Path(row["cache_path"])
        with np.load(cache_path) as arrays:
            image = arrays["image"].astype(np.float32)
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)) / 255.0)
            input_ids = torch.from_numpy(arrays["input_ids"].astype(np.int64))
            attention_mask = torch.from_numpy(arrays["attention_mask"].astype(np.int64))
            label = torch.tensor(int(arrays["label"].item()), dtype=torch.long)
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]
