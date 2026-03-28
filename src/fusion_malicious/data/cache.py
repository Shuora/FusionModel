from __future__ import annotations

from pathlib import Path

import numpy as np


def write_cached_sample(
    cache_path: Path,
    image: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    label: int,
    token_text: str | None = None,
) -> None:
    """Write a compressed cache file containing the sample tensors."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "image": image,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": np.array(label, dtype=np.int64),
    }
    if token_text is not None:
        payload["token_text"] = np.array(token_text)
    np.savez_compressed(cache_path, **payload)
