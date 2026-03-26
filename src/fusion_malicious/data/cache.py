from __future__ import annotations

from pathlib import Path

import numpy as np


def write_cached_sample(
    cache_path: Path,
    image: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    label: int,
) -> None:
    """Write a compressed cache file containing the sample tensors."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        label=np.array(label, dtype=np.int64),
    )
