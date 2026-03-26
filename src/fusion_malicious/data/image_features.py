from __future__ import annotations

import numpy as np

from fusion_malicious.data.session_bytes import normalize_session_bytes


def _rolling_entropy(values: np.ndarray, window: int = 8) -> np.ndarray:
    entropy = np.zeros_like(values, dtype=np.float32)
    for index in range(values.size):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        hist = np.bincount(chunk, minlength=256).astype(np.float32)
        hist /= hist.sum()
        hist = hist[hist > 0]
        entropy[index] = float(-(hist * np.log2(hist)).sum())
    entropy *= 255.0 / max(entropy.max(), 1.0)
    return entropy.astype(np.uint8)


def bytes_to_rgb_image(raw_bytes: bytes, size: int = 784) -> np.ndarray:
    if size != 784:
        raise ValueError("bytes_to_rgb_image only supports size=784 for 28x28 reshaping")
    base = normalize_session_bytes(raw_bytes, size=size)
    base_int = base.astype(np.int16)
    diff = np.abs(np.diff(base_int, prepend=base_int[:1])).astype(np.uint8)
    entropy = _rolling_entropy(base)
    stacked = np.stack([base, diff, entropy], axis=-1)
    return stacked.reshape(28, 28, 3)
