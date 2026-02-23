from __future__ import annotations

import numpy as np


def encode_tls_rgb(sample: dict, image_size: int = 28) -> np.ndarray:
    """Encode TLS-side metadata into a fixed RGB tensor (H, W, C)."""
    _ = sample
    return np.zeros((image_size, image_size, 3), dtype=np.float32)
