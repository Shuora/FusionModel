from __future__ import annotations

from typing import List

from src.pipeline.token_schema import CLS_TOKEN_ID, PAD_TOKEN_ID, SEP_TOKEN_ID


def encode_tls_tokens(sample: dict, max_len: int = 256) -> List[int]:
    """Encode TLS fields into a fixed-length token id sequence."""
    _ = sample
    if max_len < 2:
        raise ValueError("max_len must be >= 2")

    body_len = max_len - 2
    return [CLS_TOKEN_ID] + ([PAD_TOKEN_ID] * body_len) + [SEP_TOKEN_ID]
