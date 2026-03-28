from __future__ import annotations

from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def bytes_to_token_text(raw_bytes: bytes) -> str:
    """Return lowercase hex tokens separated by spaces."""
    return " ".join(f"{byte:02x}" for byte in raw_bytes)


def load_etbert_tokenizer(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
    use_fast: bool = False,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer used to build ET-BERT-compatible token ids."""
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )


def tokenize_session_bytes(
    raw_bytes: bytes,
    *,
    tokenizer: Any,
    max_length: int | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Tokenize normalized byte text into ids and attention mask."""
    token_text = bytes_to_token_text(raw_bytes)
    target_length = max_length or getattr(tokenizer, "model_max_length", 512)
    encoded = tokenizer(
        token_text,
        padding="max_length",
        truncation=True,
        max_length=target_length,
        return_attention_mask=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
    return token_text, input_ids, attention_mask
