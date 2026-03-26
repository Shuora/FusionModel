from __future__ import annotations


def bytes_to_token_text(raw_bytes: bytes) -> str:
    """Return lowercase hex tokens separated by spaces."""
    return " ".join(f"{byte:02x}" for byte in raw_bytes)
