from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_pcap_bytes(path: str, chunk_size: int = 4096) -> Iterator[bytes]:
    """Yield raw bytes chunks from a pcap file path."""
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
