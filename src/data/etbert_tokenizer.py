from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _parse_tls_records(payload_chunks: Sequence[bytes]) -> List[Tuple[int, int, int, bytes]]:
    records: List[Tuple[int, int, int, bytes]] = []
    for chunk in payload_chunks:
        i = 0
        while i + 5 <= len(chunk):
            content_type = chunk[i]
            version = int.from_bytes(chunk[i + 1 : i + 3], "big")
            length = int.from_bytes(chunk[i + 3 : i + 5], "big")
            end = i + 5 + length
            if length < 0 or end > len(chunk):
                break
            payload = chunk[i + 5 : end]
            records.append((content_type, version, length, payload))
            i = end
    return records


def _fallback_token_id(token: str, vocab_size: int = 30522) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return 3 + (value % max(1, vocab_size - 3))


def _build_tokens_from_payload_chunks(
    payload_chunks: Sequence[bytes],
    max_records: int = 64,
) -> List[str]:
    records = _parse_tls_records(payload_chunks)
    tokens: List[str] = ["[CLS]"]

    versions = sorted({version for _, version, _, _ in records})
    for version in versions[:2]:
        tokens.append(f"VER_{version:04x}")

    for content_type, _, length, _ in records[:max_records]:
        len_bin = int(min(63, math.log2(max(1, length))))
        tokens.append(f"RT_{content_type}")
        tokens.append(f"RL_{len_bin}")

    tokens.append("[SEP]")
    return tokens


@lru_cache(maxsize=8)
def _load_etbert_vocab_cached(vocab_path: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    path = Path(vocab_path)
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            token = line.rstrip("\r\n")
            if token not in vocab:
                vocab[token] = idx
    return vocab


def load_etbert_vocab(vocab_path: str | Path) -> Dict[str, int]:
    path = Path(vocab_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ET-BERT vocab file not found: {path}")
    return dict(_load_etbert_vocab_cached(str(path.resolve())))


def _resolve_vocab_file_oov_id(vocab: Dict[str, int]) -> int:
    if "[UNK]" in vocab:
        return vocab["[UNK]"]

    pad_id = vocab.get("[PAD]")
    for token in ("[MASK]", "[CLS]", "[SEP]"):
        token_id = vocab.get(token)
        if token_id is not None and token_id != pad_id:
            return token_id

    for token_id in sorted(set(vocab.values())):
        if token_id != pad_id:
            return token_id

    if pad_id is not None:
        return max(1, pad_id + 1)
    return 1


def encode_etbert_tokens(
    session: dict,
    vocab: Dict[str, int] | None = None,
    vocab_path: str | Path | None = None,
    max_len: int = 256,
    max_records: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_vocab_file = vocab is None and vocab_path is not None
    if vocab is None:
        if vocab_path is not None:
            vocab = load_etbert_vocab(vocab_path)
        else:
            vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}

    tokens = _build_tokens_from_payload_chunks(session.get("payload_chunks", []), max_records=max_records)
    vocab_size = max(4, (max(vocab.values()) + 1) if vocab else 0)
    oov_id = _resolve_vocab_file_oov_id(vocab)

    input_ids: List[int] = []
    for token in tokens[:max_len]:
        if token in vocab:
            input_ids.append(vocab[token])
            continue
        if use_vocab_file:
            input_ids.append(oov_id)
            continue
        input_ids.append(_fallback_token_id(token, vocab_size=vocab_size))
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    pad_count = max_len - len(input_ids)
    if pad_count > 0:
        input_ids.extend([vocab.get("[PAD]", 0)] * pad_count)
        attention_mask.extend([0] * pad_count)
        token_type_ids.extend([0] * pad_count)

    return (
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(attention_mask, dtype=np.uint8),
        np.asarray(token_type_ids, dtype=np.uint8),
    )
