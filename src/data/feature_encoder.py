from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from src.data.etbert_tokenizer import encode_etbert_tokens


PAD_ID = 0
CLS_ID = 1
SEP_ID = 2


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


def _resize_u8(values: Sequence[int], size: int = 28 * 28) -> np.ndarray:
    base = np.array(values if values else [0], dtype=np.uint8)
    return np.resize(base, size).astype(np.uint8, copy=False)


def _safe_u8(value: float) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return int(value)


def encode_session_rgb(session: Dict[str, Any], image_size: int = 28) -> np.ndarray:
    records = _parse_tls_records(session.get("payload_chunks", []))

    r_values: List[int] = []
    for content_type, version, length, _ in records[:128]:
        r_values.extend(
            [
                _safe_u8(content_type),
                _safe_u8((version >> 8) & 0xFF),
                _safe_u8(version & 0xFF),
                _safe_u8((length >> 8) & 0xFF),
                _safe_u8(length & 0xFF),
            ]
        )

    handshake_count = sum(1 for ct, _, _, _ in records if ct == 22)
    appdata_count = sum(1 for ct, _, _, _ in records if ct == 23)
    cert_count = sum(1 for ct, _, _, payload in records if ct == 22 and payload and payload[0] == 11)
    rec_lengths = [length for _, _, length, _ in records]
    g_values = [
        _safe_u8(handshake_count),
        _safe_u8(appdata_count),
        _safe_u8(cert_count),
        _safe_u8(len(records)),
    ] + [_safe_u8(length % 256) for length in rec_lengths[:256]]

    chunk_lens = [len(c) for c in session.get("payload_chunks", [])]
    duration_ms = _safe_u8((session.get("last_ts", 0.0) - session.get("first_ts", 0.0)) * 1000.0)
    b_features = [
        _safe_u8(session.get("packet_count", 0)),
        _safe_u8(session.get("byte_count", 0) / 32.0),
        duration_ms,
        _safe_u8(float(np.mean(chunk_lens)) if chunk_lens else 0.0),
        _safe_u8(float(np.std(chunk_lens)) if chunk_lens else 0.0),
        _safe_u8(max(chunk_lens) if chunk_lens else 0),
        _safe_u8(min(chunk_lens) if chunk_lens else 0),
    ]
    b_values = b_features + [_safe_u8(x % 256) for x in chunk_lens[:256]]

    size = image_size * image_size
    r = _resize_u8(r_values, size).reshape(image_size, image_size)
    g = _resize_u8(g_values, size).reshape(image_size, image_size)
    b = _resize_u8(b_values, size).reshape(image_size, image_size)
    return np.stack([r, g, b], axis=0).astype(np.uint8, copy=False)


def _hash_token(text: str, vocab_size: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return 3 + (value % max(1, vocab_size - 3))


def encode_tls_tokens(
    session: Dict[str, Any],
    max_len: int = 256,
    vocab_size: int = 8192,
    max_records: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = _parse_tls_records(session.get("payload_chunks", []))
    token_ids: List[int] = [CLS_ID]
    segment_ids: List[int] = [0]

    versions = sorted({version for _, version, _, _ in records})
    for version in versions[:2]:
        token_ids.append(_hash_token(f"VER_{version}", vocab_size))
        segment_ids.append(0)

    for content_type, _, length, _ in records[:max_records]:
        len_bin = int(min(63, math.log2(max(1, length))))
        token_ids.append(_hash_token(f"RT_{content_type}", vocab_size))
        segment_ids.append(1 if content_type == 23 else 0)
        token_ids.append(_hash_token(f"RL_{len_bin}", vocab_size))
        segment_ids.append(1 if content_type == 23 else 0)

    token_ids.append(SEP_ID)
    segment_ids.append(0)

    token_ids = token_ids[:max_len]
    segment_ids = segment_ids[:max_len]
    attention = [1] * len(token_ids)

    pad_count = max_len - len(token_ids)
    if pad_count > 0:
        token_ids.extend([PAD_ID] * pad_count)
        segment_ids.extend([0] * pad_count)
        attention.extend([0] * pad_count)

    return (
        np.asarray(token_ids, dtype=np.int32),
        np.asarray(attention, dtype=np.uint8),
        np.asarray(segment_ids, dtype=np.uint8),
    )


def save_feature_shards(
    sessions: Sequence[Dict[str, Any]],
    family_to_idx: Dict[str, int],
    rgb_path: Path | str,
    seq_path: Path | str,
    token_max_len: int = 256,
    preview_dir: Path | str | None = None,
    preview_per_family: int = 20,
) -> None:
    rgb_path = Path(rgb_path)
    seq_path = Path(seq_path)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    seq_path.parent.mkdir(parents=True, exist_ok=True)

    session_ids: List[str] = []
    labels: List[int] = []
    rgbs: List[np.ndarray] = []
    token_ids_list: List[np.ndarray] = []
    attn_list: List[np.ndarray] = []
    seg_list: List[np.ndarray] = []

    for session in sessions:
        family = str(session.get("family", ""))
        if family not in family_to_idx:
            continue
        session_ids.append(str(session.get("session_id", "")))
        labels.append(int(family_to_idx[family]))
        rgbs.append(encode_session_rgb(session))
        input_ids, attention, token_type_ids = encode_etbert_tokens(session, max_len=token_max_len)
        token_ids_list.append(input_ids)
        attn_list.append(attention)
        seg_list.append(token_type_ids)

    if rgbs:
        rgb_arr = np.stack(rgbs, axis=0).astype(np.uint8, copy=False)
    else:
        rgb_arr = np.zeros((0, 3, 28, 28), dtype=np.uint8)
    if token_ids_list:
        token_arr = np.stack(token_ids_list, axis=0).astype(np.int32, copy=False)
        attn_arr = np.stack(attn_list, axis=0).astype(np.uint8, copy=False)
        seg_arr = np.stack(seg_list, axis=0).astype(np.uint8, copy=False)
    else:
        token_arr = np.zeros((0, token_max_len), dtype=np.int32)
        attn_arr = np.zeros((0, token_max_len), dtype=np.uint8)
        seg_arr = np.zeros((0, token_max_len), dtype=np.uint8)

    sid_arr = np.asarray(session_ids, dtype="U64")
    label_arr = np.asarray(labels, dtype=np.int32)

    np.savez_compressed(
        rgb_path,
        session_id=sid_arr,
        label=label_arr,
        rgb=rgb_arr,
    )
    np.savez_compressed(
        seq_path,
        session_id=sid_arr,
        input_ids=token_arr,
        attention_mask=attn_arr,
        token_type_ids=seg_arr,
    )

    if preview_dir is not None and rgb_arr.shape[0] > 0 and preview_per_family > 0:
        _save_preview_png(
            rgb_arr=rgb_arr,
            session_ids=session_ids,
            labels=labels,
            family_to_idx=family_to_idx,
            preview_dir=Path(preview_dir),
            preview_per_family=preview_per_family,
        )


def _save_preview_png(
    rgb_arr: np.ndarray,
    session_ids: Sequence[str],
    labels: Sequence[int],
    family_to_idx: Dict[str, int],
    preview_dir: Path,
    preview_per_family: int,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    idx_to_family = {idx: family for family, idx in family_to_idx.items()}
    family_counts = _load_existing_preview_counts(preview_dir)

    for i in range(rgb_arr.shape[0]):
        label = int(labels[i])
        family = idx_to_family.get(label, "unknown")
        count = family_counts.get(family, 0)
        if count >= preview_per_family:
            continue
        family_counts[family] = count + 1
        family_dir = preview_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        sid = str(session_ids[i]).replace("/", "_")
        image = np.transpose(rgb_arr[i], (1, 2, 0)).astype(np.uint8, copy=False)
        Image.fromarray(image).save(family_dir / f"{sid}.png")


def _load_existing_preview_counts(preview_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not preview_dir.exists():
        return counts
    for family_dir in preview_dir.iterdir():
        if not family_dir.is_dir():
            continue
        counts[family_dir.name] = len(list(family_dir.glob("*.png")))
    return counts
