from pathlib import Path

import numpy as np

from src.data.feature_encoder import (
    encode_session_rgb,
    save_feature_shards,
)


def _tls_record(content_type: int, payload: bytes, version: int = 0x0303) -> bytes:
    length = len(payload).to_bytes(2, "big")
    return bytes([content_type]) + version.to_bytes(2, "big") + length + payload


def _sample_session():
    return {
        "session_id": "s1",
        "family": "FamilyA",
        "packet_count": 2,
        "byte_count": 50,
        "first_ts": 1.0,
        "last_ts": 2.0,
        "payload_chunks": [
            _tls_record(22, bytes([1]) + b"\x00" * 8),
            _tls_record(23, b"abcd"),
        ],
    }


def test_encode_session_rgb_shape_and_dtype():
    rgb = encode_session_rgb(_sample_session())
    assert rgb.shape == (3, 28, 28)
    assert rgb.dtype == np.uint8
    assert rgb.max() <= 255
    assert rgb.min() >= 0


def test_save_feature_shards_writes_expected_npz_fields(tmp_path: Path):
    sessions = [_sample_session()]
    rgb_path = tmp_path / "rgb" / "rgb_shard_00000.npz"
    etbert_path = tmp_path / "etbert" / "etbert_shard_00000.npz"
    family_to_idx = {"FamilyA": 0}

    save_feature_shards(
        sessions=sessions,
        family_to_idx=family_to_idx,
        rgb_path=rgb_path,
        seq_path=etbert_path,
        token_max_len=32,
    )

    rgb_npz = np.load(rgb_path, allow_pickle=False)
    seq_npz = np.load(etbert_path, allow_pickle=False)

    assert set(rgb_npz.files) == {"session_id", "label", "rgb"}
    assert set(seq_npz.files) == {"session_id", "input_ids", "attention_mask", "token_type_ids"}
    assert rgb_npz["rgb"].shape == (1, 3, 28, 28)
    assert seq_npz["input_ids"].shape == (1, 32)
