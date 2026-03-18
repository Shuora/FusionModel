from pathlib import Path

import numpy as np

from src.data.etbert_tokenizer import encode_etbert_tokens
from src.data.feature_encoder import save_feature_shards


def _tls_record(content_type: int, payload: bytes, version: int = 0x0303) -> bytes:
    length = len(payload).to_bytes(2, "big")
    return bytes([content_type]) + version.to_bytes(2, "big") + length + payload


def _sample_session() -> dict:
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


def test_save_feature_shards_writes_etbert_inputs(tmp_path: Path):
    rgb_path = tmp_path / "rgb" / "rgb_shard_00000.npz"
    seq_path = tmp_path / "etbert" / "etbert_shard_00000.npz"

    save_feature_shards(
        sessions=[_sample_session()],
        family_to_idx={"FamilyA": 0},
        rgb_path=rgb_path,
        seq_path=seq_path,
        token_max_len=128,
    )

    seq_npz = np.load(seq_path, allow_pickle=False)

    assert {"session_id", "input_ids", "attention_mask", "token_type_ids"} <= set(seq_npz.files)
    assert seq_npz["input_ids"].shape == (1, 128)


def test_encode_etbert_tokens_returns_padded_triplet():
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "VER_0303": 3,
        "RT_22": 4,
        "RT_23": 5,
        "RL_3": 6,
        "RL_2": 7,
    }

    input_ids, attention_mask, token_type_ids = encode_etbert_tokens(
        _sample_session(),
        vocab=vocab,
        max_len=16,
    )

    assert input_ids.shape == (16,)
    assert attention_mask.shape == (16,)
    assert token_type_ids.shape == (16,)
    assert input_ids.dtype == np.int32
    assert attention_mask.dtype == np.uint8
    assert token_type_ids.dtype == np.uint8
    assert int(attention_mask.sum()) > 2


def test_encode_etbert_tokens_supports_etbert_vocab_file(tmp_path: Path):
    vocab_path = tmp_path / "encryptd_vocab.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "[PAD]",
                "[SEP]",
                "[CLS]",
                "[UNK]",
                "[MASK]",
                "VER_0303",
                "RT_22",
                "RL_3",
                "RT_23",
                "RL_2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    input_ids, attention_mask, token_type_ids = encode_etbert_tokens(
        _sample_session(),
        vocab_path=vocab_path,
        max_len=12,
    )

    assert input_ids.shape == (12,)
    assert attention_mask.shape == (12,)
    assert token_type_ids.shape == (12,)
    assert input_ids[:7].tolist() == [2, 5, 6, 7, 8, 9, 1]
    assert input_ids[7:].tolist() == [0, 0, 0, 0, 0]
    assert int(attention_mask.sum()) == 7
    assert token_type_ids.tolist() == [0] * 12


def test_encode_etbert_tokens_vocab_path_oov_uses_unk_id(tmp_path: Path):
    vocab_path = tmp_path / "encryptd_vocab.txt"
    vocab_path.write_text(
        "\n".join(["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]) + "\n",
        encoding="utf-8",
    )

    input_ids, attention_mask, token_type_ids = encode_etbert_tokens(
        _sample_session(),
        vocab_path=vocab_path,
        max_len=10,
    )

    assert input_ids.shape == (10,)
    assert attention_mask.shape == (10,)
    assert token_type_ids.shape == (10,)
    assert input_ids[:7].tolist() == [2, 3, 3, 3, 3, 3, 1]
    assert input_ids[7:].tolist() == [0, 0, 0]


def test_encode_etbert_tokens_vocab_path_without_unk_uses_non_pad_fallback(tmp_path: Path):
    vocab_path = tmp_path / "encryptd_vocab.txt"
    vocab_path.write_text(
        "\n".join(["[PAD]", "[SEP]", "[CLS]", "[MASK]"]) + "\n",
        encoding="utf-8",
    )

    input_ids, attention_mask, token_type_ids = encode_etbert_tokens(
        _sample_session(),
        vocab_path=vocab_path,
        max_len=10,
    )

    assert input_ids.shape == (10,)
    assert attention_mask.shape == (10,)
    assert token_type_ids.shape == (10,)
    assert input_ids[:7].tolist() == [2, 3, 3, 3, 3, 3, 1]
    assert input_ids[7:].tolist() == [0, 0, 0]
