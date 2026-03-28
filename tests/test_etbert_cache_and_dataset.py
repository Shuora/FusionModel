from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fusion_malicious.data.cache import write_cached_sample
from fusion_malicious.data.dataset import CachedSessionDataset
from fusion_malicious.data.etbert_tokens import (
    bytes_to_token_text,
    tokenize_session_bytes,
)


def test_bytes_to_token_text_uses_hex_tokens() -> None:
    assert bytes_to_token_text(b"\x0a\xff") == "0a ff"


def test_write_cached_sample_and_dataset_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "sample.npz"
    write_cached_sample(
        cache_path=cache_path,
        image=np.zeros((28, 28, 3), dtype=np.uint8),
        input_ids=np.array([1, 2, 0, 0], dtype=np.int64),
        attention_mask=np.array([1, 1, 0, 0], dtype=np.int64),
        label=5,
    )
    frame = pd.DataFrame([
        {
            "cache_path": str(cache_path),
            "label_id": 5,
        }
    ])
    dataset = CachedSessionDataset(frame)
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 28, 28)
    assert sample["input_ids"].dtype == torch.long
    assert sample["label"].item() == 5


def test_cached_dataset_uses_cached_label_when_label_id_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "missing_label_id.npz"
    write_cached_sample(
        cache_path=cache_path,
        image=np.zeros((28, 28, 3), dtype=np.uint8),
        input_ids=np.array([3, 4, 5, 6], dtype=np.int64),
        attention_mask=np.array([1, 1, 1, 1], dtype=np.int64),
        label=9,
    )
    frame = pd.DataFrame([{"cache_path": str(cache_path)}])
    sample = CachedSessionDataset(frame)[0]
    assert sample["label"].item() == 9


def test_cached_dataset_mismatched_label_raises(tmp_path: Path) -> None:
    cache_path = tmp_path / "mismatch.npz"
    write_cached_sample(
        cache_path=cache_path,
        image=np.zeros((28, 28, 3), dtype=np.uint8),
        input_ids=np.array([7, 8, 9, 0], dtype=np.int64),
        attention_mask=np.array([1, 0, 1, 0], dtype=np.int64),
        label=4,
    )
    frame = pd.DataFrame([
        {
            "cache_path": str(cache_path),
            "label_id": 3,
        }
    ])
    with pytest.raises(ValueError):
        _ = CachedSessionDataset(frame)[0]


def test_write_cached_sample_records_token_text(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokenized.npz"
    text = "aa bb cc"
    write_cached_sample(
        cache_path=cache_path,
        image=np.zeros((28, 28, 3), dtype=np.uint8),
        input_ids=np.zeros(4, dtype=np.int64),
        attention_mask=np.zeros(4, dtype=np.int64),
        label=1,
        token_text=text,
    )
    with np.load(cache_path) as stored:
        assert stored["token_text"].item() == text


def test_cached_dataset_ignores_missing_cached_label_when_label_id_present(tmp_path: Path) -> None:
    cache_path = tmp_path / "label_id_only.npz"
    np.savez(
        cache_path,
        image=np.zeros((28, 28, 3), dtype=np.uint8),
        input_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        attention_mask=np.array([1, 1, 1, 1], dtype=np.int64),
    )
    frame = pd.DataFrame(
        [
            {
                "cache_path": str(cache_path),
                "label_id": 11,
            }
        ]
    )
    sample = CachedSessionDataset(frame)[0]
    assert sample["label"].item() == 11


def test_tokenize_session_bytes_calls_tokenizer() -> None:
    class DummyTokenizer:
        model_max_length = 4

        def __call__(self, text, padding, truncation, max_length, return_attention_mask):
            assert padding == "max_length"
            assert truncation is True
            assert max_length == 4
            return {
                "input_ids": [10, 20, 30, 40],
                "attention_mask": [1, 1, 1, 1],
            }

    dummy = DummyTokenizer()
    raw = b"\x01\x02"
    text, input_ids, attention_mask = tokenize_session_bytes(
        raw_bytes=raw,
        tokenizer=dummy,
        max_length=4,
    )
    assert text == "01 02"
    assert input_ids.tolist() == [10, 20, 30, 40]
    assert attention_mask.tolist() == [1, 1, 1, 1]
