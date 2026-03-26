from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fusion_malicious.data.cache import write_cached_sample
from fusion_malicious.data.dataset import CachedSessionDataset
from fusion_malicious.data.etbert_tokens import bytes_to_token_text


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
    # Provide no label metadata (the DataFrame deliberately omits label_id)
    frame = pd.DataFrame([{"cache_path": str(cache_path)}])
    dataset = CachedSessionDataset(frame)
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 28, 28)
    assert sample["input_ids"].dtype == torch.long
    assert sample["label"].item() == 5
