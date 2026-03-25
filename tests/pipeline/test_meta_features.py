from __future__ import annotations

import torch


def _dummy_level1_output() -> dict:
    return {
        "logits_img": torch.tensor(
            [[2.1, 0.3, -0.4], [0.5, 1.6, -0.2], [1.2, -0.1, 0.4]],
            dtype=torch.float32,
        ),
        "logits_tls": torch.tensor(
            [[1.1, 0.4, -0.1], [0.2, 1.0, 0.6], [0.9, 0.1, -0.2]],
            dtype=torch.float32,
        ),
        "logits_fuse": torch.tensor(
            [[1.7, 0.2, -0.5], [0.4, 1.3, 0.2], [1.1, 0.0, -0.3]],
            dtype=torch.float32,
        ),
        "summary": {
            "img_pooled_norm": torch.tensor([[0.8], [0.7], [0.9]], dtype=torch.float32),
            "txt_pooled_norm": torch.tensor([[0.6], [0.4], [0.5]], dtype=torch.float32),
            "fused_norm": torch.tensor([[0.7], [0.8], [0.85]], dtype=torch.float32),
        },
    }


def test_shared_meta_helper_builds_deterministic_blocks():
    try:
        from src.pipeline.meta_features import build_meta_feature_blocks
    except ImportError as e:  # pragma: no cover
        raise AssertionError("missing shared stage2 meta feature helper module") from e

    blocks = build_meta_feature_blocks(_dummy_level1_output())
    for key in ["logits", "confidence", "agreement", "summary"]:
        assert key in blocks
    for key in ["img", "tls", "fuse"]:
        assert key in blocks["logits"]
    assert blocks["logits"]["img"].shape == (3, 3)
    assert blocks["logits"]["tls"].shape == (3, 3)
    assert blocks["logits"]["fuse"].shape == (3, 3)
    for key in ["entropy", "max_prob"]:
        assert key in blocks["confidence"]
    assert blocks["confidence"]["entropy"].shape == (3, 3)
    assert blocks["confidence"]["max_prob"].shape == (3, 3)
    assert blocks["agreement"].shape == (3, 1)
    assert blocks["summary"].shape == (3, 3)


def test_shared_meta_helper_exports_stable_flattened_schema():
    try:
        from src.pipeline.meta_features import flatten_meta_feature_blocks
    except ImportError as e:  # pragma: no cover
        raise AssertionError("missing shared stage2 meta feature flatten helper") from e

    dummy = _dummy_level1_output()
    num_samples = dummy["logits_img"].shape[0]
    num_classes = dummy["logits_img"].shape[1]
    summary_dim = len(dummy["summary"])

    expected_dim = (3 * num_classes) + (2 * num_classes) + 1 + summary_dim

    x, feature_names, schema = flatten_meta_feature_blocks(dummy)
    assert x.shape == (num_samples, expected_dim)
    assert len(feature_names) == expected_dim
    assert schema["version"] == "stage2_meta_v1"
    assert schema["dim"] == expected_dim
    assert schema["feature_names"] == feature_names
