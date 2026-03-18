from __future__ import annotations

import torch

from src.models.fusion_model import MobileViTETBertFusionClassifier


def test_mobilevit_etbert_fusion_model_forward_shapes_and_gate_range():
    model = MobileViTETBertFusionClassifier(num_classes=3, hidden_dim=64, vocab_size=4096, max_tokens=128)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 4096, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    token_type_ids = torch.zeros(2, 128, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids)

    assert out["logits_fuse"].shape == (2, 3)
    assert out["logits_img"].shape == (2, 3)
    assert out["logits_tls"].shape == (2, 3)
    assert out["gate"].shape == (2, 1)
    assert torch.all(out["gate"] >= 0.0)
    assert torch.all(out["gate"] <= 1.0)


def test_mobilevit_etbert_fusion_model_handles_partial_attention_mask():
    model = MobileViTETBertFusionClassifier(num_classes=2, hidden_dim=32, vocab_size=2048, max_tokens=64)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 2048, (2, 64))
    attention_mask = torch.zeros(2, 64, dtype=torch.long)
    attention_mask[:, :8] = 1
    token_type_ids = torch.zeros(2, 64, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids)
    assert out["logits_fuse"].shape == (2, 2)
