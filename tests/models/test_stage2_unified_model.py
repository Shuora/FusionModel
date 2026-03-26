from __future__ import annotations

import pytest
import torch

from src.models.stage2_unified_model import DatasetConditionedHead, Stage2UnifiedClassifier
from src.stage2_registry import DATASET_NAME_TO_ID, STAGE2_DATASET_ORDER, dataset_num_classes


def test_dataset_conditioned_head_uses_dataset_specific_output_dim():
    head = DatasetConditionedHead(
        hidden_dim=16,
        dataset_order=STAGE2_DATASET_ORDER,
        output_dims={name: dataset_num_classes(name) for name in STAGE2_DATASET_ORDER},
    )
    x = torch.randn(4, 16)
    logits = head(x, dataset_name="MFCP")
    assert logits.shape == (4, 6)


def test_stage2_unified_classifier_returns_dataset_specific_logits_and_summary():
    model = Stage2UnifiedClassifier(
        dataset_vocab=DATASET_NAME_TO_ID,
        output_dims={"MTA": 7, "MFCP": 6, "USTC-TFC2016": 10},
        hidden_dim=32,
        num_heads=4,
        trunk_layers=2,
        dropout=0.1,
    )
    out = model(
        rgb=torch.randn(2, 3, 28, 28),
        input_ids=torch.randint(0, 128, (2, 128)),
        attention_mask=torch.ones(2, 128, dtype=torch.long),
        token_type_ids=torch.zeros(2, 128, dtype=torch.long),
        dataset_name="MTA",
        return_summary=True,
    )
    assert out["logits"].shape == (2, 7)
    assert isinstance(out["summary"], dict)
    assert set(out["summary"].keys()) >= {"img_pooled_norm", "seq_pooled_norm", "fused_norm"}
    assert out["summary"]["img_pooled_norm"].shape == (2, 1)
    assert out["summary"]["seq_pooled_norm"].shape == (2, 1)
    assert out["summary"]["fused_norm"].shape == (2, 1)


def test_stage2_unified_classifier_rejects_non_contiguous_dataset_vocab_ids():
    with pytest.raises(ValueError, match="dataset_vocab ids must be contiguous"):
        Stage2UnifiedClassifier(
            dataset_vocab={"MTA": 0, "MFCP": 2, "USTC-TFC2016": 3},
            output_dims={"MTA": 7, "MFCP": 6, "USTC-TFC2016": 10},
            hidden_dim=16,
            num_heads=2,
            trunk_layers=1,
            dropout=0.1,
        )


def test_stage2_unified_classifier_rejects_output_dims_key_mismatch():
    with pytest.raises(ValueError, match="output_dims keys must exactly match dataset_vocab keys"):
        Stage2UnifiedClassifier(
            dataset_vocab=DATASET_NAME_TO_ID,
            output_dims={"MTA": 7, "MFCP": 6},
            hidden_dim=16,
            num_heads=2,
            trunk_layers=1,
            dropout=0.1,
        )
