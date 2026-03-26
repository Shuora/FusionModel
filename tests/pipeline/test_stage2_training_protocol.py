from __future__ import annotations

import math

import pytest
import src.train as train_module
import src.stage2_trainer as stage2_trainer_module
from src.stage2_registry import ACCEPTANCE_GATES, STAGE2_DATASET_ORDER


def test_round_robin_dataset_batch_sampler_cycles_dataset_names_evenly():
    from src.stage2_trainer import RoundRobinDatasetBatchSampler

    sampler = RoundRobinDatasetBatchSampler(
        dataset_to_indices={
            "MTA": list(range(0, 4)),
            "MFCP": list(range(100, 104)),
            "USTC-TFC2016": list(range(200, 204)),
        },
        batch_size=2,
    )

    observed = [dataset_name for dataset_name, _ in sampler]
    assert observed == ["MTA", "MFCP", "USTC-TFC2016", "MTA", "MFCP", "USTC-TFC2016"]


def test_mean_normalized_val_top1_uses_explicit_current_and_reference():
    from src.stage2_trainer import mean_normalized_val_top1

    current = {
        "MTA": 0.70,
        "MFCP": 0.62,
        "USTC-TFC2016": 0.86,
    }
    reference = {
        dataset: float(ACCEPTANCE_GATES[dataset]["reference_top1"]) for dataset in STAGE2_DATASET_ORDER
    }
    expected = sum(
        float(current[dataset]) / float(reference[dataset]) for dataset in STAGE2_DATASET_ORDER
    ) / float(len(STAGE2_DATASET_ORDER))

    assert math.isclose(mean_normalized_val_top1(current=current, reference=reference), expected, rel_tol=1e-12, abs_tol=1e-12)


def test_run_stage_a_shared_training_returns_best_score_and_payload_contract():
    from src.stage2_trainer import run_stage_a_shared_training

    dataset_to_indices = {
        "MTA": [0, 1, 2],
        "MFCP": [10, 11],
        "USTC-TFC2016": [20, 21, 22, 23],
    }
    current = {"MTA": 0.70, "MFCP": 0.62, "USTC-TFC2016": 0.86}
    reference = {
        "MTA": 0.6977,
        "MFCP": 0.6167,
        "USTC-TFC2016": 0.8554,
    }
    expected_score = sum(float(current[name]) / float(reference[name]) for name in STAGE2_DATASET_ORDER) / float(
        len(STAGE2_DATASET_ORDER)
    )

    result = run_stage_a_shared_training(
        dataset_to_indices=dataset_to_indices,
        batch_size=2,
        current=current,
        reference=reference,
    )
    assert "batch_sampler" in result
    assert "best_score" in result
    assert "best_payload" in result
    assert math.isclose(float(result["best_score"]), expected_score, rel_tol=1e-12, abs_tol=1e-12)
    assert "per_dataset" in result["best_payload"]
    assert "score" in result["best_payload"]
    assert math.isclose(float(result["best_payload"]["score"]), expected_score, rel_tol=1e-12, abs_tol=1e-12)


def test_build_stage2_single_dataset_contract_reindexes_single_dataset_to_zero():
    from src.stage2_trainer import build_stage2_single_dataset_contract

    contract = build_stage2_single_dataset_contract(dataset_name="MTA", output_dim=7)

    assert contract["dataset_name"] == "MTA"
    assert contract["dataset_vocab"] == {"MTA": 0}
    assert contract["output_dims"] == {"MTA": 7}


def test_run_stage2_shared_stage_a_bridges_kwargs_to_stage2_trainer(monkeypatch):
    captured: dict[str, object] = {}
    passthrough = {"best_score": 1.23, "best_payload": {"score": 1.23, "per_dataset": {}}, "batch_sampler": object()}

    def fake_run_stage_a_shared_training(**kwargs):
        captured["kwargs"] = kwargs
        return passthrough

    monkeypatch.setattr(stage2_trainer_module, "run_stage_a_shared_training", fake_run_stage_a_shared_training)

    kwargs = {
        "dataset_to_indices": {"MTA": [1], "MFCP": [2], "USTC-TFC2016": [3]},
        "batch_size": 4,
        "current": {"MTA": 0.71, "MFCP": 0.63, "USTC-TFC2016": 0.87},
        "reference": {"MTA": 0.6977, "MFCP": 0.6167, "USTC-TFC2016": 0.8554},
    }
    result = train_module.run_stage2_shared_stage_a(**kwargs)

    assert result is passthrough
    assert captured["kwargs"] == kwargs


def test_round_robin_dataset_batch_sampler_rejects_unknown_dataset_key():
    from src.stage2_trainer import RoundRobinDatasetBatchSampler

    with pytest.raises(ValueError, match="unknown dataset"):
        RoundRobinDatasetBatchSampler(
            dataset_to_indices={
                "MTA": [0, 1],
                "UNKNOWN": [9],
            },
            batch_size=2,
        )


def test_run_stage_a_shared_training_rejects_non_positive_reference_with_value_error():
    from src.stage2_trainer import run_stage_a_shared_training

    with pytest.raises(ValueError, match="invalid reference_top1"):
        run_stage_a_shared_training(
            dataset_to_indices={
                "MTA": [0],
                "MFCP": [1],
                "USTC-TFC2016": [2],
            },
            batch_size=1,
            current={"MTA": 0.7, "MFCP": 0.62, "USTC-TFC2016": 0.86},
            reference={"MTA": 0.0, "MFCP": 0.6167, "USTC-TFC2016": 0.8554},
        )
