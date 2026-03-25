from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import src.moe as moe_module
from src.moe import main as moe_main
from src.report import resolve_canonical_final_metric_source_and_path
from src.train import main as train_main


def _prepare_dummy_processed(root: Path) -> None:
    dataset = "DemoSet"
    policy = "strict"
    rgb_dir = root / dataset / policy / "rgb"
    etbert_dir = root / dataset / policy / "etbert"
    manifest_dir = root / dataset / policy / "manifest"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    etbert_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = 12
    session_ids = np.array([f"s{i}" for i in range(1, n + 1)], dtype="U64")
    labels = np.array([0, 1] * (n // 2), dtype=np.int32)
    rgbs = np.random.default_rng(10).integers(0, 256, size=(n, 3, 28, 28), dtype=np.uint8)
    input_ids = np.random.default_rng(11).integers(0, 1024, size=(n, 128), dtype=np.int32)
    attention = np.ones((n, 128), dtype=np.uint8)
    token_types = np.zeros((n, 128), dtype=np.uint8)

    np.savez_compressed(
        rgb_dir / "rgb_shard_00000.npz",
        session_id=session_ids,
        label=labels,
        rgb=rgbs,
    )
    np.savez_compressed(
        etbert_dir / "etbert_shard_00000.npz",
        session_id=session_ids,
        input_ids=input_ids,
        attention_mask=attention,
        token_type_ids=token_types,
    )

    splits = ["train"] * 6 + ["val"] * 4 + ["test"] * 2
    families = ["Fam0" if x == 0 else "Fam1" for x in labels]
    rows = [
        {
            "session_id": sid,
            "dataset": dataset,
            "family": fam,
            "capture_id": f"{fam}.pcap",
            "split": sp,
            "policy": policy,
        }
        for sid, fam, sp in zip(session_ids, families, splits)
    ]
    with (manifest_dir / "session_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_moe_pipeline_outputs_router_and_metrics(tmp_path: Path):
    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed(processed_root)

    code = train_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "moe-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "moe-run"
    code = moe_main(
        [
            "--run-dir",
            str(run_dir),
            "--epochs",
            "2",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    moe_dir = run_dir / "moe"
    assert (moe_dir / "router.ckpt").exists()
    assert (moe_dir / "moe_metrics.json").exists()

    metrics = json.loads((moe_dir / "moe_metrics.json").read_text(encoding="utf-8"))
    assert "top1" in metrics
    assert "macro_f1" in metrics
    assert metrics["n_test_samples"] > 0


def test_moe_router_features_follow_shared_meta_schema_contract():
    try:
        from src.pipeline.meta_features import ROUTER_META_FEATURE_NAMES, build_router_meta_features
    except ImportError as e:  # pragma: no cover
        raise AssertionError("MoE should consume shared stage2 meta feature helper") from e

    out = {
        "logits_img": torch.tensor([[2.0, 0.0, -1.0], [0.2, 1.4, -0.5]], dtype=torch.float32),
        "logits_tls": torch.tensor([[0.5, 1.0, -0.5], [1.1, -0.1, 0.4]], dtype=torch.float32),
        "logits_fuse": torch.tensor([[1.2, 0.3, -0.2], [0.9, 0.8, -0.3]], dtype=torch.float32),
        "summary": {
            "img_pooled_norm": torch.tensor([[0.8], [0.9]], dtype=torch.float32),
            "txt_pooled_norm": torch.tensor([[0.5], [0.4]], dtype=torch.float32),
            "fused_norm": torch.tensor([[0.7], [0.6]], dtype=torch.float32),
        },
    }
    router_x = moe_module._router_features(out)
    expected_router, router_feature_names, router_schema = build_router_meta_features(out)
    assert router_x.shape == (2, len(ROUTER_META_FEATURE_NAMES))
    assert router_feature_names == list(ROUTER_META_FEATURE_NAMES)
    assert router_schema["dim"] == len(ROUTER_META_FEATURE_NAMES)
    assert router_schema["feature_names"] == list(ROUTER_META_FEATURE_NAMES)
    assert torch.allclose(router_x, expected_router, atol=1e-6)


def test_level3_moe_hook_uses_shared_meta_and_keeps_level2_default(tmp_path: Path):
    from src.pipeline.meta_features import ROUTER_META_FEATURE_NAMES, STAGE2_META_SCHEMA_VERSION

    processed_root = tmp_path / "outputs" / "processed"
    run_root = tmp_path / "runs"
    _prepare_dummy_processed(processed_root)

    code = train_main(
        [
            "--processed-root",
            str(processed_root),
            "--policy",
            "strict",
            "--stage",
            "fusion",
            "--run-root",
            str(run_root),
            "--run-id",
            "moe-level3-hook-run",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    run_dir = run_root / "moe-level3-hook-run"
    stack_dir = run_dir / "stacking"
    stack_dir.mkdir(parents=True, exist_ok=True)
    stacking_final_path = stack_dir / "final_metrics.json"
    stacking_final_payload = {
        "top1": 0.88,
        "macro_f1": 0.87,
        "macro_recall": 0.86,
        "n_test_samples": 2,
        "metric_source": "stacking_final",
        "is_final_stage2_result": True,
    }
    stacking_final_path.write_text(
        json.dumps(
            stacking_final_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stacking_final_text_before = stacking_final_path.read_text(encoding="utf-8")

    code = moe_main(
        [
            "--run-dir",
            str(run_dir),
            "--epochs",
            "2",
            "--batch-size",
            "4",
        ]
    )
    assert code == 0

    moe_dir = run_dir / "moe"
    assert (moe_dir / "router.ckpt").exists()
    assert (moe_dir / "moe_metrics.json").exists()
    assert (moe_dir / "router_meta_schema.json").exists()
    assert not (moe_dir / "final_metrics.json").exists()

    schema_payload = json.loads((moe_dir / "router_meta_schema.json").read_text(encoding="utf-8"))
    assert schema_payload["version"] == STAGE2_META_SCHEMA_VERSION
    assert schema_payload["feature_names"] == list(ROUTER_META_FEATURE_NAMES)
    assert schema_payload["dim"] == len(ROUTER_META_FEATURE_NAMES)

    metric_source, metric_path = resolve_canonical_final_metric_source_and_path(run_dir)
    assert metric_source == "stacking"
    assert metric_path == stacking_final_path

    # Optional level3 enhancer: explicitly invoked, consumes shared meta features, and writes a
    # stage2-final artifact without mutating level2 outputs.
    code = moe_main(
        [
            "--run-dir",
            str(run_dir),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--level3",
        ]
    )
    assert code == 0

    assert (moe_dir / "final_metrics.json").exists()
    metric_source, metric_path = resolve_canonical_final_metric_source_and_path(run_dir)
    assert metric_source == "moe"
    assert metric_path == (moe_dir / "final_metrics.json")

    assert stacking_final_path.read_text(encoding="utf-8") == stacking_final_text_before
