from pathlib import Path


def test_ablation_runner_outputs_table(tmp_path: Path):
    from src.fusion.run_ablation import run_ablations

    cfg = {
        "output_root": str(tmp_path),
        "run_name": "ablation_smoke",
        "experiments": [
            {"name": "full", "use_fusion": True, "use_rgb": True, "use_stacking": True},
            {"name": "no_stacking", "use_fusion": True, "use_rgb": True, "use_stacking": False},
        ],
    }

    summary_path = run_ablations(cfg)

    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert "experiment" in text
    assert "macro_f1" in text
