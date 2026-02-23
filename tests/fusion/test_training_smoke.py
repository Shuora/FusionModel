from pathlib import Path


def test_training_creates_run_artifacts(tmp_path: Path):
    from src.fusion.train_stagewise import run_train

    cfg = {
        "run_name": "smoke_train",
        "output_root": str(tmp_path),
        "num_epochs": 2,
        "model": {"num_classes": 4},
    }

    run_dir = run_train(cfg)

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (tmp_path / "logs" / "train" / "smoke_train.log").exists()
