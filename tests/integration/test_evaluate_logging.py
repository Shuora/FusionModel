from pathlib import Path


def test_evaluate_writes_classified_log(tmp_path: Path):
    from src.fusion.evaluate import run_evaluate

    run_dir = tmp_path / "runs" / "demo_run"
    payload = run_evaluate(run_dir)

    assert (run_dir / "evaluation.json").exists()
    assert "metrics" in payload
    assert (tmp_path / "logs" / "evaluate" / "demo_run.log").exists()
