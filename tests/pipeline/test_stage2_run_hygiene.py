from pathlib import Path

from src.stage2_registry import build_stage2_run_layout


def test_stage2_run_layout_separates_stage_a_and_stage_b(tmp_path: Path):
    layout = build_stage2_run_layout(run_root=tmp_path / "runs", run_date="2026-03-26")
    assert layout.shared_run_dir == tmp_path / "runs" / "2026-03-26" / "stage2-unified-shared"
    assert layout.stage_b_run_dirs["MTA"] == tmp_path / "runs" / "2026-03-26" / "stage2-unified-mta"
    assert layout.stage_b_run_dirs["MFCP"] == tmp_path / "runs" / "2026-03-26" / "stage2-unified-mfcp"
    assert layout.stage_b_run_dirs["USTC-TFC2016"] == tmp_path / "runs" / "2026-03-26" / "stage2-unified-ustc-tfc2016"
    assert len(set(layout.stage_b_run_dirs.values())) == 3
