from datetime import datetime
from pathlib import Path

import pytest
import torch

from fusion_malicious.config import StageConfig, build_run_layout
from fusion_malicious.utils.device import require_cuda


def test_build_run_layout_uses_date_then_task_name(tmp_path: Path) -> None:
    layout = build_run_layout(
        root=tmp_path,
        task_name="binary_iscx_mta_mfcp",
        now=datetime(2026, 3, 26, 10, 30, 0),
    )
    assert layout.run_dir == tmp_path / "2026-03-26" / "binary_iscx_mta_mfcp"


def test_stage_config_defaults_match_three_stage_plan() -> None:
    stage = StageConfig(name="warmup", enable_fusion=False, text_train_mode="head_only")
    assert stage.name == "warmup"
    assert stage.enable_fusion is False
    assert stage.text_train_mode == "head_only"


def test_require_cuda_raises_when_cuda_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda()
