from datetime import datetime
from pathlib import Path
from typing import Union

import importlib.util
import os
import subprocess
import sys

import pytest
import yaml

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


CONFIG_EXPECTATIONS = [
    (
        Path("configs/binary.yaml"),
        {
            "task_name": "binary_iscx_mta_mfcp",
            "num_classes": 2,
            "train_split": 0.7,
            "val_split": 0.1,
            "test_split": 0.2,
            "image_size": 112,
            "fixed_bytes": 784,
        },
    ),
    (
        Path("configs/mta.yaml"),
        {
            "task_name": "mta_7cls",
            "num_classes": 7,
            "train_split": 0.7,
            "val_split": 0.1,
            "test_split": 0.2,
            "image_size": 112,
            "fixed_bytes": 784,
        },
    ),
    (
        Path("configs/mfcp.yaml"),
        {
            "task_name": "mfcp_6cls",
            "num_classes": 6,
            "train_split": 0.7,
            "val_split": 0.1,
            "test_split": 0.2,
            "image_size": 112,
            "fixed_bytes": 784,
        },
    ),
    (
        Path("configs/ustc.yaml"),
        {
            "task_name": "ustc_10cls",
            "num_classes": 10,
            "train_split": 0.7,
            "val_split": 0.1,
            "test_split": 0.2,
            "image_size": 112,
            "fixed_bytes": 784,
        },
    ),
]


@pytest.mark.parametrize("path,expected", CONFIG_EXPECTATIONS)
def test_config_metadata(path: Path, expected: dict[str, Union[int, float]]) -> None:
    config = yaml.safe_load(path.read_text())
    for key, value in expected.items():
        assert config[key] == value


def _load_evaluate_module():
    script_path = repo_root / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("scripts.evaluate", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_layout_uses_date(tmp_path):
    from fusion_malicious.config import build_run_layout

    layout = build_run_layout(tmp_path / "runs", "binary", now=datetime(2026, 3, 26))
    assert layout.run_dir == tmp_path / "runs" / "2026-03-26" / "binary"


def test_evaluate_rejects_non_checkpoint_file(tmp_path):
    module = _load_evaluate_module()
    bad_checkpoint = tmp_path / "bad_checkpoint.pt"
    bad_checkpoint.write_text("not a checkpoint")
    with pytest.raises(RuntimeError):
        module.validate_checkpoint(bad_checkpoint)


def test_training_scripts_help_runs():
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONUSERBASE"] = ""
    for script in ("scripts/train_binary.py", "scripts/train_multiclass.py"):
        subprocess.run(
            [sys.executable, str(repo_root / script), "--help"],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
