from pathlib import Path
from typing import Union

import pytest
import yaml


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
