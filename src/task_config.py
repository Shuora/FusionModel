from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset_names: Tuple[str, ...]
    labels: Tuple[str, ...]
    train_ratio: float = 0.8


TASK_CONFIGS: dict[str, TaskConfig] = {
    "binary_benign_vs_malicious": TaskConfig(
        name="binary_benign_vs_malicious",
        dataset_names=("ISCX-VPN-NonVPN-2016", "USTC-TFC2016", "MTA", "MFCP"),
        labels=("benign", "malicious"),
    ),
    "ustc_multiclass": TaskConfig(
        name="ustc_multiclass",
        dataset_names=("USTC-TFC2016",),
        labels=(),
    ),
    "mta_multiclass": TaskConfig(
        name="mta_multiclass",
        dataset_names=("MTA",),
        labels=(),
    ),
    "mfcp_multiclass": TaskConfig(
        name="mfcp_multiclass",
        dataset_names=("MFCP",),
        labels=(),
    ),
    # Stress test tasks (Imbalance Gradient)
    **{f"mta_ratio{r}": TaskConfig(name=f"mta_ratio{r}", dataset_names=("MTA",), labels=()) for r in (2, 5, 10, 15)},
    **{f"mfcp_ratio{r}": TaskConfig(name=f"mfcp_ratio{r}", dataset_names=("MFCP",), labels=()) for r in (2, 5, 10, 15)},
}


def get_task_config(name: str) -> TaskConfig:
    try:
        return TASK_CONFIGS[name]
    except KeyError as exc:
        raise KeyError(f"unknown task: {name}") from exc
