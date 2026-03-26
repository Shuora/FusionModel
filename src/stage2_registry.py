from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE2_DATASET_ORDER = ("MTA", "MFCP", "USTC-TFC2016")
_NUM_CLASSES = {"MTA": 7, "MFCP": 6, "USTC-TFC2016": 10}
DATASET_ID_TO_NAME = {0: "MTA", 1: "MFCP", 2: "USTC-TFC2016"}
DATASET_NAME_TO_ID = {name: idx for idx, name in DATASET_ID_TO_NAME.items()}
ACCEPTANCE_GATES = {
    "MTA": {"test_top1_min": 0.70, "reference_top1": 0.6977},
    "MFCP": {"test_top1_min": 0.70, "reference_top1": 0.6167},
    "USTC-TFC2016": {"test_top1_min": 0.86, "reference_top1": 0.8554},
}


def dataset_num_classes(dataset: str) -> int:
    return int(_NUM_CLASSES[dataset])


@dataclass(frozen=True)
class Stage2RunLayout:
    root_dir: Path
    shared_run_dir: Path
    stage_b_run_dirs: dict[str, Path]
    acceptance_path: Path


def build_stage2_run_layout(*, run_root: Path, run_date: str) -> Stage2RunLayout:
    date_root = Path(run_root) / run_date
    return Stage2RunLayout(
        root_dir=date_root,
        shared_run_dir=date_root / "stage2-unified-shared",
        stage_b_run_dirs={
            "MTA": date_root / "stage2-unified-mta",
            "MFCP": date_root / "stage2-unified-mfcp",
            "USTC-TFC2016": date_root / "stage2-unified-ustc-tfc2016",
        },
        acceptance_path=date_root / "stage2_acceptance.json",
    )
