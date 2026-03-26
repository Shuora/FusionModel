from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path
    task_name: str
    date_text: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.date_text / self.task_name


@dataclass(frozen=True)
class StageConfig:
    name: str
    enable_fusion: bool
    text_train_mode: str


def build_run_layout(root: str | Path, task_name: str, now: datetime | None = None) -> RunLayout:
    timestamp = now or datetime.now()
    date_text = timestamp.strftime("%Y-%m-%d")
    return RunLayout(root=Path(root), task_name=task_name, date_text=date_text)
