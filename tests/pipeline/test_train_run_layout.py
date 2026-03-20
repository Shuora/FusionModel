from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime
from pathlib import Path


def _load_train_module():
    fake_runtime_device = types.ModuleType("src.runtime_device")
    fake_runtime_device.resolve_runtime_device = lambda requested: (requested, "cpu", "unavailable")
    sys.modules["src.runtime_device"] = fake_runtime_device
    sys.modules.pop("src.train", None)
    import src.train as train_module

    return importlib.reload(train_module)


def test_default_run_dir_uses_date_partition_and_unique_leaf(tmp_path: Path, monkeypatch):
    train_module = _load_train_module()

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 3, 20, 14, 5, 7, 123456)

    monkeypatch.setattr(train_module, "datetime", FrozenDateTime)

    run_id, run_dir = train_module._build_run_identity(tmp_path / "runs")

    assert run_id == "140507-123456"
    assert run_dir == tmp_path / "runs" / "2026-03-20" / "140507-123456"
    assert run_dir.parent.name == "2026-03-20"
