from __future__ import annotations

from pathlib import Path

import pytest

from src.run_dir import resolve_run_dir


def _write_run_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("run_id: demo\n", encoding="utf-8")


def test_resolve_run_dir_returns_exact_existing_dir(tmp_path: Path):
    run_dir = tmp_path / "runs" / "stage1-binary"
    _write_run_dir(run_dir)

    resolved = resolve_run_dir(run_dir)

    assert resolved == run_dir


def test_resolve_run_dir_finds_dated_partition_from_short_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dated_run_dir = tmp_path / "runs" / "2026-03-21" / "stage1-binary"
    _write_run_dir(dated_run_dir)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_run_dir(Path("runs") / "stage1-binary")

    assert resolved == dated_run_dir


def test_resolve_run_dir_prefers_latest_dated_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    older = tmp_path / "runs" / "2026-03-20" / "stage1-binary"
    newer = tmp_path / "runs" / "2026-03-21" / "stage1-binary"
    _write_run_dir(older)
    _write_run_dir(newer)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_run_dir(Path("runs") / "stage1-binary")

    assert resolved == newer
