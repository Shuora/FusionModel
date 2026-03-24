from __future__ import annotations

from datetime import datetime
from pathlib import Path


def current_run_date_partition(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y-%m-%d")


def build_dated_run_dir(run_root: str | Path, run_id: str, now: datetime | None = None) -> Path:
    return Path(run_root) / current_run_date_partition(now) / str(run_id)


def build_timestamped_run_identity(run_root: str | Path, now: datetime | None = None) -> tuple[str, Path]:
    current = now or datetime.now()
    run_id = current.strftime("%H%M%S-%f")
    return run_id, build_dated_run_dir(run_root=run_root, run_id=run_id, now=current)


def resolve_run_dir(run_dir: str | Path) -> Path:
    candidate = Path(run_dir)
    if _is_run_dir(candidate):
        return candidate.resolve()

    matches = _find_run_dir_matches(candidate)
    if not matches:
        return candidate.resolve()
    if len(matches) == 1:
        return matches[0].resolve()

    dated_matches = [path for path in matches if _looks_like_date_partition(path.parent.name)]
    if dated_matches:
        return sorted(dated_matches, key=lambda path: (path.parent.name, str(path)))[-1].resolve()
    return sorted(matches, key=str)[-1].resolve()


def _find_run_dir_matches(candidate: Path) -> list[Path]:
    roots: list[Path] = []
    if candidate.parent != Path("."):
        roots.append(candidate.parent)
    roots.append(Path("runs"))

    deduped_roots: list[Path] = []
    for root in roots:
        if root not in deduped_roots:
            deduped_roots.append(root)

    matches: list[Path] = []
    for root in deduped_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(candidate.name)):
            if path.name != candidate.name:
                continue
            if _is_run_dir(path) and path not in matches:
                matches.append(path)
    return matches


def _is_run_dir(path: Path) -> bool:
    return (path / "config.yaml").exists()


def _looks_like_date_partition(name: str) -> bool:
    parts = str(name).split("-")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)
