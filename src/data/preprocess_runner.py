from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.structured_logging import format_log_line
from src.data.preprocess import preprocess_source


DEFAULT_POLICY_FILTER_MAP: Dict[str, str] = {
    "strict": "strict",
    "full": "strict",
    "relaxed": "relaxed",
    "session_full": "session_full",
}


def run_preprocess_policies(
    source_root: Path | str,
    output_root: Path | str,
    policies: Sequence[str],
    datasets: Sequence[str] | None = None,
    seed: int = 42,
    cleanup_sessions: bool = True,
    preview_per_family: int = 20,
    show_progress: bool = True,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for policy in policies:
        policy_name = policy.strip()
        if not policy_name:
            continue
        filter_mode = DEFAULT_POLICY_FILTER_MAP.get(policy_name, "strict")
        log_fn(
            format_log_line(
                level="info",
                module="data",
                event="policy_run_start",
                kv={
                    "policy": policy_name,
                    "filter_mode": filter_mode,
                    "datasets": ",".join(datasets) if datasets else "all",
                    "cleanup_sessions": cleanup_sessions,
                    "preview_per_family": preview_per_family,
                },
            )
        )
        results[policy_name] = preprocess_source(
            source_root=source_root,
            output_root=output_root,
            policy=policy_name,
            filter_mode=filter_mode,
            datasets=datasets,
            seed=seed,
            cleanup_sessions=cleanup_sessions,
            preview_per_family=preview_per_family,
            show_progress=show_progress,
            log_fn=log_fn,
        )
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TLS preprocess for multiple policies")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policies", nargs="+", default=["strict", "full"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cleanup-sessions",
        dest="cleanup_sessions",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--keep-sessions",
        dest="cleanup_sessions",
        action="store_false",
    )
    parser.add_argument("--preview-per-family", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_preprocess_policies(
        source_root=args.source_root,
        output_root=args.output_root,
        policies=args.policies,
        datasets=args.datasets,
        seed=args.seed,
        cleanup_sessions=args.cleanup_sessions,
        preview_per_family=args.preview_per_family,
        show_progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
