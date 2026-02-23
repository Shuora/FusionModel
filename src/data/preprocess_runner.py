from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from src.common.structured_logging import format_log_line
from src.data.preprocess import preprocess_source


DEFAULT_POLICY_FILTER_MAP: Dict[str, str] = {
    "strict": "strict",
    "full": "strict",
    "relaxed": "relaxed",
}


def run_preprocess_policies(
    source_root: Path | str,
    output_root: Path | str,
    policies: Sequence[str],
    seed: int = 42,
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
                kv={"policy": policy_name, "filter_mode": filter_mode},
            )
        )
        results[policy_name] = preprocess_source(
            source_root=source_root,
            output_root=output_root,
            policy=policy_name,
            filter_mode=filter_mode,
            seed=seed,
            show_progress=show_progress,
            log_fn=log_fn,
        )
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TLS preprocess for multiple policies")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policies", nargs="+", default=["strict", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_preprocess_policies(
        source_root=args.source_root,
        output_root=args.output_root,
        policies=args.policies,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
