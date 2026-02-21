from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def _flatten(groups: Sequence[str], grouped_rows: Dict[str, List[dict]]) -> List[dict]:
    out: List[dict] = []
    for capture_id in groups:
        out.extend(grouped_rows[capture_id])
    return out


def group_split_by_capture(
    rows: Iterable[dict],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split rows by capture_id groups to prevent cross-split leakage."""
    grouped_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row.get("capture_id", "unknown"))].append(dict(row))

    capture_ids = list(grouped_rows.keys())
    if not capture_ids:
        return [], [], []

    rng = random.Random(seed)
    rng.shuffle(capture_ids)

    n_groups = len(capture_ids)
    if n_groups == 1:
        train_ids, val_ids, test_ids = capture_ids, [], []
    elif n_groups == 2:
        train_ids, val_ids, test_ids = [capture_ids[0]], [], [capture_ids[1]]
    else:
        train_cut = max(1, int(n_groups * ratios[0]))
        val_cut = train_cut + int(n_groups * ratios[1])

        if val_cut >= n_groups:
            val_cut = n_groups - 1
        if train_cut >= val_cut:
            train_cut = max(1, val_cut - 1)

        train_ids = capture_ids[:train_cut]
        val_ids = capture_ids[train_cut:val_cut]
        test_ids = capture_ids[val_cut:]

        if not test_ids:
            test_ids = [train_ids.pop()]

    return (
        _flatten(train_ids, grouped_rows),
        _flatten(val_ids, grouped_rows),
        _flatten(test_ids, grouped_rows),
    )
