import math
import random
from collections import defaultdict

from fusion_malicious.data.records import SessionRecord


def stratified_split_records(
    records: list[SessionRecord],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[SessionRecord]]:
    total = round(train_size + val_size + test_size, 5)
    if total != 1.0:
        raise ValueError("split sizes must sum to 1.0")

    ratios = (train_size, val_size, test_size)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record.label_id].append(record)
    for group in groups.values():
        rng.shuffle(group)

    split = {"train": [], "val": [], "test": []}
    for group in groups.values():
        train_count, val_count, test_count = _allocate_counts(len(group), ratios)
        start = 0
        for name, count in zip(["train", "val", "test"], (train_count, val_count, test_count)):
            if count:
                split[name].extend(group[start : start + count])
            start += count

    for subset in split.values():
        rng.shuffle(subset)
    return split


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    floors = [math.floor(total * ratio) for ratio in ratios]
    allocated = sum(floors)
    remainder = total - allocated
    fractions = [
        (index, (total * ratio) - floors[index]) for index, ratio in enumerate(ratios)
    ]
    fractions.sort(key=lambda item: item[1], reverse=True)
    for index, _ in fractions[:remainder]:
        floors[index] += 1
    return tuple(floors)
