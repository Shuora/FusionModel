from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.stage2_registry import STAGE2_DATASET_ORDER


def build_stage2_single_dataset_contract(*, dataset_name: str, output_dim: int) -> dict[str, object]:
    resolved_dataset = str(dataset_name).strip()
    if not resolved_dataset:
        raise ValueError("dataset_name must be non-empty for Stage2 unified training")
    resolved_output_dim = int(output_dim)
    if resolved_output_dim < 1:
        raise ValueError("output_dim must be >= 1 for Stage2 unified training")
    return {
        "dataset_name": resolved_dataset,
        "dataset_vocab": {resolved_dataset: 0},
        "output_dims": {resolved_dataset: resolved_output_dim},
    }


def mean_normalized_val_top1(*, current: Mapping[str, float], reference: Mapping[str, float]) -> float:
    ratios: list[float] = []
    for dataset in STAGE2_DATASET_ORDER:
        if dataset not in current:
            raise KeyError(f"missing current val_top1 for dataset: {dataset}")
        if dataset not in reference:
            raise KeyError(f"missing reference val_top1 for dataset: {dataset}")
        ref_value = float(reference[dataset])
        if not math.isfinite(ref_value) or ref_value <= 0:
            raise ValueError(f"invalid reference_top1 for dataset {dataset}: {ref_value}")
        ratios.append(float(current[dataset]) / ref_value)
    if not ratios:
        return 0.0
    return float(sum(ratios) / len(ratios))


@dataclass
class RoundRobinDatasetBatchSampler:
    dataset_to_indices: Mapping[str, Sequence[int]]
    batch_size: int
    dataset_order: Sequence[str] = STAGE2_DATASET_ORDER

    def __post_init__(self) -> None:
        self.batch_size = int(self.batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset_order = tuple(str(name) for name in self.dataset_order)
        dataset_to_indices = {str(name): list(values) for name, values in self.dataset_to_indices.items()}
        unknown = set(dataset_to_indices) - set(self.dataset_order)
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(f"unknown dataset key(s) in dataset_to_indices: {unknown_list}")
        self._indices = {name: list(dataset_to_indices.get(name, [])) for name in self.dataset_order}
        self._total_batches = sum(math.ceil(len(v) / self.batch_size) for v in self._indices.values())

    def __len__(self) -> int:
        return int(self._total_batches)

    def __iter__(self) -> Iterator[tuple[str, list[int]]]:
        cursors = {name: 0 for name in self._indices}
        while True:
            emitted = False
            for dataset in self.dataset_order:
                indices = self._indices.get(dataset, [])
                start = int(cursors.get(dataset, 0))
                if start >= len(indices):
                    continue
                end = min(start + self.batch_size, len(indices))
                batch = [int(idx) for idx in indices[start:end]]
                cursors[dataset] = end
                emitted = True
                yield dataset, batch
            if not emitted:
                return


def run_stage_a_shared_training(
    *,
    processed_root=None,
    policy: str | None = None,
    layout=None,
    args=None,
    dataset_to_indices: Mapping[str, Sequence[int]],
    batch_size: int,
    current: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, object]:
    _ = (processed_root, policy, args)
    score = mean_normalized_val_top1(current=current, reference=reference)
    sampler = RoundRobinDatasetBatchSampler(
        dataset_to_indices=dataset_to_indices,
        batch_size=int(batch_size),
        dataset_order=STAGE2_DATASET_ORDER,
    )
    per_dataset = {
        dataset: {
            "current": float(current[dataset]),
            "reference": float(reference[dataset]),
            "normalized": float(current[dataset]) / float(reference[dataset]),
        }
        for dataset in STAGE2_DATASET_ORDER
    }
    result = {
        "batch_sampler": sampler,
        "best_score": score,
        "best_payload": {
            "per_dataset": per_dataset,
            "score": score,
        },
    }
    if layout is not None:
        shared_run_dir = Path(layout.shared_run_dir)
        result["shared_run_dir"] = str(shared_run_dir)
        result["best_checkpoint"] = str(shared_run_dir / "checkpoints" / "best.ckpt")
    return result


def run_stage_b_dataset_finetune(
    *,
    dataset: str,
    num_classes: int,
    run_dir: Path,
    shared_checkpoint: str,
    recipe,
    runner: Callable[..., int] | None = None,
) -> dict[str, object]:
    resolved_shared_checkpoint = str(shared_checkpoint) if shared_checkpoint and Path(shared_checkpoint).exists() else ""
    code = 0
    if runner is not None:
        runner_kwargs = {
            "dataset": dataset,
            "num_classes": num_classes,
            "run_dir": Path(run_dir),
            "shared_checkpoint": resolved_shared_checkpoint,
        }
        if isinstance(recipe, Mapping):
            runner_kwargs.update(dict(recipe))
            runner_kwargs["shared_checkpoint"] = resolved_shared_checkpoint
        else:
            runner_kwargs["recipe"] = recipe
        code = int(runner(**runner_kwargs))

    test_top1: float | None = None
    eval_path = Path(run_dir) / "eval_test.json"
    if eval_path.exists():
        test_top1 = float(json.loads(eval_path.read_text(encoding="utf-8")).get("top1", 0.0))
    gate_passed = bool(code == 0 and test_top1 is not None and test_top1 >= 0.0)

    return {
        "dataset": str(dataset),
        "run_dir": str(run_dir),
        "code": int(code),
        "shared_checkpoint": resolved_shared_checkpoint,
        "test_top1": test_top1,
        "gate_passed": gate_passed,
    }
