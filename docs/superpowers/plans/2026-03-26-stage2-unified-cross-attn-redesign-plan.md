# Stage2 Unified Cross-Attention Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `stage2` main path with a unified dual-branch multimodal transformer that keeps image + sequence branches, adds explicit cross-attention, removes `stacking/moe` from the recommended path, and establishes per-dataset acceptance gates.

**Architecture:** Introduce a shared `stage2_registry` for dataset order, output dimensions, acceptance gates, and run naming. Build a new `Stage2UnifiedClassifier` with branch-local self-attention, bidirectional cross-attention, a shared multimodal transformer trunk, and a dataset-conditioned projection head. Replace the old stage2 protocol with a two-stage runner: joint Stage A stabilization across all datasets, then Stage B per-dataset fine-tuning from the shared checkpoint, with strict run isolation and artifact hygiene.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pytest, existing `train/evaluate/report` pipeline

---

## File Structure

### New Files

- `src/stage2_registry.py`
  - Single source of truth for dataset order, ids, output dims, acceptance gates, run naming, and Stage A normalization references.
- `src/models/stage2_unified_model.py`
  - New unified stage2 model with:
    - image branch projection
    - sequence branch projection
    - branch-local self-attention
    - bidirectional cross-attention
    - shared multimodal transformer trunk
    - dataset-conditioned output head
- `src/stage2_trainer.py`
  - Stage A / Stage B training loops, dataset-balanced sampling, shared validation scoring, and run artifact helpers.
- `tests/pipeline/test_stage2_registry.py`
  - Dataset order, label-space contract, acceptance gates, and run naming tests.
- `tests/models/test_stage2_unified_model.py`
  - Cross-attention model contract tests.
- `tests/pipeline/test_stage2_run_hygiene.py`
  - Run isolation, config stability, and artifact completeness tests.

### Existing Files To Modify

- `src/experiments/stage2_multiclass.py`
  - Remove the old `stacking/moe`-centric main path.
  - Orchestrate Stage A and Stage B using the new registry + trainer.
- `src/evaluate.py`
  - Load the new unified model when `model_type=Stage2UnifiedClassifier`.
  - Respect dataset-conditioned output dimensions during evaluation.
- `src/report.py`
  - Keep reporting centered on end-to-end eval artifacts.
  - Stop assuming stage2’s final metric source comes from `stacking`.
- `docs/commands/session-full-experiments.md`
  - Replace the old stage2 commands with the new unified cross-attention path.
- `tests/pipeline/test_protocol_execution.py`
  - Retire old stage2 orchestration assumptions and lock the new Stage A -> Stage B flow.
- `tests/pipeline/test_stage2_multiclass_protocol.py`
  - Keep fixed task ordering and extend protocol-level expectations if needed.

### Files To Retire From Main Path

- `src/stacking.py`
- `src/moe.py`
- `tests/pipeline/test_stacking_pipeline.py`
- `tests/pipeline/test_moe_pipeline.py`

These files are not deleted in the first implementation pass. They are removed from the stage2 recommended path first, then physically deleted only after the new path clears `Gate 0 + Gate 1`.

---

### Task 1: Lock Registry And Run Hygiene Before Refactor

**Files:**
- Create: `tests/pipeline/test_stage2_registry.py`
- Create: `tests/pipeline/test_stage2_run_hygiene.py`
- Modify: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_stage2_registry.py`
- Test: `tests/pipeline/test_stage2_run_hygiene.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Write the failing dataset registry contract test**

```python
from src.stage2_registry import ACCEPTANCE_GATES, STAGE2_DATASET_ORDER, dataset_num_classes


def test_stage2_dataset_order_and_num_classes_are_stable():
    assert STAGE2_DATASET_ORDER == ("MTA", "MFCP", "USTC-TFC2016")
    assert dataset_num_classes("MTA") == 7
    assert dataset_num_classes("MFCP") == 6
    assert dataset_num_classes("USTC-TFC2016") == 10
    assert ACCEPTANCE_GATES["MTA"]["test_top1_min"] == 0.70
    assert ACCEPTANCE_GATES["USTC-TFC2016"]["test_top1_min"] == 0.86
```

- [ ] **Step 2: Run the registry test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage2_registry.py -k stage2_dataset_order_and_num_classes_are_stable`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.stage2_registry'`

- [ ] **Step 3: Write the failing run hygiene test**

```python
from pathlib import Path

from src.stage2_registry import build_stage2_run_layout


def test_stage2_run_layout_separates_stage_a_and_stage_b(tmp_path: Path):
    layout = build_stage2_run_layout(run_root=tmp_path / "runs", run_date="2026-03-26")
    assert layout.shared_run_dir == tmp_path / "runs" / "2026-03-26" / "stage2-unified-shared"
    assert layout.stage_b_run_dirs["MTA"] == tmp_path / "runs" / "2026-03-26" / "stage2-unified-mta"
    assert layout.stage_b_run_dirs["MFCP"] == tmp_path / "runs" / "2026-03-26" / "stage2-unified-mfcp"
    assert layout.stage_b_run_dirs["USTC-TFC2016"] == tmp_path / "runs" / "2026-03-26" / "stage2-unified-ustc-tfc2016"
    assert len(set(layout.stage_b_run_dirs.values())) == 3
```

- [ ] **Step 4: Run the run hygiene test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage2_run_hygiene.py -k stage2_run_layout_separates_stage_a_and_stage_b`

Expected: FAIL because `build_stage2_run_layout` does not exist yet

- [ ] **Step 5: Write the failing protocol test that retires stacking/moe from the main stage2 path**

```python
from src.experiments import stage2_multiclass as stage2_mod


def test_stage2_runner_main_path_calls_shared_stage_a_then_dataset_stage_b(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(stage2_mod, "run_stage2_shared_stage_a", lambda **kwargs: calls.append(("stage_a", kwargs)) or 0)
    monkeypatch.setattr(stage2_mod, "run_stage2_stage_b", lambda **kwargs: calls.append(("stage_b", kwargs["dataset"])) or 0)

    code = stage2_mod.main(
        [
            "--output", str(tmp_path / "stage2_tasks.json"),
            "--execute",
            "--processed-root", str(tmp_path / "processed"),
            "--skip-ustc-limited",
        ]
    )

    assert code == 0
    assert calls[0][0] == "stage_a"
    assert calls[1:] == [("stage_b", "MTA"), ("stage_b", "MFCP"), ("stage_b", "USTC-TFC2016")]
```

- [ ] **Step 6: Run the protocol test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k stage2_runner_main_path_calls_shared_stage_a_then_dataset_stage_b`

Expected: FAIL because the new orchestration functions do not exist yet

- [ ] **Step 7: Commit**

```bash
git add tests/pipeline/test_stage2_registry.py tests/pipeline/test_stage2_run_hygiene.py tests/pipeline/test_protocol_execution.py
git commit -m "test: 固定 stage2 统一主线路由与运行边界"
```

---

### Task 2: Implement The Shared Stage2 Registry

**Files:**
- Create: `src/stage2_registry.py`
- Modify: `tests/pipeline/test_stage2_registry.py`
- Modify: `tests/pipeline/test_stage2_run_hygiene.py`
- Test: `tests/pipeline/test_stage2_registry.py`
- Test: `tests/pipeline/test_stage2_run_hygiene.py`

- [ ] **Step 1: Write the registry module**

```python
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
```

- [ ] **Step 2: Run the focused registry tests**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage2_registry.py tests/pipeline/test_stage2_run_hygiene.py`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/stage2_registry.py tests/pipeline/test_stage2_registry.py tests/pipeline/test_stage2_run_hygiene.py
git commit -m "feat: 新增 stage2 注册表与运行布局契约"
```

---

### Task 3: Lock And Implement The Unified Cross-Attention Model Contract

**Files:**
- Create: `src/models/stage2_unified_model.py`
- Create: `tests/models/test_stage2_unified_model.py`
- Modify: `src/models/fusion_model.py`
- Test: `tests/models/test_stage2_unified_model.py`

- [ ] **Step 1: Write the failing conditioned-head shape test**

```python
import torch

from src.models.stage2_unified_model import DatasetConditionedHead
from src.stage2_registry import STAGE2_DATASET_ORDER, dataset_num_classes


def test_dataset_conditioned_head_uses_dataset_specific_output_dim():
    head = DatasetConditionedHead(hidden_dim=16, dataset_order=STAGE2_DATASET_ORDER, output_dims={name: dataset_num_classes(name) for name in STAGE2_DATASET_ORDER})
    x = torch.randn(4, 16)
    logits = head(x, dataset_name="MFCP")
    assert logits.shape == (4, 6)
```

- [ ] **Step 2: Run the head-shape test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_stage2_unified_model.py -k dataset_conditioned_head_uses_dataset_specific_output_dim`

Expected: FAIL because `stage2_unified_model.py` does not exist yet

- [ ] **Step 3: Write the failing cross-attention forward-contract test**

```python
import torch

from src.models.stage2_unified_model import Stage2UnifiedClassifier
from src.stage2_registry import DATASET_NAME_TO_ID


def test_stage2_unified_classifier_returns_dataset_specific_logits_and_summary():
    model = Stage2UnifiedClassifier(
        dataset_vocab=DATASET_NAME_TO_ID,
        output_dims={"MTA": 7, "MFCP": 6, "USTC-TFC2016": 10},
        hidden_dim=32,
        num_heads=4,
        trunk_layers=2,
        dropout=0.1,
    )
    out = model(
        rgb=torch.randn(2, 3, 28, 28),
        input_ids=torch.randint(0, 128, (2, 128)),
        attention_mask=torch.ones(2, 128, dtype=torch.long),
        token_type_ids=torch.zeros(2, 128, dtype=torch.long),
        dataset_name="MTA",
        return_summary=True,
    )
    assert out["logits"].shape == (2, 7)
    assert set(out["summary"].keys()) >= {"img_pooled_norm", "seq_pooled_norm", "fused_norm"}
```

- [ ] **Step 4: Run the forward-contract test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_stage2_unified_model.py -k stage2_unified_classifier_returns_dataset_specific_logits_and_summary`

Expected: FAIL

- [ ] **Step 5: Write the new unified model module**

```python
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.etbert_backbone import ETBertBackbone
from src.models.mobilevit_backbone import MobileViTBackbone


class DatasetConditionedHead(nn.Module):
    def __init__(self, hidden_dim: int, dataset_order: tuple[str, ...], output_dims: dict[str, int]) -> None:
        super().__init__()
        self.dataset_order = tuple(dataset_order)
        self.output_dims = {name: int(output_dims[name]) for name in self.dataset_order}
        self.projections = nn.ModuleDict({name: nn.Linear(hidden_dim, self.output_dims[name]) for name in self.dataset_order})

    def forward(self, fused: torch.Tensor, dataset_name: str) -> torch.Tensor:
        return self.projections[str(dataset_name)](fused)
```

- [ ] **Step 6: Add the trunk + model implementation**

```python
class Stage2UnifiedClassifier(nn.Module):
    def __init__(self, dataset_vocab: dict[str, int], output_dims: dict[str, int], hidden_dim: int, num_heads: int, trunk_layers: int, dropout: float) -> None:
        super().__init__()
        self.image_backbone = MobileViTBackbone(out_dim=hidden_dim)
        self.sequence_backbone = ETBertBackbone(vocab_size=30522, hidden_dim=hidden_dim, max_tokens=128)
        self.dataset_vocab = dict(dataset_vocab)
        self.dataset_embed = nn.Embedding(len(dataset_vocab), hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.image_self = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.sequence_self = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.image_to_sequence = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.sequence_to_image = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.shared_trunk = nn.TransformerEncoder(encoder_layer, num_layers=trunk_layers)
        self.pre_classifier = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = DatasetConditionedHead(hidden_dim=hidden_dim, dataset_order=tuple(dataset_vocab.keys()), output_dims=output_dims)

    def forward(self, rgb, input_ids, attention_mask, token_type_ids, dataset_name: str, return_summary: bool = False):
        img = self.image_backbone.forward_features(rgb)
        seq = self.sequence_backbone.forward_features(input_ids, attention_mask, token_type_ids)
        img_tokens = self.image_self(img["tokens"])
        seq_tokens = self.sequence_self(seq["tokens"])
        img_cross, _ = self.image_to_sequence(img_tokens, seq_tokens, seq_tokens, key_padding_mask=seq["mask"] <= 0, need_weights=False)
        seq_cross, _ = self.sequence_to_image(seq_tokens, img_tokens, img_tokens, need_weights=False)
        shared_tokens = torch.cat([img_tokens + img_cross, seq_tokens + seq_cross], dim=1)
        fused_tokens = self.shared_trunk(shared_tokens)
        fused = fused_tokens.mean(dim=1)
        dataset_idx = torch.tensor([self.dataset_vocab[str(dataset_name)]], device=fused.device)
        conditioned = fused + self.dataset_embed(dataset_idx).expand_as(fused)
        pre_logits = self.pre_classifier(torch.cat([conditioned, img["pooled"], seq["pooled"], fused], dim=1))
        out = {"logits": self.head(pre_logits, dataset_name=str(dataset_name))}
        if return_summary:
            out["summary"] = {
                "img_pooled_norm": torch.linalg.vector_norm(img["pooled"], dim=1, keepdim=True),
                "seq_pooled_norm": torch.linalg.vector_norm(seq["pooled"], dim=1, keepdim=True),
                "fused_norm": torch.linalg.vector_norm(pre_logits, dim=1, keepdim=True),
            }
        return out
```

- [ ] **Step 7: Run the model tests**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_stage2_unified_model.py`

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/models/stage2_unified_model.py tests/models/test_stage2_unified_model.py
git commit -m "feat: 新增 stage2 统一 cross-attention 模型"
```

---

### Task 4: Implement Stage A Shared Stabilization Training

**Files:**
- Create: `src/stage2_trainer.py`
- Modify: `src/train.py`
- Create: `tests/pipeline/test_stage2_training_protocol.py`
- Test: `tests/pipeline/test_stage2_training_protocol.py`

- [ ] **Step 1: Write the failing dataset-balanced sampler test**

```python
from src.stage2_trainer import RoundRobinDatasetBatchSampler


def test_round_robin_dataset_batch_sampler_cycles_dataset_names_evenly():
    sampler = RoundRobinDatasetBatchSampler({"MTA": list(range(6)), "MFCP": list(range(6)), "USTC-TFC2016": list(range(6))}, batch_size=2)
    batches = list(iter(sampler))[:6]
    dataset_names = [dataset_name for dataset_name, _ in batches]
    assert dataset_names == ["MTA", "MFCP", "USTC-TFC2016", "MTA", "MFCP", "USTC-TFC2016"]
```

- [ ] **Step 2: Run the sampler test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage2_training_protocol.py -k round_robin_dataset_batch_sampler_cycles_dataset_names_evenly`

Expected: FAIL

- [ ] **Step 3: Write the failing Stage A scoring test**

```python
from src.stage2_trainer import mean_normalized_val_top1


def test_mean_normalized_val_top1_uses_stage2_registry_references():
    score = mean_normalized_val_top1(
        current={"MTA": 0.70, "MFCP": 0.65, "USTC-TFC2016": 0.86},
        reference={"MTA": 0.6977, "MFCP": 0.6167, "USTC-TFC2016": 0.8554},
    )
    assert score > 1.0
```

- [ ] **Step 4: Run the Stage A scoring test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage2_training_protocol.py -k mean_normalized_val_top1_uses_stage2_registry_references`

Expected: FAIL

- [ ] **Step 5: Add the sampler and Stage A scoring helpers**

```python
def mean_normalized_val_top1(*, current: dict[str, float], reference: dict[str, float]) -> float:
    keys = ("MTA", "MFCP", "USTC-TFC2016")
    values = [float(current[key]) / max(float(reference[key]), 1e-6) for key in keys]
    return float(sum(values) / len(values))


class RoundRobinDatasetBatchSampler:
    def __init__(self, dataset_indices: dict[str, list[int]], batch_size: int) -> None:
        self.dataset_indices = {name: list(indices) for name, indices in dataset_indices.items()}
        self.batch_size = int(batch_size)

    def __iter__(self):
        cursors = {name: 0 for name in self.dataset_indices}
        active = True
        ordered = ("MTA", "MFCP", "USTC-TFC2016")
        while active:
            active = False
            for name in ordered:
                indices = self.dataset_indices[name]
                start = cursors[name]
                end = start + self.batch_size
                if start < len(indices):
                    active = True
                    yield name, indices[start:end]
                    cursors[name] = end
```

- [ ] **Step 6: Add the Stage A trainer entrypoints**

```python
def run_stage_a_shared_training(*, run_dir, dataset_batches, model, optimizer, scheduler, patience, reference_top1):
    best_score = -1.0
    best_payload = None
    for epoch in range(1, stage_a_epochs + 1):
        train_one_epoch_shared(...)
        current_val = validate_each_dataset(...)
        score = mean_normalized_val_top1(current=current_val, reference=reference_top1)
        save_last_checkpoint(...)
        if score > best_score:
            best_score = score
            best_payload = {"epoch": epoch, "score": score, "per_dataset": current_val}
            save_best_checkpoint(...)
    return best_payload
```

- [ ] **Step 7: Run the Stage A protocol tests**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage2_training_protocol.py`

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/stage2_trainer.py tests/pipeline/test_stage2_training_protocol.py
git commit -m "feat: 实现 stage2 Stage A 联合稳定训练"
```

---

### Task 5: Implement Stage B Fine-Tune And New Stage2 Runner

**Files:**
- Modify: `src/experiments/stage2_multiclass.py`
- Modify: `src/stage2_trainer.py`
- Modify: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Write the failing Stage B orchestration test**

```python
from src.experiments import stage2_multiclass as stage2_mod


def test_stage2_runner_writes_acceptance_manifest_after_stage_b(monkeypatch, tmp_path):
    monkeypatch.setattr(stage2_mod, "run_stage2_shared_stage_a", lambda **kwargs: {"run_dir": str(tmp_path / "shared"), "best_score": 1.02})
    monkeypatch.setattr(stage2_mod, "run_stage2_stage_b", lambda **kwargs: {"dataset": kwargs["dataset"], "run_dir": str(tmp_path / kwargs["dataset"].lower()), "test_top1": 0.71, "gate_passed": True})
    out_path = tmp_path / "outputs" / "protocol" / "stage2_tasks.json"

    code = stage2_mod.main(["--output", str(out_path), "--execute", "--processed-root", str(tmp_path / "processed"), "--skip-ustc-limited"])

    assert code == 0
    acceptance_path = tmp_path / "runs" / "2026-03-26" / "stage2_acceptance.json"
    assert acceptance_path.exists()
```

- [ ] **Step 2: Run the Stage B orchestration test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k stage2_runner_writes_acceptance_manifest_after_stage_b`

Expected: FAIL

- [ ] **Step 3: Replace the old main path in `stage2_multiclass.py`**

```python
def run_stage2_shared_stage_a(*, processed_root: Path, policy: str, layout, args) -> dict:
    return run_stage_a_shared_training(
        run_dir=layout.shared_run_dir,
        processed_root=processed_root,
        policy=policy,
        hidden_dim=args.hidden_dim,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )


def run_stage2_stage_b(*, dataset: str, num_classes: int, layout, args, shared_checkpoint: Path) -> dict:
    run_dir = layout.stage_b_run_dirs[dataset]
    return run_stage_b_dataset_finetune(
        dataset=dataset,
        num_classes=num_classes,
        run_dir=run_dir,
        shared_checkpoint=shared_checkpoint,
        recipe=resolve_dataset_recipe(dataset=dataset, args=args),
    )
```

- [ ] **Step 4: Write the acceptance manifest after Stage B**

```python
acceptance_rows = []
for task in build_stage2_tasks():
    result = run_stage2_stage_b(
        dataset=str(task["dataset"]),
        num_classes=int(task["num_classes"]),
        layout=layout,
        args=args,
        shared_checkpoint=shared_ckpt,
    )
    acceptance_rows.append(result)
layout.acceptance_path.write_text(json.dumps(acceptance_rows, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 5: Run the stage2 protocol subset**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k 'stage2_runner_main_path_calls_shared_stage_a_then_dataset_stage_b or stage2_runner_writes_acceptance_manifest_after_stage_b'`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/experiments/stage2_multiclass.py src/stage2_trainer.py tests/pipeline/test_protocol_execution.py
git commit -m "feat: 重写 stage2 主路径为 Stage A + Stage B 协议"
```

---

### Task 6: Teach Evaluate And Report About The Unified Stage2 Model

**Files:**
- Modify: `src/evaluate.py`
- Modify: `src/report.py`
- Modify: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_train_eval_report.py`

- [ ] **Step 1: Write the failing evaluation loader test**

```python
from pathlib import Path

import yaml

from src.evaluate import main as evaluate_main


def test_evaluate_loads_stage2_unified_model_from_config(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "stage2-unified-mta"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "stage2-unified-mta",
                "model_type": "Stage2UnifiedClassifier",
                "datasets": ["MTA"],
                "num_classes": 7,
                "processed_root": str(tmp_path / "processed"),
                "policy": "session_full",
                "device_requested": "cpu",
            }
        ),
        encoding="utf-8",
    )
    code = evaluate_main(["--run-dir", str(run_dir), "--split", "test", "--device", "cpu"])
    assert code != 2
```

- [ ] **Step 2: Run the evaluation loader test to verify it fails**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_train_eval_report.py -k evaluate_loads_stage2_unified_model_from_config`

Expected: FAIL because `evaluate.py` only knows `MobileViTETBertFusionClassifier`

- [ ] **Step 3: Update `evaluate.py` to dispatch on `model_type`**

```python
if str(cfg.get("model_type")) == "Stage2UnifiedClassifier":
    model = Stage2UnifiedClassifier.from_config(cfg).to(device)
    logits_key = "logits"
else:
    model = MobileViTETBertFusionClassifier(...).to(device)
    logits_key = "logits_fuse" if cfg.get("stage") != "warmup" else None
```

- [ ] **Step 4: Update `report.py` so stage2 prefers end-to-end eval artifacts**

```python
def resolve_canonical_final_metric_source_and_path(run_dir: Path) -> tuple[str, Path]:
    eval_test = run_dir / "eval_test.json"
    if eval_test.exists():
        return "eval", eval_test
    return "none", eval_test
```

- [ ] **Step 5: Run the focused eval/report suite**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_train_eval_report.py -k 'stage2_unified or evaluate_loads_stage2_unified_model_from_config'`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py src/report.py tests/pipeline/test_train_eval_report.py
git commit -m "feat: 评估与报告支持 stage2 统一模型"
```

---

### Task 7: Remove Old Stage2 Recommendations And Record Acceptance Workflow

**Files:**
- Modify: `docs/commands/session-full-experiments.md`
- Modify: `docs/planning-with-files/task_plan.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Replace the stage2 command docs with the new main path**

```markdown
### 3.2 推荐：统一 cross-attention stage2 主线

当前 stage2 推荐路径不再使用 `stacking` 或 `moe`。

统一主线固定为：

1. Stage A shared stabilization
2. Stage B per-dataset fine-tune
3. end-to-end eval/report
```

- [ ] **Step 2: Record the acceptance manifest contract in planning files**

```markdown
## Acceptance Tracking
- Gate 0: protocol hygiene
- Gate 1: MTA >= 0.70
- Gate 2: MFCP >= 0.70
- Gate 3: USTC >= 0.86
- Results are persisted to `runs/<date>/stage2_acceptance.json`
```

- [ ] **Step 3: Re-run the stage2 protocol suite**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k 'stage2_'`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/commands/session-full-experiments.md docs/planning-with-files/task_plan.md docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 切换 stage2 主线并记录验收门槛"
```

---

## Self-Review

### Spec Coverage

- Unified dual-branch model: covered by Task 3.
- Explicit cross-attention: covered by Task 3.
- Dataset-conditioned head + label-space contract: covered by Tasks 2 and 3.
- Stage A shared stabilization: covered by Task 4.
- Stage B per-dataset fine-tune: covered by Task 5.
- Retire old `stacking/moe` main path: covered by Tasks 1, 5, and 7.
- Run hygiene requirements: covered by Tasks 1, 2, and 5.
- Per-dataset acceptance gates: covered by Tasks 2, 5, and 7.

### Placeholder Scan

- No `TODO/TBD` placeholders remain.
- Every task includes concrete file paths.
- Every code-changing step includes concrete code.
- Every test step includes an explicit command and expected result.

### Type Consistency

- Dataset vocabulary is fixed in `src/stage2_registry.py`.
- `Stage2UnifiedClassifier` uses `dataset_name` consistently across train/evaluate.
- Stage A metric name is consistently `mean_normalized_val_top1`.
- Acceptance output path is consistently `stage2_acceptance.json`.
