# SourceData Attention Two-Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor preprocessing and training to use `SourceData`, standardize processed datasets per task, keep only attention-based fusion plus attention stacking, and support independent binary and per-dataset multiclass training tasks.

**Architecture:** Introduce one shared task-definition module used by both preprocessing and training. Preprocessing will transform heterogeneous raw datasets under `SourceData` into standardized task-specific processed roots under `ProcessedData/<task_name>`, then image generation will derive paired PNGs from generated `.bin` files. Training will consume only processed roots, delete non-attention neural fusion paths, and expose task-oriented attention and attention-stacking entrypoints.

**Tech Stack:** Python, PyTorch, torchvision, transformers, NumPy, PIL, dpkt, pytest, git worktrees

---

### Task 1: Create Isolated Worktree And Baseline Inventory

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/progress.md`
- Verify: `/home/shuora/Traffic/FusionModel/.worktrees/`

- [ ] **Step 1: Verify `.worktrees` is ignored before creating a worktree**

Run: `git check-ignore -q .worktrees`
Expected: exit code `0`

- [ ] **Step 2: Create a dedicated worktree for the refactor**

Run: `git worktree add .worktrees/codex-source-data-attention -b codex/source-data-attention`
Expected: git reports a new worktree on branch `codex/source-data-attention`

- [ ] **Step 3: Record the implementation start in the progress file**

Add this line under the existing `2026-03-28` section in `/home/shuora/Traffic/FusionModel/progress.md`:

```md
- Implementation plan approved; next step is isolated worktree creation for the SourceData attention refactor.
```

- [ ] **Step 4: Verify baseline test command availability without running `mvn test`**

Run: `pytest --version`
Expected: prints the installed pytest version or a clear "command not found" signal to guide follow-up setup

- [ ] **Step 5: Commit the planning-only update in the worktree if policy requires a checkpoint**

Run: `git status --short`
Expected: only expected planning/worktree-related changes appear

### Task 2: Introduce Shared Task Definitions And Tests

**Files:**
- Create: `/home/shuora/Traffic/FusionModel/src/task_config.py`
- Create: `/home/shuora/Traffic/FusionModel/tests/test_task_config.py`
- Modify: `/home/shuora/Traffic/FusionModel/findings.md`

- [ ] **Step 1: Write the failing tests for task definition lookup and label resolution metadata**

Create `/home/shuora/Traffic/FusionModel/tests/test_task_config.py` with tests like:

```python
from task_config import TASK_CONFIGS, get_task_config


def test_binary_task_exists_and_uses_expected_labels():
    cfg = get_task_config("binary_benign_vs_malicious")
    assert cfg.name == "binary_benign_vs_malicious"
    assert cfg.labels == ["benign", "malicious"]


def test_multiclass_tasks_are_dataset_specific():
    assert get_task_config("ustc_multiclass").dataset_names == ["USTC-TFC2016"]
    assert get_task_config("mta_multiclass").dataset_names == ["MTA"]
    assert get_task_config("mfcp_multiclass").dataset_names == ["MFCP"]


def test_unknown_task_raises_key_error():
    try:
        get_task_config("missing_task")
    except KeyError as exc:
        assert "missing_task" in str(exc)
    else:
        raise AssertionError("expected KeyError")
```

- [ ] **Step 2: Run the new tests to verify RED**

Run: `pytest tests/test_task_config.py -v`
Expected: FAIL because `task_config.py` does not exist yet

- [ ] **Step 3: Write the minimal shared task definition module**

Create `/home/shuora/Traffic/FusionModel/src/task_config.py` with dataclasses and a simple registry like:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset_names: list[str]
    labels: list[str]
    train_ratio: float


TASK_CONFIGS = {
    "binary_benign_vs_malicious": TaskConfig(
        name="binary_benign_vs_malicious",
        dataset_names=["ISCX-VPN-NonVPN-2016", "USTC-TFC2016", "MTA", "MFCP"],
        labels=["benign", "malicious"],
        train_ratio=0.8,
    ),
    # add the three multiclass task definitions here
}


def get_task_config(name: str) -> TaskConfig:
    try:
        return TASK_CONFIGS[name]
    except KeyError as exc:
        raise KeyError(f"unknown task: {name}") from exc
```

- [ ] **Step 4: Re-run the task config tests to verify GREEN**

Run: `pytest tests/test_task_config.py -v`
Expected: PASS

- [ ] **Step 5: Update findings with the new shared-source-of-truth decision**

Add this note to `/home/shuora/Traffic/FusionModel/findings.md`:

```md
- Preprocessing and training will share a single task-definition module to avoid duplicated label and path rules.
```

### Task 3: Refactor Raw Discovery And Session Splitting For `SourceData`

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/src/split_data.py`
- Create: `/home/shuora/Traffic/FusionModel/tests/test_split_data_tasks.py`
- Modify: `/home/shuora/Traffic/FusionModel/progress.md`

- [ ] **Step 1: Write failing tests for raw-file discovery across the four supported layouts**

Create `/home/shuora/Traffic/FusionModel/tests/test_split_data_tasks.py` with tests like:

```python
from pathlib import Path

from split_data import discover_task_inputs


def test_discover_ustc_flat_files(tmp_path: Path):
    root = tmp_path / "SourceData" / "USTC-TFC2016"
    root.mkdir(parents=True)
    (root / "Geodo.pcap").write_bytes(b"x")

    items = discover_task_inputs(tmp_path / "SourceData", "ustc_multiclass")

    assert len(items) == 1
    assert items[0].label == "Geodo"


def test_discover_mta_family_directory(tmp_path: Path):
    root = tmp_path / "SourceData" / "MTA" / "Dridex"
    root.mkdir(parents=True)
    (root / "Dridex.pcap").write_bytes(b"x")

    items = discover_task_inputs(tmp_path / "SourceData", "mta_multiclass")

    assert [item.label for item in items] == ["Dridex"]
```

- [ ] **Step 2: Run the split discovery tests to verify RED**

Run: `pytest tests/test_split_data_tasks.py -v`
Expected: FAIL because the current `split_data.py` does not expose the task-driven discovery API

- [ ] **Step 3: Refactor `split_data.py` to expose task-driven discovery and output roots**

Implement at least these pieces in `/home/shuora/Traffic/FusionModel/src/split_data.py`:

```python
@dataclass(frozen=True)
class RawSample:
    raw_path: Path
    label: str
    dataset_name: str


def discover_task_inputs(source_root: Path, task_name: str) -> list[RawSample]:
    ...


def split_task_inputs(samples: list[RawSample], train_ratio: float, seed: int) -> dict[str, list[RawSample]]:
    ...


def build_processed_root(base_dir: Path, task_name: str) -> Path:
    return base_dir / "ProcessedData" / task_name
```

Also update the CLI so it can accept `--task_name`, `--source_root`, and `--processed_root`.

- [ ] **Step 4: Add `.pcapng` extension support in discovery and route parsing through a dedicated reader helper**

Implement a reader branch with a signature like:

```python
def iter_packets(capture_path: Path):
    suffix = capture_path.suffix.lower()
    if suffix == ".pcap":
        ...
    elif suffix == ".pcapng":
        ...
    else:
        raise ValueError(f"unsupported capture type: {capture_path}")
```

- [ ] **Step 5: Re-run the split discovery tests to verify GREEN**

Run: `pytest tests/test_split_data_tasks.py -v`
Expected: PASS

- [ ] **Step 6: Record the preprocessing refactor milestone**

Append this line to `/home/shuora/Traffic/FusionModel/progress.md`:

```md
- Started refactoring `split_data.py` into a task-driven SourceData preprocessor with processed-root outputs.
```

### Task 4: Refactor RGB Image Generation Around Processed Roots

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/src/ssl_tls_rgb_image.py`
- Create: `/home/shuora/Traffic/FusionModel/tests/test_ssl_tls_rgb_image.py`

- [ ] **Step 1: Write a failing test for processed-root path preservation**

Create `/home/shuora/Traffic/FusionModel/tests/test_ssl_tls_rgb_image.py` with a test like:

```python
from pathlib import Path

from ssl_tls_rgb_image import get_output_path


def test_output_path_preserves_split_and_label(tmp_path: Path):
    dataset_root = tmp_path / "ProcessedData" / "binary_benign_vs_malicious"
    bin_path = dataset_root / "pcap_data" / "Train" / "benign" / "sample.bin"
    bin_path.parent.mkdir(parents=True)
    bin_path.write_bytes(b"abc")

    out = get_output_path(bin_path, dataset_root / "pcap_data", dataset_root / "image_data")

    assert out == dataset_root / "image_data" / "Train" / "benign" / "sample.png"
```

- [ ] **Step 2: Run the image-path test to verify RED**

Run: `pytest tests/test_ssl_tls_rgb_image.py -v`
Expected: FAIL because the current signature depends on module-level globals

- [ ] **Step 3: Refactor the module to remove hard-coded dataset roots**

Update `/home/shuora/Traffic/FusionModel/src/ssl_tls_rgb_image.py` so the core path helpers accept explicit roots:

```python
def get_output_path(bin_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel_path = bin_path.relative_to(input_dir)
    return output_dir / rel_path.with_suffix(".png")
```

Also add CLI args for `--dataset_root`, `--input_dir`, and `--output_dir`, defaulting them from the processed dataset root.

- [ ] **Step 4: Re-run the image-path test to verify GREEN**

Run: `pytest tests/test_ssl_tls_rgb_image.py -v`
Expected: PASS

### Task 5: Remove Non-Attention Fusion Paths And Add Task-Oriented Dataset Resolution

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/src/fusion_common.py`
- Create: `/home/shuora/Traffic/FusionModel/tests/test_fusion_task_resolution.py`
- Modify: `/home/shuora/Traffic/FusionModel/findings.md`

- [ ] **Step 1: Write failing tests for task-root dataset resolution and attention-only model selection**

Create `/home/shuora/Traffic/FusionModel/tests/test_fusion_task_resolution.py` with tests like:

```python
from pathlib import Path

import pytest

from fusion_common import initialize_fusion_model, resolve_task_dataset_dirs


def test_resolve_task_dataset_dirs_uses_processed_root(tmp_path: Path):
    root = tmp_path / "ProcessedData" / "ustc_multiclass"
    for rel in [
        "image_data/Train/Geodo",
        "image_data/Test/Geodo",
        "pcap_data/Train/Geodo",
        "pcap_data/Test/Geodo",
    ]:
        (root / rel).mkdir(parents=True)

    train_img, train_pcap, test_img, test_pcap, resolved = resolve_task_dataset_dirs(tmp_path / "ProcessedData", "ustc_multiclass")

    assert resolved == "ustc_multiclass"
    assert train_img.endswith("image_data/Train")
    assert train_pcap.endswith("pcap_data/Train")


def test_initialize_fusion_model_rejects_removed_modes():
    with pytest.raises(ValueError):
        initialize_fusion_model(2, fusion_mode="concat")
    with pytest.raises(ValueError):
        initialize_fusion_model(2, fusion_mode="weighted")
```

- [ ] **Step 2: Run the fusion task-resolution tests to verify RED**

Run: `pytest tests/test_fusion_task_resolution.py -v`
Expected: FAIL because `resolve_task_dataset_dirs` does not exist and removed modes are still accepted

- [ ] **Step 3: Implement task-root dataset resolution in `fusion_common.py`**

Add a helper like:

```python
def resolve_task_dataset_dirs(processed_root: str | os.PathLike, task_name: str):
    task_root = Path(processed_root) / task_name
    train_image_dir = task_root / "image_data" / "Train"
    train_pcap_dir = task_root / "pcap_data" / "Train"
    test_image_dir = task_root / "image_data" / "Test"
    test_pcap_dir = task_root / "pcap_data" / "Test"
    ...
```

Wire the training path to use this helper instead of legacy grouped-layout discovery for the new task flow.

- [ ] **Step 4: Remove `concat` and `weighted` neural fusion branches**

Simplify `initialize_fusion_model` to accept only `attention`, for example:

```python
def initialize_fusion_model(num_classes: int, fusion_mode: str = "attention", attention_dim: int = 256) -> nn.Module:
    if fusion_mode != "attention":
        raise ValueError(f"unsupported fusion mode: {fusion_mode}")
    return AttentionFusionModel(num_classes=num_classes, attention_dim=attention_dim)
```

Delete the obsolete `FusionModel` branch if nothing else uses it.

- [ ] **Step 5: Re-run the fusion task-resolution tests to verify GREEN**

Run: `pytest tests/test_fusion_task_resolution.py -v`
Expected: PASS

- [ ] **Step 6: Update findings to reflect the reduced neural fusion surface**

Add this note to `/home/shuora/Traffic/FusionModel/findings.md`:

```md
- The neural training path will reject any fusion mode other than `attention`; stacking remains available only for the attention base model.
```

### Task 6: Narrow Entry Scripts To Attention Tasks And Verify End-To-End Selection

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/src/train_fusion_attention.py`
- Modify: `/home/shuora/Traffic/FusionModel/src/train_fusion_attention_stacking.py`
- Modify: `/home/shuora/Traffic/FusionModel/src/run_all_modes.py`
- Delete: `/home/shuora/Traffic/FusionModel/src/train_fusion_concat.py`
- Delete: `/home/shuora/Traffic/FusionModel/src/train_fusion_concat_stacking.py`
- Delete: `/home/shuora/Traffic/FusionModel/src/train_fusion_concat_all_ensembles.py`
- Delete: `/home/shuora/Traffic/FusionModel/src/train_fusion_weighted.py`
- Delete: `/home/shuora/Traffic/FusionModel/src/train_fusion_weighted_stacking.py`
- Create: `/home/shuora/Traffic/FusionModel/tests/test_run_all_modes.py`
- Modify: `/home/shuora/Traffic/FusionModel/progress.md`

- [ ] **Step 1: Write failing tests for task-oriented CLI mode selection**

Create `/home/shuora/Traffic/FusionModel/tests/test_run_all_modes.py` with tests like:

```python
import argparse

from run_all_modes import build_parser


def test_parser_only_exposes_attention_modes():
    parser = build_parser()
    mode_action = next(action for action in parser._actions if action.dest == "mode")
    assert sorted(mode_action.choices) == ["all", "attention", "attention_stacking"]
```

- [ ] **Step 2: Run the CLI test to verify RED**

Run: `pytest tests/test_run_all_modes.py -v`
Expected: FAIL because the parser still exposes concat and weighted modes or lacks `build_parser`

- [ ] **Step 3: Narrow the entrypoint surface to attention-only runs**

Implement a parser helper in `/home/shuora/Traffic/FusionModel/src/run_all_modes.py` like:

```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(...)
    p.add_argument("--mode", choices=["attention", "attention_stacking", "all"], default="all")
    p.add_argument("--task_name", required=True)
    return p
```

Update the runner logic so `all` means both attention runners for the selected task only.

- [ ] **Step 4: Remove obsolete concat/weighted entry scripts from the tree**

Run: `git rm src/train_fusion_concat.py src/train_fusion_concat_stacking.py src/train_fusion_concat_all_ensembles.py src/train_fusion_weighted.py src/train_fusion_weighted_stacking.py`
Expected: git stages deletion of the obsolete entry scripts

- [ ] **Step 5: Re-run the CLI test to verify GREEN**

Run: `pytest tests/test_run_all_modes.py -v`
Expected: PASS

- [ ] **Step 6: Run a focused regression subset covering all new task-facing modules**

Run: `pytest tests/test_task_config.py tests/test_split_data_tasks.py tests/test_ssl_tls_rgb_image.py tests/test_fusion_task_resolution.py tests/test_run_all_modes.py -v`
Expected: PASS

- [ ] **Step 7: Record the implementation completion checkpoint**

Append this line to `/home/shuora/Traffic/FusionModel/progress.md`:

```md
- Attention-only task-oriented training path and SourceData preprocessing refactor implemented; verification in progress.
```
