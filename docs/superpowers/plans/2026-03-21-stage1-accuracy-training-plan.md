# Stage1 Accuracy Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve current stage1 binary test accuracy by making validation selection more stable and allowing binary runs to choose an accuracy-oriented checkpoint and decision threshold.

**Architecture:** Keep the existing train/evaluate/report pipeline intact, but tighten the training decision loop. Validation split generation becomes stratified, best checkpoint selection becomes configurable, and binary evaluation can consume a calibrated threshold saved by training.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing `src.train` / `src.evaluate` pipeline

---

### Task 1: Add failing tests for stratified validation and best-metric selection

**Files:**
- Modify: `tests/pipeline/test_train_eval_report.py`
- Modify: `src/train.py`

- [ ] **Step 1: Write the failing test**

Add tests that assert:
- derived validation split keeps both classes present for imbalanced binary labels
- `config.yaml` records `best_metric`
- `best.ckpt` metadata follows `val_acc` when `--best-metric val_acc` is passed

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k 'stratified or best_metric'`
Expected: FAIL because current train flow does not support these behaviors.

- [ ] **Step 3: Write minimal implementation**

Implement stratified split helper and configurable best-metric handling in `src/train.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k 'stratified or best_metric'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/pipeline/test_train_eval_report.py src/train.py
git commit -m "feat: support accuracy-oriented train selection"
```

### Task 2: Add failing tests for binary threshold calibration

**Files:**
- Modify: `tests/pipeline/test_train_eval_report.py`
- Modify: `src/train.py`
- Modify: `src/evaluate.py`

- [ ] **Step 1: Write the failing test**

Add tests that assert:
- binary training stores a calibrated decision threshold
- evaluation uses that threshold when present
- multiclass evaluation remains unchanged

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k 'threshold'`
Expected: FAIL because current train/evaluate flow does not persist or consume a calibrated threshold.

- [ ] **Step 3: Write minimal implementation**

Implement threshold search on validation logits, persist chosen threshold, and read it during evaluation.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k 'threshold'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/pipeline/test_train_eval_report.py src/train.py src/evaluate.py
git commit -m "feat: add binary decision threshold calibration"
```

### Task 3: Sync documentation and verify end-to-end behavior

**Files:**
- Modify: `docs/planning-with-files/task_plan.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`
- Modify: `docs/commands/session-full-experiments.md`

- [ ] **Step 1: Update documentation**

Document:
- new `--best-metric` option
- stratified validation behavior
- binary threshold calibration behavior

- [ ] **Step 2: Run focused regression**

Run: `pytest -q tests/pipeline/test_train_eval_report.py`
Expected: PASS

- [ ] **Step 3: Run syntax verification**

Run: `python -m py_compile src/train.py src/evaluate.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/planning-with-files/task_plan.md docs/planning-with-files/findings.md docs/planning-with-files/progress.md docs/commands/session-full-experiments.md src/train.py src/evaluate.py tests/pipeline/test_train_eval_report.py
git commit -m "docs: record accuracy-oriented training workflow"
```
