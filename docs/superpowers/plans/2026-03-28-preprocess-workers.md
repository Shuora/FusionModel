# Preprocess Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fine-grained filtering, resumable post-processing, progress logging, and multiprocessing to dataset preparation.

**Architecture:** Keep SplitCap collection logic mostly intact, then centralize manifest and duplicate decisions in the parent process before dispatching heavy post-processing work to worker processes. Resume behavior is driven by final output files rather than ad hoc state.

**Tech Stack:** Python 3.9, pandas, multiprocessing, pytest, existing FusionModel data utilities

---

### Task 1: Cover new CLI and filtering behavior

**Files:**
- Modify: `tests/test_splitcap_cleaning_and_manifest.py`
- Modify: `scripts/prepare_dataset.py`

- [ ] **Step 1: Write failing tests for fine-grained include-path filtering**
- [ ] **Step 2: Run the targeted tests and verify they fail for missing filtering support**
- [ ] **Step 3: Implement the minimal parser and discovery filtering changes**
- [ ] **Step 4: Re-run the targeted tests and verify they pass**

### Task 2: Cover resumable post-processing behavior

**Files:**
- Modify: `tests/test_splitcap_cleaning_and_manifest.py`
- Modify: `scripts/prepare_dataset.py`

- [ ] **Step 1: Write failing tests for skipping existing `sessions_clean` and `cache` outputs**
- [ ] **Step 2: Run the targeted tests and verify they fail for missing resume behavior**
- [ ] **Step 3: Implement minimal sample planning and skip checks**
- [ ] **Step 4: Re-run the targeted tests and verify they pass**

### Task 3: Add multiprocessing and progress reporting

**Files:**
- Modify: `tests/test_splitcap_cleaning_and_manifest.py`
- Modify: `scripts/prepare_dataset.py`
- Modify: `scripts/run_prepare_binary.sh`
- Modify: `scripts/run_prepare_multiclass.sh`

- [ ] **Step 1: Write failing tests for worker-count parsing and progress configuration**
- [ ] **Step 2: Run the targeted tests and verify they fail for missing arguments and orchestration**
- [ ] **Step 3: Implement worker orchestration and progress logging**
- [ ] **Step 4: Re-run targeted tests and then the full related test file**

### Task 4: Verify and document behavior

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Update execution docs with actual outcomes**
- [ ] **Step 2: Run final verification commands**
- [ ] **Step 3: Summarize resulting CLI usage and tradeoffs**

### Task 5: Surface planning-stage progress

**Files:**
- Modify: `scripts/prepare_dataset.py`
- Modify: `tests/test_splitcap_cleaning_and_manifest.py`

- [ ] **Step 1: Write a failing test that captures `[plan]` heartbeat output during `prepare_cached_rows(...)`**
- [ ] **Step 2: Run the targeted test and verify current code stays silent until planning completes**
- [ ] **Step 3: Implement lightweight planning logs without changing duplicate-filter semantics**
- [ ] **Step 4: Re-run the targeted test and then the full related test file**
