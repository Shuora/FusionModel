# Preprocess Planning Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the pre-cache planning phase without changing payload-based deduplication semantics or output sample selection.

**Architecture:** Keep the parent process responsible for manifest order and first-seen duplicate decisions, but move expensive per-session payload extraction and fingerprint computation into a process pool. Consume worker results in manifest order so the kept/dropped sample set remains identical to the serial implementation.

**Tech Stack:** Python 3.9, concurrent.futures.ProcessPoolExecutor, pytest, scapy-based session byte extraction

---

### Task 1: Cover planning-worker orchestration with tests

**Files:**
- Modify: `tests/test_splitcap_cleaning_and_manifest.py`
- Modify: `scripts/prepare_dataset.py`

- [ ] **Step 1: Write a failing test that proves `prepare_cached_rows(...)` can consume payload inspection results through an executor-backed path while preserving record order.**
- [ ] **Step 2: Run the targeted test and verify current code still performs payload reads inline in the parent process.**
- [ ] **Step 3: Implement a top-level payload inspection worker and ordered result consumption in `prepare_cached_rows(...)`.**
- [ ] **Step 4: Re-run the targeted test and confirm it passes.**

### Task 2: Preserve logging and resume behavior

**Files:**
- Modify: `tests/test_splitcap_cleaning_and_manifest.py`
- Modify: `scripts/prepare_dataset.py`

- [ ] **Step 1: Extend tests so progress logging still reports `[plan]` heartbeats when planning work is parallelized.**
- [ ] **Step 2: Keep `clean_hits`, `cache_hits`, empty filtering, and duplicate filtering expectations unchanged.**
- [ ] **Step 3: Run the full related pytest file and confirm no regressions.**

### Task 3: Document the optimization

**Files:**
- Modify: `docs/superpowers/specs/2026-03-28-preprocess-workers-design.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Record the new planning-stage execution model and invariants.**
- [ ] **Step 2: Capture validation results and runtime tradeoffs.**
