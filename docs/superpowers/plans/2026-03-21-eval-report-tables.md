# Evaluation Report Tables Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `evaluate/report` 补充 classification report artifact，并在 `report.md` 中直接渲染混淆矩阵与分类指标表格。

**Architecture:** 保持现有 `train -> evaluate -> report` 链路不变，只扩展 `evaluate` 的落盘产物与 `report` 的 Markdown 渲染。测试沿用现有 `tests/pipeline/test_train_eval_report.py` 的 smoke/fallback 场景，先写失败测试，再补最小实现。

**Tech Stack:** Python, pandas, sklearn, pytest, Markdown

---

### Task 1: Lock Expected Evaluation Artifacts

**Files:**
- Modify: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_train_eval_report.py`

- [ ] **Step 1: Write the failing test**

补充 `evaluate` smoke/fallback 断言，要求生成：
- `classification_report_<split>.csv`
- `classification_report_<split>.json`

并要求 `report.md` 包含：
- `## Confusion Matrix`
- `## Classification Report`

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/pipeline/test_train_eval_report.py -k 'smoke or fallback'`
Expected: FAIL，因为当前实现不会生成 classification report artifact，也不会把表格渲染进 `report.md`。

### Task 2: Emit Classification Report Artifacts

**Files:**
- Modify: `src/evaluate.py`
- Test: `tests/pipeline/test_train_eval_report.py`

- [ ] **Step 1: Write minimal implementation**

在 `evaluate.py` 中：
- 基于 `y_eval/pred` 生成 per-class precision/recall/f1/support
- 输出 `classification_report_<split>.csv`
- 输出 `classification_report_<split>.json`
- 保持现有 `eval_*.json` / confusion matrix artifact 不变

- [ ] **Step 2: Run targeted tests**

Run: `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/pipeline/test_train_eval_report.py -k 'smoke or fallback'`
Expected: PASS

### Task 3: Render Tables in Markdown Report

**Files:**
- Modify: `src/report.py`
- Test: `tests/pipeline/test_train_eval_report.py`

- [ ] **Step 1: Write minimal implementation**

在 `report.py` 中：
- 读取 `confusion_matrix_<split>.csv`
- 读取 `classification_report_<split>.csv`
- 将两者渲染到 `report.md` 的 Markdown 表格
- 保持 stacking/moe fallback 行为不变

- [ ] **Step 2: Run focused regression**

Run: `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/pipeline/test_train_eval_report.py`
Expected: PASS

### Task 4: Record Findings and Verify End State

**Files:**
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Update docs**

记录：
- 新增 classification report artifact
- `report.md` 现在直接显示 confusion matrix 与 classification report 表

- [ ] **Step 2: Final verification**

Run: `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py -k 'stage1_binary_execute_runs_train_eval_report or train_evaluate_report_smoke or evaluate_fallback_uses_effective_split_and_report_discovers_it'`
Expected: PASS
