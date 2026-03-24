# Stage1 High-Score Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign stage1 binary so the new `score_optimized` path can realistically target `98%+` holdout accuracy through coordinated protocol, training, and fusion-role changes.

**Architecture:** Add a new high-score-oriented `stage1_binary` protocol mode with explicit `train/val/test` outputs, then build a two-stage `warmup -> fusion` training flow on top of it. Keep single-branch pooled features as the stable base, and treat cross-attention fusion as a residual enhancer instead of the only decision path. Every layer change must be isolated with ablation-friendly verification.

**Tech Stack:** Python, PyTorch, pandas, pytest, existing train/evaluate/report pipeline

---

### Task 1: Lock the new `score_optimized` protocol in tests

**Files:**
- Modify: `tests/pipeline/test_stage1_binary_protocol.py`
- Modify: `tests/pipeline/test_pipeline_data_protocol.py`
- Modify: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_stage1_binary_protocol.py`
- Test: `tests/pipeline/test_pipeline_data_protocol.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Write the failing protocol-balance test**

Add a test named like `test_stage1_manifest_score_optimized_outputs_explicit_train_val_test_balanced_binary_distribution` that asserts:
- `protocol_mode="score_optimized"` is accepted
- output contains explicit `train`, `val`, `test`
- binary labels are near-balanced per split
- no runtime-derived validation is needed later

- [ ] **Step 2: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage1_binary_protocol.py -k score_optimized_outputs_explicit_train_val_test`
Expected: FAIL because `score_optimized` does not exist yet

- [ ] **Step 3: Write the failing dataset-balance test**

Add a test asserting `score_optimized` does not let one dataset dominate the split composition beyond the threshold chosen in the spec implementation.

- [ ] **Step 4: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage1_binary_protocol.py -k score_optimized_dataset_balance`
Expected: FAIL

- [ ] **Step 5: Write the failing execute-path protocol propagation test**

Add a `test_protocol_execution.py` case asserting `stage1_binary --execute --protocol-mode score_optimized` keeps the explicit manifest and does not silently fall back to train-derived validation.

- [ ] **Step 6: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k score_optimized`
Expected: FAIL

- [ ] **Step 7: Write the failing pipeline-data split propagation test**

Add a `tests/pipeline/test_pipeline_data_protocol.py` case asserting an explicit `val` split emitted by `score_optimized` survives through `src.pipeline_data` without being collapsed back into train-derived validation behavior.

- [ ] **Step 8: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_pipeline_data_protocol.py -k explicit_val_split`
Expected: FAIL

- [ ] **Step 9: Commit**

```bash
git add tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_protocol_execution.py
git commit -m "test: 固定 stage1 高分协议约束"
```

### Task 2: Implement the `score_optimized` protocol mode

**Files:**
- Modify: `src/experiments/stage1_binary.py`
- Modify: `src/pipeline_data.py`
- Test: `tests/pipeline/test_stage1_binary_protocol.py`
- Test: `tests/pipeline/test_pipeline_data_protocol.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Add `score_optimized` to CLI and manifest builder**

Implement a separate branch from `paper_balanced` / `paper_strict` so the new mode is explicit and does not change old behaviors.

- [ ] **Step 2: Implement split generation rules**

Add deterministic logic that:
- writes explicit `train`, `val`, `test`
- enforces binary balance targets
- constrains dataset composition
- remains reproducible from stable sorting

- [ ] **Step 3: Run focused protocol tests**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_protocol_execution.py -k 'score_optimized or explicit_val_split'`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/experiments/stage1_binary.py src/pipeline_data.py tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_protocol_execution.py
git commit -m "feat: 新增 stage1 高分协议模式"
```

### Task 3: Run and record the protocol-only baseline

**Files:**
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Run protocol-only baseline**

Run the first real baseline with:
- `protocol_mode=score_optimized`
- current stable train path
- no new two-stage orchestration yet

Expected outputs:
- explicit protocol summary
- train/val/test class distribution
- dataset composition summary
- best checkpoint metrics
- final holdout test report

- [ ] **Step 2: Record baseline metrics**

Write down:
- `val_acc`
- `test accuracy`
- `test_macro_f1`
- whether the run already satisfies any success criteria

- [ ] **Step 3: Commit**

```bash
git add docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 记录高分协议基线"
```

### Task 4: Lock two-stage training orchestration in tests

**Files:**
- Modify: `tests/pipeline/test_protocol_execution.py`
- Modify: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_train_eval_report.py`

- [ ] **Step 1: Write the failing warmup-then-fusion orchestration test**

Add a `stage1_binary --execute` test asserting the high-score path can run:
- warmup training first
- fusion training second
- both under the same run family / explicit handoff

- [ ] **Step 2: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k warmup_then_fusion`
Expected: FAIL

- [ ] **Step 3: Write the failing train-resume / stage handoff test**

Add a train-side test asserting fusion stage can load a warmup checkpoint and preserve the intended config contract.

- [ ] **Step 4: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_train_eval_report.py -k warmup_checkpoint_handoff`
Expected: FAIL

- [ ] **Step 5: Commit**

```bash
git add tests/pipeline/test_protocol_execution.py tests/pipeline/test_train_eval_report.py
git commit -m "test: 固定两阶段训练编排"
```

### Task 5: Implement high-score training orchestration and stability controls

**Files:**
- Modify: `src/train.py`
- Modify: `src/experiments/stage1_binary.py`
- Test: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Add explicit warmup-to-fusion handoff support**

Implement the minimum surface needed for:
- warmup checkpoint loading
- fusion-stage resume/init
- stable run naming / directory ownership

- [ ] **Step 2: Add training stability controls**

Implement these concrete controls from the spec:
- lower-risk lr path via scheduler support
- explicit `class_weight` support for high-score binary mode
- explicit freeze/unfreeze control for warmup -> fusion transition
- existing early stopping reuse
- keep gradient safety intact

- [ ] **Step 3: Write RED tests for selection redesign**

Add train-side tests asserting:
- checkpoint selection can use `val_acc + val_macro_f1 + threshold stability`
- unstable thresholds lose ties against equally accurate but stable epochs

- [ ] **Step 4: Run the tests and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_train_eval_report.py -k 'checkpoint_selection or threshold_stability'`
Expected: FAIL

- [ ] **Step 5: Implement checkpoint selection redesign**

Make the high-score path’s best-checkpoint logic explicit and testable rather than buried in generic best-metric handling.

- [ ] **Step 6: Run focused orchestration tests**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py -k 'warmup_then_fusion or handoff or early_stopping or checkpoint_selection or threshold_stability'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/train.py src/experiments/stage1_binary.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py
git commit -m "feat: 实现 stage1 两阶段高分训练流程"
```

### Task 6: Run and record the protocol + two-stage training baseline

**Files:**
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Run the second real baseline**

Run the first full high-score training baseline with:
- `protocol_mode=score_optimized`
- explicit `warmup -> fusion`
- no fusion-role redesign yet

- [ ] **Step 2: Record baseline metrics**

Write down:
- delta vs protocol-only baseline
- `val_acc`
- `test accuracy`
- `test_macro_f1`
- 3-epoch stability window

- [ ] **Step 3: Commit**

```bash
git add docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 记录两阶段训练基线"
```

### Task 7: Lock the new fusion role in model tests

**Files:**
- Modify: `tests/models/test_fusion_model.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] **Step 1: Write the failing image-dominant residual fusion test**

Add a test asserting the fusion path behaves like a residual enhancer:
- single-branch pooled features remain first-class inputs
- fusion augments rather than replaces them

- [ ] **Step 2: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py -k residual_enhancer`
Expected: FAIL

- [ ] **Step 3: Write the failing warmup/fusion compatibility regression**

Add a model-level regression asserting:
- warmup keeps branch heads stable
- fusion stage adds context without breaking the output contract

- [ ] **Step 4: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py -k warmup_fusion_role`
Expected: FAIL

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_fusion_model.py
git commit -m "test: 固定高分版 fusion 角色"
```

### Task 8: Implement the high-score fusion role redesign

**Files:**
- Modify: `src/models/fusion_model.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] **Step 1: Implement residual-enhancer fusion**

Keep:
- `img_pooled` as the strongest direct signal
- `txt_pooled` as auxiliary direct signal

Make fusion:
- consume token-level context
- combine with pooled shortcuts
- not become the only discriminative path

- [ ] **Step 2: Preserve output contract**

Ensure:
- `logits_fuse`
- `logits_img`
- `logits_tls`
- optional `img_tokens` / `txt_tokens`

all remain compatible with the existing pipeline.

- [ ] **Step 3: Run focused model tests**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/models/fusion_model.py tests/models/test_fusion_model.py
git commit -m "refactor: 重构高分版 fusion 角色"
```

### Task 9: Lock and enforce the test-holdout rule

**Files:**
- Modify: `tests/pipeline/test_protocol_execution.py`
- Modify: `src/experiments/stage1_binary.py`

- [ ] **Step 1: Write the failing holdout-rule test**

Add a test asserting the high-score execution path does not automatically re-run holdout test evaluation during intermediate baselines, and only runs test evaluation in the final explicitly requested evaluation step.

- [ ] **Step 2: Run the test and confirm RED**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k holdout_rule`
Expected: FAIL

- [ ] **Step 3: Implement explicit holdout-rule control**

Add the minimum CLI / execution guard needed so high-score protocol runs cannot silently turn test into a tuning loop.

- [ ] **Step 4: Run the test and confirm GREEN**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k holdout_rule`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/stage1_binary.py tests/pipeline/test_protocol_execution.py
git commit -m "feat: 固化高分方案 holdout 规则"
```

### Task 10: Add ablation-friendly verification and docs

**Files:**
- Modify: `docs/commands/session-full-experiments.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Document the new high-score mode**

Add docs for:
- `score_optimized`
- warmup -> fusion flow
- recommended command sequence
- test-holdout rule
- required output artifacts:
  - protocol summary
  - train/val/test class distribution
  - dataset composition summary
  - best checkpoint metrics
  - final test evaluation report

- [ ] **Step 2: Run the final targeted regression suite**

Run: `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py`
Expected: PASS

- [ ] **Step 3: Run the final full high-score experiment**

Run the final candidate with:
- `score_optimized`
- two-stage training
- fusion-role redesign
- holdout-rule-compliant evaluation

- [ ] **Step 4: Check success criteria against real metrics**

Record and verify:
- `val_acc >= 0.98`
- `test accuracy >= 0.98`
- `test_macro_f1 >= 0.97`
- 3-epoch stability window

If any one fails, mark the redesign incomplete and do not claim success.

- [ ] **Step 5: Run ablation-friendly experiment commands**

Run and record, in order:
1. protocol-only baseline on `score_optimized`
2. protocol + two-stage training baseline
3. protocol + two-stage training + fusion-role redesign

For each run, record:
- holdout `test accuracy`
- `test macro_f1`
- class distribution summary
- dataset composition summary
- best checkpoint epoch / metric

- [ ] **Step 4: Record ablation checkpoints in planning files**

Write down:
- protocol-only gain
- training-only gain
- fusion-role gain
- residual risks / next experiments

- [ ] **Step 7: Commit**

```bash
git add docs/commands/session-full-experiments.md docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 补齐 stage1 高分方案说明"
```
