# Cross-Attention Stabilization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the cross-attention fusion model so stage1 binary no longer collapses to the majority class, and add reusable early stopping support to training entry points.

**Architecture:** Keep the current bidirectional token fusion encoder, but restore auxiliary supervision to pre-fusion backbone pooled features and add pooled-feature shortcuts into the fusion head path. Add a generic early stopping state machine in `src.train`, keyed off the existing `--best-metric`, and thread the new CLI parameter through `stage1_binary`.

**Tech Stack:** Python, PyTorch, pytest, existing train/evaluate pipeline

---

### Task 1: Lock model regression behavior with failing tests

**Files:**
- Modify: `tests/models/test_fusion_model.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] **Step 1: Replace the old aux-head expectation with a failing pre-fusion test**

Replace the existing “aux heads use fused context” expectation with a test named:
- `test_mobilevit_etbert_fusion_model_aux_heads_use_prefusion_pooled_features`

The test should replace image/text backbones and fusion encoder with deterministic dummies, then assert:
- `head_img` receives the pre-fusion pooled image feature
- `head_tls` receives the pre-fusion pooled text feature
- these inputs stay different from fused context values

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/models/test_fusion_model.py -k aux_heads_use_prefusion_pooled_features`
Expected: FAIL because current model feeds fused context into aux heads

- [ ] **Step 3: Write the failing test for fusion shortcut**

Add a test named:
- `test_mobilevit_etbert_fusion_model_fusion_head_keeps_prefusion_shortcut`

Configure:
- pooled features with one known value
- fused contexts with a different value
- `fusion_proj`/head recording modules

Assert that the fusion path input still contains pooled information and is not only the fused context pair.

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest -q tests/models/test_fusion_model.py -k fusion_head_keeps_prefusion_shortcut`
Expected: FAIL because current fusion path only concatenates fused contexts and `fusion_proj` still assumes `hidden_dim * 2` input

- [ ] **Step 5: Re-run warmup bypass coverage to lock non-fusion behavior**

Run: `pytest -q tests/models/test_fusion_model.py -k warmup_bypasses_fusion_encoder`
Expected: PASS before implementation, and remains PASS after implementation

- [ ] **Step 6: Commit**

```bash
git add tests/models/test_fusion_model.py
git commit -m "test: 固定 fusion 模型稳定性约束"
```

### Task 2: Implement stabilized fusion model

**Files:**
- Modify: `src/models/fusion_model.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] **Step 1: Implement minimal model change**

Update `MobileViTETBertFusionClassifier.forward(...)` so that:
- `img_pooled_pre` and `txt_pooled_pre` are preserved from backbone outputs
- `head_img` uses `img_pooled_pre`
- `head_tls` uses `txt_pooled_pre`
- fusion head input combines fused contexts and pooled shortcuts
- `fusion_proj` input dimension is updated from `hidden_dim * 2` to match the new concatenated input width
- external output contract remains unchanged:
  - `logits_fuse/logits_img/logits_tls`
  - `img_tokens/txt_tokens` when `return_features=True`

- [ ] **Step 2: Run focused model tests**

Run: `pytest -q tests/models/test_fusion_model.py -k 'prefusion or fusion_head_keeps_prefusion_shortcut or warmup_bypasses_fusion_encoder or can_optionally_return_debug_tokens or forward_shapes_without_gate'`
Expected: PASS

- [ ] **Step 3: Refactor names only if needed**

Keep helper variable names explicit (`img_pooled_pre`, `txt_pooled_pre`, etc.) and avoid extra architectural churn.

- [ ] **Step 4: Commit**

```bash
git add src/models/fusion_model.py tests/models/test_fusion_model.py
git commit -m "fix: 稳定 cross-attention 融合路径"
```

### Task 3: Add early stopping tests first

**Files:**
- Modify: `tests/pipeline/test_train_eval_report.py`
- Modify: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Write the failing train early-stopping trigger test**

Add a test that stubs:
- data loading
- model forward
- validation metric progression

Assert:
- training exits before configured total epochs when `--early-stopping-patience 2`
- `metrics.csv` contains only executed epochs
- `best.ckpt` exists and its saved `epoch` / `best_metric_value` correspond to the best epoch under the configured `--best-metric`
- `train.log` contains `early_stopping_triggered`
- `config.yaml` contains `early_stopping_patience: 2`

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k early_stopping_triggered`
Expected: FAIL because `src.train` has no early stopping yet

- [ ] **Step 3: Write the failing best-metric binding test**

Add a test that makes `val_acc` and `val_macro_f1` diverge across epochs, then asserts early stopping follows the configured `--best-metric` rather than a hard-coded metric.

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k early_stopping_respects_best_metric`
Expected: FAIL because current training loop has no early stopping state machine

- [ ] **Step 5: Write the failing plateau/tie semantics test**

Add a test asserting that when `current_best_value` plateaus exactly at the previous best:
- patience still increments
- tie does not reset the counter
- training stops once the configured patience is exhausted

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k early_stopping_tie_does_not_reset_patience`
Expected: FAIL because current training loop has no early stopping state machine

- [ ] **Step 7: Write the failing “disabled by default” test**

Add a test asserting that when `--early-stopping-patience` is omitted or `0`, training does not stop early.

- [ ] **Step 8: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_train_eval_report.py -k early_stopping_disabled_by_default`
Expected: FAIL because the option does not exist yet

- [ ] **Step 9: Write the failing stage1 protocol passthrough test**

Add a test asserting `src.experiments.stage1_binary --execute` forwards `--early-stopping-patience` into `train_main(...)`.

- [ ] **Step 10: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_protocol_execution.py -k early_stopping`
Expected: FAIL because the CLI argument is not threaded through yet

- [ ] **Step 11: Commit**

```bash
git add tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py
git commit -m "test: 固定 early stopping 行为"
```

### Task 4: Implement early stopping in train and protocol passthrough

**Files:**
- Modify: `src/train.py`
- Modify: `src/experiments/stage1_binary.py`
- Test: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Add CLI/config support**

In `src.train.py`:
- add `--early-stopping-patience`
- default it to `0`
- persist it in `config.yaml`

In `src.experiments.stage1_binary.py`:
- add the same CLI option
- pass it to `train_main(...)`

- [ ] **Step 2: Implement state machine**

In `src.train.py`, after each epoch:
- compare `current_best_value` to `best_value`
- track `epochs_without_improvement`
- treat `current_best_value > best_value` as the only improvement case
- when patience is reached, log an `early_stopping_triggered` event and break

- [ ] **Step 3: Run focused pipeline tests**

Run: `pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py -k 'early_stopping or forwards_early_stopping'`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/train.py src/experiments/stage1_binary.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py
git commit -m "feat: 增加通用 early stopping"
```

### Task 5: Run targeted regression and sync docs

**Files:**
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Run regression suite**

Run: `pytest -q tests/models/test_fusion_model.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py`
Expected: PASS

- [ ] **Step 2: Record outcome in planning files**

Document:
- what changed in fusion supervision
- how early stopping works
- what tests were run

- [ ] **Step 3: Commit**

```bash
git add docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 同步 cross-attention 修复与早停记录"
```
