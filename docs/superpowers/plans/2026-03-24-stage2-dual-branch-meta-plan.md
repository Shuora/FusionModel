# Stage2 Dual-Branch Meta-Enhancement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `stage2` high-score path that keeps dual-branch attention fusion as the primary model, then adds a `stacking`-based meta-classifier to push `MTA / MFCP / USTC-TFC2016` toward `test top1 >= 0.96` without disturbing the existing `stage1` high-score line.

**Architecture:** Keep the current fusion model as Level 1, but make it export a stable meta-feature contract. Rebuild `stacking` around exported OOF/meta artifacts instead of ad hoc retraining, then upgrade `stage2_multiclass` from a single-stage wrapper into a `fusion -> meta-classifier -> optional router/moe` protocol runner. Preserve `moe` as an extensible Level 3 hook, not the first-class path.

**Tech Stack:** Python, PyTorch, NumPy, pandas, scikit-learn, xgboost, pytest, existing `train/evaluate/report` pipeline

---

## File Structure

### Existing files to extend

- `src/models/fusion_model.py`
  - Keep Level 1 dual-branch attention fusion as the primary classifier.
  - Expose any additional lightweight fusion summaries needed by Level 2.
- `src/train.py`
  - Keep Level 1 training stable.
  - Add any needed dispatch/config plumbing for the new `stage2` meta path.
- `src/stacking.py`
  - Convert from “retrain base models inside stacking” to “consume exported fusion/meta artifacts”.
- `src/moe.py`
  - Reuse the same shared meta-feature schema.
  - Keep as optional Level 3 enhancement hook.
- `src/experiments/stage2_multiclass.py`
  - Upgrade from a thin one-stage launcher to the new multi-step `stage2` protocol runner.
  - Own fold/OOF orchestration for Level 2 artifact generation.
- `src/report.py`
  - Make final report selection meta-aware so `stage2` reports the true final metric source.
- `tests/pipeline/test_protocol_execution.py`
  - Lock the new `stage2` orchestration behavior.
- `tests/pipeline/test_stacking_pipeline.py`
  - Lock exported meta artifacts and stacking behavior.
- `tests/pipeline/test_moe_pipeline.py`
  - Keep Level 3 hook aligned with the shared meta-feature contract.
- `tests/pipeline/test_train_stage_dispatch.py`
  - Lock new dispatch semantics and argument forwarding.
- `docs/commands/session-full-experiments.md`
  - Update runnable commands after implementation.

### New files to create

- `src/meta_features.py`
  - Single source of truth for Level 2 / Level 3 feature extraction and artifact schema.
- `tests/pipeline/test_meta_features.py`
  - Small focused tests for the meta-feature contract if the new helper module is introduced.

## Task 1: Lock the Level 1 -> Level 2 meta-feature contract in tests

**Files:**
- Create: `tests/pipeline/test_meta_features.py`
- Modify: `tests/pipeline/test_stacking_pipeline.py`
- Modify: `tests/pipeline/test_moe_pipeline.py`
- Modify: `tests/models/test_fusion_model.py`
- Test: `tests/pipeline/test_meta_features.py`
- Test: `tests/pipeline/test_stacking_pipeline.py`
- Test: `tests/pipeline/test_moe_pipeline.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] **Step 1: Write the failing fusion-summary export test**

Add a model test that asserts the fusion model can expose the lightweight data needed for meta-classification:
- three logits (`fusion / image / text`)
- lightweight summary stats
- optional token features still remain opt-in rather than always-on

- [ ] **Step 2: Run the focused model test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py -k meta_feature_summary`
Expected: FAIL because the summary contract is not locked yet

- [ ] **Step 3: Write the failing shared meta-feature helper test**

Create `tests/pipeline/test_meta_features.py` asserting a helper can build:
- logits block
- confidence block
- agreement block
- lightweight summary block
with deterministic output shape

- [ ] **Step 4: Run the helper test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_meta_features.py`
Expected: FAIL because `src/meta_features.py` does not exist yet

- [ ] **Step 5: Write the failing stacking-artifact schema test**

Extend `tests/pipeline/test_stacking_pipeline.py` so the pipeline must persist:
- OOF meta train artifacts
- meta test artifacts
- feature names / schema metadata
- metrics file for the final meta-classifier

- [ ] **Step 6: Run the stacking test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stacking_pipeline.py -k meta_artifacts`
Expected: FAIL because the new schema files are not emitted yet

- [ ] **Step 7: Write the failing MoE shared-feature test**

Extend `tests/pipeline/test_moe_pipeline.py` to assert MoE reads the shared meta-feature schema instead of maintaining a private incompatible feature definition.

- [ ] **Step 8: Run the MoE test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_moe_pipeline.py -k shared_meta`
Expected: FAIL

- [ ] **Step 9: Commit**

```bash
git add tests/models/test_fusion_model.py tests/pipeline/test_meta_features.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py
git commit -m "test: 固定 stage2 元特征契约"
```

## Task 2: Implement the shared meta-feature module and fusion summaries

**Files:**
- Create: `src/meta_features.py`
- Modify: `src/models/fusion_model.py`
- Modify: `src/stacking.py`
- Modify: `src/moe.py`
- Test: `tests/models/test_fusion_model.py`
- Test: `tests/pipeline/test_meta_features.py`
- Test: `tests/pipeline/test_stacking_pipeline.py`
- Test: `tests/pipeline/test_moe_pipeline.py`

- [ ] **Step 1: Create `src/meta_features.py`**

Implement a shared helper layer for:
- converting model outputs to stable Level 2 feature blocks
- returning feature names alongside arrays
- reusing the same feature contract in stacking and MoE

- [ ] **Step 2: Extend `fusion_model` with lightweight summary outputs**

Expose only lightweight summaries needed by Level 2, such as:
- confidence-oriented summary values
- optional fusion norms / attention aggregates
Do not make full token sequences mandatory in standard forward paths.

- [ ] **Step 3: Replace private stacking feature logic with the shared helper**

Remove duplicated feature logic from `src/stacking.py` and switch it to the new shared helper.

- [ ] **Step 4: Replace private MoE router feature logic with the shared helper**

Make `src/moe.py` consume the same shared feature schema instead of its current private handcrafted feature block.

- [ ] **Step 5: Run the focused feature tests**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py tests/pipeline/test_meta_features.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py -k 'meta_feature or shared_meta or summary or meta_artifacts'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/meta_features.py src/models/fusion_model.py src/stacking.py src/moe.py tests/models/test_fusion_model.py tests/pipeline/test_meta_features.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py
git commit -m "feat: 统一 stage2 元特征导出"
```

## Task 3: Rebuild `stacking` as the Level 2 meta-classifier

**Files:**
- Modify: `src/stacking.py`
- Modify: `tests/pipeline/test_stacking_pipeline.py`
- Test: `tests/pipeline/test_stacking_pipeline.py`

- [ ] **Step 1: Write the failing OOF-only training test**

Add a test asserting the Level 2 meta-classifier is trained on exported OOF/meta artifacts rather than hidden private retraining behavior that bypasses the artifact boundary.

- [ ] **Step 2: Run the test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stacking_pipeline.py -k oof_only`
Expected: FAIL

- [ ] **Step 3: Write the failing final-metric-source test**

Add a test asserting the stacking output writes a final metrics file that should be treated as the final `stage2` result rather than just an auxiliary artifact.

- [ ] **Step 4: Run the test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stacking_pipeline.py -k final_metric_source`
Expected: FAIL

- [ ] **Step 5: Implement the new stacking flow**

Refactor `src/stacking.py` so it:
- loads shared meta features
- writes explicit schema-aware OOF artifacts
- trains the Level 2 meta-classifier on those artifacts
- emits final metrics plus model artifacts in a stable layout

- [ ] **Step 6: Make the Feature Dump / OOF Generator explicit**

Implement the OOF/meta export boundary as a first-class part of the stacking flow:
- define `stacking.py` as the artifact consumer rather than the fold orchestrator
- require train/val meta samples to come from runner-managed KFold OOF generation
- reserve a parameterized path for future holdout-only generation if needed
- persist schema version, feature names, fold ids, and split provenance with every dump

- [ ] **Step 7: Run the stacking suite**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_stacking_pipeline.py`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/stacking.py tests/pipeline/test_stacking_pipeline.py
git commit -m "feat: 重构 stage2 二级 stacking 分类器"
```

## Task 4: Lock the new `stage2` protocol runner in tests

**Files:**
- Modify: `tests/pipeline/test_protocol_execution.py`
- Modify: `tests/pipeline/test_train_stage_dispatch.py`
- Test: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_train_stage_dispatch.py`

- [ ] **Step 1: Write the failing `fusion -> stacking` orchestration test**

Add a protocol execution test asserting `stage2_multiclass --execute` can:
- train the Level 1 fusion run first
- generate Level 2 meta artifacts
- train the stacking meta-classifier second
- keep all outputs under the same dataset run family

- [ ] **Step 2: Run the protocol test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k stage2_fusion_then_stacking`
Expected: FAIL

- [ ] **Step 3: Write the failing dispatch-argument test**

Add a dispatch test asserting `train.py` and/or the stage2 runner can forward the new meta-classifier options explicitly rather than relying on hidden defaults.

- [ ] **Step 4: Run the dispatch test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_train_stage_dispatch.py -k meta_classifier`
Expected: FAIL

- [ ] **Step 5: Write the failing stage2 summary test**

Add a test asserting `stage2_execution_summary.json` records both the Level 1 run path and the final meta-enhanced metric source for each dataset task.

- [ ] **Step 6: Run the summary test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k stage2_execution_summary`
Expected: FAIL

- [ ] **Step 7: Commit**

```bash
git add tests/pipeline/test_protocol_execution.py tests/pipeline/test_train_stage_dispatch.py
git commit -m "test: 固定 stage2 两级编排"
```

## Task 5: Implement the independent `stage2` fusion-first protocol

**Files:**
- Modify: `src/experiments/stage2_multiclass.py`
- Modify: `src/train.py`
- Modify: `src/report.py`
- Modify: `tests/pipeline/test_protocol_execution.py`
- Modify: `tests/pipeline/test_train_stage_dispatch.py`
- Modify: `tests/pipeline/test_stacking_pipeline.py`
- Test: `tests/pipeline/test_protocol_execution.py`
- Test: `tests/pipeline/test_train_stage_dispatch.py`
- Test: `tests/pipeline/test_stacking_pipeline.py`

- [ ] **Step 1: Add explicit Level 2 protocol options**

Introduce runner arguments for the `stage2` path such as:
- enabling/disabling meta-classifier
- choosing the Level 2 implementation
- keeping Level 3 `router/moe` optional

Do not overload the old `--stage stacking` meaning in a way that breaks existing expectations silently.

- [ ] **Step 2: Implement `fusion -> meta-classifier` execution order**

Make `src/experiments/stage2_multiclass.py` run:
1. Level 1 fusion training
2. runner-managed KFold OOF artifact generation for Level 2
3. Level 2 stacking meta-classifier over dumped artifacts
4. final report / summary writing

- [ ] **Step 3: Make OOF ownership and anti-leakage rules explicit**

Implement the OOF Generator in the runner layer rather than hiding it inside stacking:
- the runner owns fold splits and fold-level fusion training
- stacking consumes dumped artifacts only
- tests must be able to assert that Level 2 is not trained on in-sample single-run features

- [ ] **Step 4: Make report discovery final-metric aware**

Update `src/report.py` so a run with Level 2 artifacts reports the final Level 2 metric source instead of blindly preferring `eval_test.json` from Level 1.

- [ ] **Step 5: Keep `train.py` dispatches coherent**

Make sure `train.py` dispatch behavior remains understandable and testable:
- old stacking/moe unit behavior stays valid where intended
- new stage2 meta protocol uses explicit orchestration rather than hidden side effects

- [ ] **Step 6: Run the focused orchestration tests**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_stacking_pipeline.py -k 'stage2_fusion_then_stacking or meta_classifier or metric_source or execution_summary'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/experiments/stage2_multiclass.py src/train.py src/report.py tests/pipeline/test_protocol_execution.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_stacking_pipeline.py
git commit -m "feat: 实现 stage2 双级执行协议"
```

## Task 6: Add the optional Level 3 router/MoE hook without making it primary

**Files:**
- Modify: `src/moe.py`
- Modify: `src/report.py`
- Modify: `tests/pipeline/test_moe_pipeline.py`
- Test: `tests/pipeline/test_moe_pipeline.py`

- [ ] **Step 1: Write the failing Level 3 hook test**

Add a test asserting the optional Level 3 enhancer:
- consumes shared meta features
- writes its own artifacts cleanly
- does not replace Level 2 as the default path

- [ ] **Step 2: Run the test to confirm RED**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_moe_pipeline.py -k level3`
Expected: FAIL

- [ ] **Step 3: Implement the optional Level 3 path**

Keep `moe.py` aligned with the shared meta-feature contract and ensure its outputs are clearly treated as an optional post-`stacking` or alternative post-Level-2 enhancer.

- [ ] **Step 4: Run the MoE suite**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_moe_pipeline.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moe.py src/report.py tests/pipeline/test_moe_pipeline.py
git commit -m "feat: 预留 stage2 三级增强挂点"
```

## Task 7: Update runnable docs and planning records

**Files:**
- Modify: `docs/commands/session-full-experiments.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Update the stage2 command docs**

Document the new runnable path for:
- Level 1 fusion-only baseline
- `fusion + stacking meta-classifier`
- optional Level 3 enhancer

- [ ] **Step 2: Record the implementation boundary and expected run artifacts**

Update planning files with:
- artifact layout
- metric-source interpretation
- which path is `v1` vs optional `v2`

- [ ] **Step 3: Commit**

```bash
git add docs/commands/session-full-experiments.md docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 更新 stage2 双级方案文档"
```

## Task 8: Final regression pass before execution baselines

**Files:**
- Modify: `docs/planning-with-files/progress.md`
- Test: `tests/models/test_fusion_model.py`
- Test: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: Run the full targeted regression suite**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py tests/pipeline/test_meta_features.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_protocol_execution.py tests/pipeline/test_train_eval_report.py`
Expected: PASS

- [ ] **Step 2: Run the stage1 compatibility regression**

Run: `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/models/test_fusion_model.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_protocol_execution.py -k 'stage1 or warmup or fusion_mode or report'`
Expected: PASS, confirming the new stage2 path did not silently regress stage1 behavior

- [ ] **Step 3: Record the regression result**

Write the command and pass/fail result into `docs/planning-with-files/progress.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/planning-with-files/progress.md
git commit -m "test: 完成 stage2 双级方案回归验证"
```

## Task 9: Run acceptance baselines against the spec success criteria

**Files:**
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: Run the Level 1 fusion baseline on all three datasets**

Run the new `stage2` protocol in Level 1-only mode for:
- `MTA`
- `MFCP`
- `USTC-TFC2016`

Record each dataset’s final `test top1 / accuracy` as the pre-meta baseline.

- [ ] **Step 2: Run the Level 2 `fusion + stacking` path on all three datasets**

Run the accepted `v1` path for:
- `MTA`
- `MFCP`
- `USTC-TFC2016`

Save the final outputs so the report/summary clearly distinguishes:
- Level 1 baseline
- Level 2 final result

- [ ] **Step 3: If needed, run the optional Level 3 path**

Only if Level 2 still misses the target on any dataset:
- run the optional router/MoE enhancement
- record whether it improves or regresses final `test top1`

- [ ] **Step 4: Write the acceptance table**

Record in planning docs and summary artifacts:
- dataset name
- Level 1 `test top1`
- Level 2 `test top1`
- optional Level 3 `test top1`
- whether the `>= 0.96` success criterion is met

- [ ] **Step 5: Commit**

```bash
git add docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: 记录 stage2 验收基线结果"
```
