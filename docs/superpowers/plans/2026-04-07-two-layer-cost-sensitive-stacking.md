# Two-Layer Cost-Sensitive Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `attention_stacking` to a two-layer, cost-sensitive stacking pipeline that improves minority-class recall and macro-F1 across all four tasks while preserving backward-compatible training commands.

**Architecture:** Keep the existing attention fusion base model training unchanged, then run Level-1 meta learners with OOF predictions and probability calibration, followed by a Level-2 cost-sensitive blender trained on stacked Level-1 signals and uncertainty features. Apply per-class threshold optimization on Level-2 OOF outputs with an objective that favors minority recall plus macro-F1, while keeping single-layer and soft-voting outputs as ablation baselines.

**Tech Stack:** Python 3, NumPy, scikit-learn metrics/utilities, existing `xgboost/lightgbm/catboost/mlp` meta learners in `src/fusion_common.py`, `unittest`.

---

### Task 1: Add CLI/kwargs plumbing for two-layer stacking controls

**Files:**
- Modify: `src/fusion_common.py`
- Modify: `src/train_fusion_attention_stacking.py`
- Test: `tests/test_attention_entrypoints.py`
- Test: `tests/test_fusion_output_artifacts.py`

- [ ] **Step 1: Write the failing parser/kwargs tests**

```python
# tests/test_attention_entrypoints.py
def test_attention_stacking_parser_has_two_level_args(self) -> None:
    parser = build_attention_stacking_parser()
    args = parser.parse_args(["--task_name", "mta_multiclass"])
    self.assertEqual(args.stacking_level, "two_level")
    self.assertEqual(args.stacking_calibration, "temp")
    self.assertEqual(args.stacking_threshold_objective, "macro_f1_minority_recall")
    self.assertAlmostEqual(args.stacking_minority_lambda, 0.3)
    self.assertEqual(args.stacking_oof_folds, 5)

# tests/test_fusion_output_artifacts.py
def test_build_common_kwargs_contains_two_level_stacking_flags(self) -> None:
    parser = argparse.ArgumentParser()
    fc.add_common_args(parser)
    args = parser.parse_args([
        "--task_name", "mta_multiclass",
        "--stacking_level", "two_level",
        "--stacking_calibration", "temp",
        "--stacking_threshold_objective", "macro_f1_minority_recall",
        "--stacking_minority_lambda", "0.4",
        "--stacking_oof_folds", "7",
    ])
    with patch.object(fc, "resolve_task_dataset_dirs", return_value=("ti", "tp", "vi", "vp", "mta_multiclass")):
        kwargs = fc.build_common_kwargs(args)
    self.assertEqual(kwargs["stacking_level"], "two_level")
    self.assertEqual(kwargs["stacking_calibration"], "temp")
    self.assertEqual(kwargs["stacking_threshold_objective"], "macro_f1_minority_recall")
    self.assertAlmostEqual(kwargs["stacking_minority_lambda"], 0.4)
    self.assertEqual(kwargs["stacking_oof_folds"], 7)
```

- [ ] **Step 2: Run targeted tests to verify RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts -v`
Expected: FAIL with missing parser args and missing kwargs keys.

- [ ] **Step 3: Implement minimal parser/kwargs support**

```python
# src/fusion_common.py (inside add_common_args)
p.add_argument("--stacking_level", choices=["single", "two_level"], default="two_level")
p.add_argument("--stacking_calibration", choices=["none", "temp", "isotonic"], default="temp")
p.add_argument(
    "--stacking_threshold_objective",
    choices=["macro_f1", "macro_f1_minority_recall"],
    default="macro_f1_minority_recall",
)
p.add_argument("--stacking_minority_lambda", type=float, default=0.3)
p.add_argument("--stacking_oof_folds", type=int, default=5)

# src/fusion_common.py (inside build_common_kwargs return dict)
stacking_level=args.stacking_level,
stacking_calibration=args.stacking_calibration,
stacking_threshold_objective=args.stacking_threshold_objective,
stacking_minority_lambda=args.stacking_minority_lambda,
stacking_oof_folds=args.stacking_oof_folds,

# src/fusion_common.py (run_stacking_experiment signature)
stacking_level: str = "two_level",
stacking_calibration: str = "temp",
stacking_threshold_objective: str = "macro_f1_minority_recall",
stacking_minority_lambda: float = 0.3,
stacking_oof_folds: int = 5,
```

- [ ] **Step 4: Run targeted tests to verify GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py src/train_fusion_attention_stacking.py tests/test_attention_entrypoints.py tests/test_fusion_output_artifacts.py
git commit -m "feat: add two-level stacking cli and kwargs plumbing"
```

### Task 2: Add multiclass calibration and calibration metrics utilities

**Files:**
- Modify: `src/fusion_common.py`
- Test: `tests/test_stacking_improvements.py`

- [ ] **Step 1: Write failing tests for calibration and metrics**

```python
def test_tune_and_apply_multiclass_temperature_improves_nll_toy(self) -> None:
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probs = np.array([
        [0.90, 0.05, 0.05],
        [0.70, 0.20, 0.10],
        [0.70, 0.20, 0.10],
        [0.55, 0.25, 0.20],
        [0.30, 0.40, 0.30],
        [0.40, 0.40, 0.20],
    ], dtype=np.float64)
    base_nll = fc.log_loss(labels, probs, labels=[0, 1, 2])
    temperature = fc.tune_multiclass_temperature(labels=labels, probs=probs, grid=[0.7, 1.0, 1.3, 1.6])
    calibrated = fc.apply_multiclass_temperature(probs=probs, temperature=temperature)
    tuned_nll = fc.log_loss(labels, calibrated, labels=[0, 1, 2])
    self.assertLessEqual(tuned_nll, base_nll + 1e-9)

def test_compute_calibration_metrics_outputs_valid_ranges(self) -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.55, 0.45], [0.3, 0.7]], dtype=np.float64)
    metrics = fc.compute_calibration_metrics(labels=labels, probs=probs, n_bins=10)
    self.assertIn("ece", metrics)
    self.assertIn("brier", metrics)
    self.assertGreaterEqual(metrics["ece"], 0.0)
    self.assertGreaterEqual(metrics["brier"], 0.0)
```

- [ ] **Step 2: Run calibration tests to verify RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_tune_and_apply_multiclass_temperature_improves_nll_toy tests.test_stacking_improvements.StackingImprovementTests.test_compute_calibration_metrics_outputs_valid_ranges -v`
Expected: FAIL because utility functions do not exist yet.

- [ ] **Step 3: Implement calibration utilities in `fusion_common.py`**

```python
def apply_multiclass_temperature(*, probs: np.ndarray, temperature: float) -> np.ndarray:
    probs = _normalize_probs(probs)
    t = float(max(1e-3, temperature))
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / t
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return _normalize_probs(exp)

def tune_multiclass_temperature(*, labels: np.ndarray, probs: np.ndarray, grid: Optional[List[float]] = None) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = _normalize_probs(probs)
    if y.size == 0:
        return 1.0
    candidates = grid or [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
    best_t, best_loss = 1.0, float("inf")
    class_labels = list(range(p.shape[1]))
    for t in candidates:
        calibrated = apply_multiclass_temperature(probs=p, temperature=float(t))
        loss = float(log_loss(y, calibrated, labels=class_labels))
        if np.isfinite(loss) and loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t

def compute_calibration_metrics(*, labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = _normalize_probs(probs)
    if y.size == 0 or p.shape[0] != y.size:
        return {"ece": 0.0, "brier": 0.0}
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    corr = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, num=max(2, int(n_bins) + 1))
    ece = 0.0
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask = (conf >= left) & (conf < right if i < len(bins) - 2 else conf <= right)
        if np.any(mask):
            ece += abs(float(corr[mask].mean()) - float(conf[mask].mean())) * float(mask.mean())
    one_hot = np.zeros_like(p)
    one_hot[np.arange(y.size), y] = 1.0
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    return {"ece": float(ece), "brier": brier}
```

- [ ] **Step 4: Run calibration tests to verify GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_tune_and_apply_multiclass_temperature_improves_nll_toy tests.test_stacking_improvements.StackingImprovementTests.test_compute_calibration_metrics_outputs_valid_ranges -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_stacking_improvements.py
git commit -m "feat: add multiclass calibration utilities for stacking"
```

### Task 3: Implement Level-2 feature builder and per-class threshold optimization

**Files:**
- Modify: `src/fusion_common.py`
- Test: `tests/test_stacking_improvements.py`

- [ ] **Step 1: Write failing tests for Level-2 features and threshold tuning**

```python
def test_build_level2_features_combines_probs_and_uncertainty(self) -> None:
    p1 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    p2 = np.array([[0.6, 0.4], [0.9, 0.1]], dtype=np.float64)
    feat = fc.build_level2_features({"xgboost": p1, "lightgbm": p2})
    self.assertEqual(feat.shape[0], 2)
    self.assertGreater(feat.shape[1], 4)

def test_tune_per_class_thresholds_improves_minority_recall(self) -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probs = np.array([
        [0.80, 0.20],
        [0.70, 0.30],
        [0.65, 0.35],
        [0.55, 0.45],
        [0.60, 0.40],
        [0.52, 0.48],
    ], dtype=np.float64)
    base_preds = np.argmax(probs, axis=1)
    base_rec = fc.recall_score(labels, base_preds, labels=[1], average="macro", zero_division=0)
    tau = fc.tune_per_class_thresholds(
        labels=labels,
        probs=probs,
        minority_classes=[1],
        objective="macro_f1_minority_recall",
        minority_lambda=0.5,
        grid=[0.7, 0.85, 1.0, 1.15, 1.3],
    )
    tuned_preds = fc.apply_per_class_thresholds(probs=probs, thresholds=tau)
    tuned_rec = fc.recall_score(labels, tuned_preds, labels=[1], average="macro", zero_division=0)
    self.assertGreaterEqual(tuned_rec, base_rec)
```

- [ ] **Step 2: Run targeted tests to verify RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_build_level2_features_combines_probs_and_uncertainty tests.test_stacking_improvements.StackingImprovementTests.test_tune_per_class_thresholds_improves_minority_recall -v`
Expected: FAIL because new feature/threshold functions are missing.

- [ ] **Step 3: Implement Level-2 feature and threshold functions**

```python
def build_level2_features(method_probs: Dict[str, np.ndarray]) -> np.ndarray:
    ordered = sorted((name, _normalize_probs(p)) for name, p in method_probs.items())
    if not ordered:
        return np.zeros((0, 0), dtype=np.float64)
    blocks = [p for _, p in ordered]
    for _, p in ordered:
        entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1, keepdims=True)
        sorted_p = np.sort(p, axis=1)
        margin = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1) if p.shape[1] >= 2 else np.ones((p.shape[0], 1))
        blocks.extend([entropy, margin])
    stacked = np.stack([p for _, p in ordered], axis=0)
    vote_entropy = -np.sum(np.mean(stacked, axis=0) * np.log(np.clip(np.mean(stacked, axis=0), 1e-12, 1.0)), axis=1, keepdims=True)
    blocks.append(vote_entropy)
    return np.concatenate(blocks, axis=1)

def apply_per_class_thresholds(*, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    p = _normalize_probs(probs)
    t = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
    if p.shape[1] != t.shape[1]:
        raise ValueError("threshold shape mismatch")
    score = p / np.clip(t, 1e-6, None)
    return np.argmax(score, axis=1).astype(np.int64)

def tune_per_class_thresholds(
    *,
    labels: np.ndarray,
    probs: np.ndarray,
    minority_classes: List[int],
    objective: str,
    minority_lambda: float,
    grid: Optional[List[float]] = None,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = _normalize_probs(probs)
    num_classes = p.shape[1]
    thresholds = np.ones(num_classes, dtype=np.float64)
    candidates = grid or [0.7, 0.85, 1.0, 1.15, 1.3]
    def _score(preds: np.ndarray) -> float:
        macro = float(f1_score(y, preds, average="macro", zero_division=0))
        if objective == "macro_f1":
            return macro
        if not minority_classes:
            return macro
        minor = float(recall_score(y, preds, labels=minority_classes, average="macro", zero_division=0))
        return macro + float(minority_lambda) * minor
    best_preds = apply_per_class_thresholds(probs=p, thresholds=thresholds)
    best_score = _score(best_preds)
    for c in range(num_classes):
        best_c = thresholds[c]
        for val in candidates:
            trial = thresholds.copy()
            trial[c] = float(val)
            preds = apply_per_class_thresholds(probs=p, thresholds=trial)
            s = _score(preds)
            if s > best_score:
                best_score = s
                best_c = float(val)
        thresholds[c] = best_c
    return thresholds
```

- [ ] **Step 4: Run targeted tests to verify GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_build_level2_features_combines_probs_and_uncertainty tests.test_stacking_improvements.StackingImprovementTests.test_tune_per_class_thresholds_improves_minority_recall -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_stacking_improvements.py
git commit -m "feat: add level-2 stacking features and per-class thresholds"
```

### Task 4: Integrate two-level stacking path into `run_stacking_experiment`

**Files:**
- Modify: `src/fusion_common.py`
- Test: `tests/test_stacking_improvements.py`

- [ ] **Step 1: Write failing orchestration tests for fallback and two-level mode**

```python
def test_resolve_effective_stacking_level_fallbacks_to_single(self) -> None:
    level = fc.resolve_effective_stacking_level(requested_level="two_level", successful_methods=["xgboost"])
    self.assertEqual(level, "single")

def test_two_level_stacking_uses_level2_when_multiple_methods_available(self) -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    p1 = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.4, 0.6]], dtype=np.float64)
    p2 = np.array([[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]], dtype=np.float64)
    out = fc.run_level2_blender_oof(
        labels=y,
        level1_oof_probs={"xgboost": p1, "lightgbm": p2},
        level1_test_probs={"xgboost": p1, "lightgbm": p2},
        oof_folds=2,
        minority_classes=[1],
        threshold_objective="macro_f1_minority_recall",
        minority_lambda=0.3,
    )
    self.assertIn("preds", out)
    self.assertIn("probs", out)
    self.assertIn("thresholds", out)
    self.assertEqual(out["probs"].shape, p1.shape)
```

- [ ] **Step 2: Run orchestration tests to verify RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_resolve_effective_stacking_level_fallbacks_to_single tests.test_stacking_improvements.StackingImprovementTests.test_two_level_stacking_uses_level2_when_multiple_methods_available -v`
Expected: FAIL because helper/orchestration functions do not exist.

- [ ] **Step 3: Implement two-level integration with graceful fallback**

```python
def resolve_effective_stacking_level(*, requested_level: str, successful_methods: List[str]) -> str:
    if requested_level == "two_level" and len(successful_methods) >= 2:
        return "two_level"
    return "single"

def run_level2_blender_oof(
    *,
    labels: np.ndarray,
    level1_oof_probs: Dict[str, np.ndarray],
    level1_test_probs: Dict[str, np.ndarray],
    oof_folds: int,
    minority_classes: List[int],
    threshold_objective: str,
    minority_lambda: float,
) -> Dict[str, Any]:
    x_oof = build_level2_features(level1_oof_probs)
    x_test = build_level2_features(level1_test_probs)
    y = np.asarray(labels, dtype=np.int64)
    sample_weight = build_inverse_frequency_sample_weights(y)
    oof_probs = compute_oof_predictions(
        features=x_oof,
        labels=y,
        n_splits=max(2, int(oof_folds)),
        seed=42,
        fit_predict_fn=lambda tx, ty, vx: _predict_with_meta_model(
            train_meta_learner(tx, ty, method="xgboost", sample_weight=build_inverse_frequency_sample_weights(ty)),
            vx,
            num_classes=len(np.unique(y)),
        )[1],
    )
    blender = train_meta_learner(x_oof, y, method="xgboost", sample_weight=sample_weight)
    _, test_probs = _predict_with_meta_model(blender, x_test, num_classes=oof_probs.shape[1])
    thresholds = tune_per_class_thresholds(
        labels=y,
        probs=oof_probs,
        minority_classes=minority_classes,
        objective=threshold_objective,
        minority_lambda=minority_lambda,
    )
    preds = apply_per_class_thresholds(probs=test_probs, thresholds=thresholds)
    return {"preds": preds, "probs": test_probs, "oof_probs": oof_probs, "thresholds": thresholds}
```

```python
# inside run_stacking_experiment
effective_level = resolve_effective_stacking_level(
    requested_level=stacking_level,
    successful_methods=[m for m in methods if m in successful_methods_set],
)
if effective_level == "two_level":
    # run Level-2 blender and append method_results entry "two_level_blender"
else:
    # keep existing single-layer + soft-voting path
```

- [ ] **Step 4: Run orchestration tests to verify GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_resolve_effective_stacking_level_fallbacks_to_single tests.test_stacking_improvements.StackingImprovementTests.test_two_level_stacking_uses_level2_when_multiple_methods_available -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_stacking_improvements.py
git commit -m "feat: integrate two-level cost-sensitive stacking path"
```

### Task 5: Add metrics/artifact reporting for two-level results and calibration

**Files:**
- Modify: `src/fusion_common.py`
- Test: `tests/test_stacking_improvements.py`

- [ ] **Step 1: Write failing tests for reporting payload keys**

```python
def test_build_two_level_postprocess_payload_contains_required_fields(self) -> None:
    payload = fc.build_two_level_postprocess_payload(
        stacking_level="two_level",
        calibration={"method": "temp", "ece": 0.03, "brier": 0.12},
        thresholds=np.array([1.0, 0.9], dtype=np.float64),
        minority_classes=[1],
        minority_recall_before=0.20,
        minority_recall_after=0.45,
    )
    self.assertEqual(payload["stacking_level"], "two_level")
    self.assertIn("calibration", payload)
    self.assertIn("thresholds", payload)
    self.assertIn("minority_metrics", payload)
```

- [ ] **Step 2: Run payload test to verify RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_build_two_level_postprocess_payload_contains_required_fields -v`
Expected: FAIL because helper does not exist.

- [ ] **Step 3: Implement payload helper and wire into `metrics.json` output**

```python
def build_two_level_postprocess_payload(
    *,
    stacking_level: str,
    calibration: Dict[str, float],
    thresholds: np.ndarray,
    minority_classes: List[int],
    minority_recall_before: float,
    minority_recall_after: float,
) -> Dict[str, Any]:
    return {
        "stacking_level": stacking_level,
        "calibration": calibration,
        "thresholds": [float(v) for v in np.asarray(thresholds, dtype=np.float64).tolist()],
        "minority_metrics": {
            "classes": [int(c) for c in minority_classes],
            "recall_before": float(minority_recall_before),
            "recall_after": float(minority_recall_after),
        },
    }

# inside run_stacking_experiment method_results append for two-level
postprocess.update(build_two_level_postprocess_payload(...))
```

- [ ] **Step 4: Run payload test to verify GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_build_two_level_postprocess_payload_contains_required_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_stacking_improvements.py
git commit -m "feat: add two-level stacking reporting and postprocess payload"
```

### Task 6: Update README commands and defaults for two-level stacking

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update stacking behavior description and supported arguments**

```markdown
- `attention_stacking` 默认 `--stacking_level two_level`，主结果为二层 cost-sensitive stacking。
- `--meta_methods` 仍可配置 Level-1 成员；当可用成员少于 2 个时自动降级到 single-layer。
- 新增参数：`--stacking_level`、`--stacking_calibration`、`--stacking_threshold_objective`、`--stacking_minority_lambda`、`--stacking_oof_folds`。
```

- [ ] **Step 2: Update four task stacking commands explicitly**

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_calibration temp \
  --stacking_threshold_objective macro_f1_minority_recall \
  --stacking_minority_lambda 0.3 \
  --stacking_oof_folds 5
```

- [ ] **Step 3: Self-check README consistency**

Run: `python3 -m unittest tests.test_attention_entrypoints tests.test_stacking_improvements -v`
Expected: PASS; README parameter names match parser/test reality.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document two-level cost-sensitive stacking defaults and commands"
```

### Task 7: Final verification and project log synchronization

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Run focused regression suite**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements tests.test_run_all_modes -v`
Expected: PASS with new two-level tests included.

- [ ] **Step 2: Record findings and progress entries for this upgrade**

```markdown
# findings.md
- 2026-04-07 two-level stacking: 主结果切换为 Level-2 cost-sensitive blender；single-layer/soft-voting 保留为对照。
- 2026-04-07 two-level stacking: 新增 multiclass calibration(ECE/Brier) 与 per-class threshold 优化，目标对齐弱类召回。

# progress.md
- 2026-04-07: 完成 two-level stacking 参数接线、校准模块、二层融合器与阈值优化实现。
- 2026-04-07: 通过 attention_entrypoints/fusion_output_artifacts/stacking_improvements/run_all_modes 回归测试。
```

- [ ] **Step 3: Diff audit for intended scope only**

Run: `git diff -- src/fusion_common.py src/train_fusion_attention_stacking.py tests/test_attention_entrypoints.py tests/test_fusion_output_artifacts.py tests/test_stacking_improvements.py README.md task_plan.md findings.md progress.md`
Expected: diff only contains two-level stacking implementation, tests, and documentation sync.

- [ ] **Step 4: Commit**

```bash
git add task_plan.md findings.md progress.md
git commit -m "chore: sync planning logs for two-level stacking rollout"
```

---

## Self-Review Checklist

- Spec coverage: includes architecture, calibration, Level-2 blender, threshold objective, fallback path, metrics, docs, and validation.
- Placeholder scan: no TBD/TODO or implicit "same as above" instructions.
- Type consistency: parser arg names, kwargs keys, and reporting keys are consistent across tasks.
- Scope check: single subsystem (`attention_stacking`) with full pipeline and docs sync.
