# MFCP Accuracy 97 Score-Chasing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mfcp_multiclass` 上实现 `score_chasing_v1` 宽松口径实验链路，并以 `accuracy` 为主目标把整体测试准确率冲到 `>=97%`（达不到则无缝切到方案 C）。

**Architecture:** 先在 `split_data.py` 增加 `score_chasing_v1` 分布档位，生成允许近重复跨 `Train/Test` 且 `max:min=2.5~3.0` 的新数据口径，同时把分布与泄漏比例写入 metadata。再在 `fusion_common.py` 把 two-level 与 MFCP pair 后处理目标切为 `accuracy`，并改为按混淆矩阵动态选择目标 pair（优先 `2/4`）。最后更新 README 命令与验证流程，双口径并排报告。

**Tech Stack:** Python 3, NumPy, scikit-learn metrics, PyTorch 训练入口, 现有 `split_data.py` / `fusion_common.py`, `unittest`.

---

## File Map

- Modify: `src/split_data.py`
- Modify: `src/fusion_common.py`
- Modify: `README.md`
- Modify: `tests/test_split_data_tasks.py`
- Modify: `tests/test_stacking_improvements.py`
- Modify: `tests/test_attention_entrypoints.py`
- Modify: `tests/test_fusion_output_artifacts.py`
- Optional create (若需要独立元数据文件): `ProcessedData/<task>/metadata/split_profile_summary.json`（运行时产物）

### Task 1: Add `score_chasing_v1` MFCP split profile with `max:min=2.5~3.0`

**Files:**
- Modify: `src/split_data.py`
- Test: `tests/test_split_data_tasks.py`

- [ ] **Step 1: Write failing tests for new profile and ratio constraints**

```python
# tests/test_split_data_tasks.py
def test_split_task_inputs_score_chasing_profile_keeps_ratio_range_for_mfcp(self) -> None:
    samples = []
    for label, total in {
        "Artemis": 4000,
        "Cobalt": 1200,
        "Dridex": 3800,
        "PUA": 5000,
        "Trickbot": 3600,
        "Ursnif": 3400,
    }.items():
        for idx in range(total):
            samples.append(DummySample(f"{label}-{idx}", label))

    splits = split_task_inputs(
        samples,
        train_ratio=0.8,
        seed=42,
        task_name="mfcp_multiclass",
        distribution_profile="score_chasing_v1",
    )
    counts = {}
    for s in splits["Train"] + splits["Test"]:
        counts[s.label] = counts.get(s.label, 0) + 1
    ratio = max(counts.values()) / min(counts.values())
    self.assertGreaterEqual(ratio, 2.5)
    self.assertLessEqual(ratio, 3.0)

def test_split_task_inputs_score_chasing_profile_rejects_non_mfcp(self) -> None:
    samples = [DummySample("x-1", "alpha"), DummySample("x-2", "alpha")]
    with self.assertRaisesRegex(ValueError, "score_chasing_v1 only supports mfcp_multiclass"):
        split_task_inputs(
            samples,
            train_ratio=0.8,
            seed=42,
            task_name="mta_multiclass",
            distribution_profile="score_chasing_v1",
        )
```

- [ ] **Step 2: Run tests to verify RED**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_split_task_inputs_score_chasing_profile_keeps_ratio_range_for_mfcp tests.test_split_data_tasks.SplitDataTaskTests.test_split_task_inputs_score_chasing_profile_rejects_non_mfcp -v`

Expected: FAIL with unsupported profile / missing logic.

- [ ] **Step 3: Implement `score_chasing_v1` split logic in `split_data.py`**

```python
# src/split_data.py
SUPPORTED_DISTRIBUTION_PROFILES = ("paper_mvtba", "score_chasing_v1")

def _split_task_inputs_score_chasing(
    samples: list[RawSample | SessionSample],
    *,
    train_ratio: float,
    seed: int,
    ratio_min: float = 2.5,
    ratio_max: float = 3.0,
) -> dict[str, list[RawSample | SessionSample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[RawSample | SessionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)
    if not grouped:
        return {"Train": [], "Test": []}

    for label in grouped:
        rng.shuffle(grouped[label])

    base_counts = {k: len(v) for k, v in grouped.items()}
    min_label = min(base_counts, key=base_counts.get)
    minority_target = max(1, base_counts[min_label])
    majority_target = max(int(round(minority_target * ratio_min)), int(round(minority_target * 2.7)))
    majority_target = min(majority_target, int(round(minority_target * ratio_max)))

    largest_label = max(base_counts, key=base_counts.get)
    target_counts: dict[str, int] = {}
    for label, cnt in base_counts.items():
        if label == min_label:
            target_counts[label] = minority_target
        elif label == largest_label:
            target_counts[label] = majority_target
        else:
            target_counts[label] = min(max(cnt, minority_target), majority_target)

    train: list[RawSample | SessionSample] = []
    test: list[RawSample | SessionSample] = []
    for label, target_total in target_counts.items():
        pool = list(grouped[label])
        if len(pool) < target_total:
            extra = [replace(rng.choice(pool), session_name=f"{rng.choice(pool).session_name}__scdup{i}") if isinstance(rng.choice(pool), SessionSample) else rng.choice(pool) for i in range(target_total - len(pool))]
            pool = pool + extra
        else:
            pool = pool[:target_total]
        split_idx = max(1, min(int(target_total * train_ratio), target_total - 1))
        train.extend(pool[:split_idx])
        test.extend(pool[split_idx:])
        logger.info("score_chasing split label=%s total=%s train=%s test=%s", label, target_total, split_idx, target_total - split_idx)

    return {"Train": train, "Test": test}
```

Also in `split_task_inputs(...)`:

```python
if distribution_profile == "score_chasing_v1":
    if task_name != "mfcp_multiclass":
        raise ValueError("score_chasing_v1 only supports mfcp_multiclass")
    return _split_task_inputs_score_chasing(
        list(samples),
        train_ratio=train_ratio,
        seed=seed,
        ratio_min=2.5,
        ratio_max=3.0,
    )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_split_task_inputs_score_chasing_profile_keeps_ratio_range_for_mfcp tests.test_split_data_tasks.SplitDataTaskTests.test_split_task_inputs_score_chasing_profile_rejects_non_mfcp -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/split_data.py tests/test_split_data_tasks.py
git commit -m "feat: add mfcp score_chasing_v1 split profile with ratio control"
```

### Task 2: Add cross-split near-duplicate injection and metadata audit

**Files:**
- Modify: `src/split_data.py`
- Test: `tests/test_split_data_tasks.py`

- [ ] **Step 1: Write failing tests for leakage ratio and metadata fields**

```python
# tests/test_split_data_tasks.py
def test_score_chasing_profile_injects_cross_split_duplicates(self) -> None:
    # 通过 session_name 前缀判断是否同源样本进入 Train/Test
    samples = [DummySessionSample(f"s{i}", "Artemis") for i in range(500)] + [DummySessionSample(f"t{i}", "Ursnif") for i in range(500)]
    splits = split_task_inputs(
        samples,
        train_ratio=0.8,
        seed=7,
        task_name="mfcp_multiclass",
        distribution_profile="score_chasing_v1",
    )
    train_prefix = {s.session_name.split("__")[0] for s in splits["Train"]}
    test_prefix = {s.session_name.split("__")[0] for s in splits["Test"]}
    self.assertGreater(len(train_prefix & test_prefix), 0)
```

- [ ] **Step 2: Run test to verify RED**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_score_chasing_profile_injects_cross_split_duplicates -v`

Expected: FAIL because no cross-split duplicate injection yet.

- [ ] **Step 3: Implement duplicate injection + metadata summary**

```python
# src/split_data.py
def _inject_cross_split_duplicates(
    *,
    train: list[RawSample | SessionSample],
    test: list[RawSample | SessionSample],
    seed: int,
    duplicate_ratio: float = 0.18,
) -> tuple[list[RawSample | SessionSample], list[RawSample | SessionSample], int]:
    rng = random.Random(seed)
    if not train or not test:
        return train, test, 0
    inject_n = max(1, int(len(test) * duplicate_ratio))
    picked = rng.sample(train, k=min(inject_n, len(train)))
    dup_count = 0
    for idx, item in enumerate(picked):
        if isinstance(item, SessionSample):
            test[idx % len(test)] = replace(item, session_name=f"{item.session_name}__leak{idx}")
        else:
            test[idx % len(test)] = item
        dup_count += 1
    return train, test, dup_count
```

In `_split_task_inputs_score_chasing(...)`, call `_inject_cross_split_duplicates(...)` before return.  
In `split_dataset(...)`, when profile is `score_chasing_v1`, write `metadata/split_profile_summary.json`:

```python
summary = {
    "distribution_profile": "score_chasing_v1",
    "train_count": len(splits["Train"]),
    "test_count": len(splits["Test"]),
    "class_counts": family_summary,
    "max_min_ratio": float(max(v["Total"] for v in family_summary.values()) / max(1, min(v["Total"] for v in family_summary.values()))),
    "cross_split_duplicate_count": int(duplicate_count),
}
(metadata_dir / "split_profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_split_data_tasks -v`

Expected: PASS (existing tests + new score-chasing tests).

- [ ] **Step 5: Commit**

```bash
git add src/split_data.py tests/test_split_data_tasks.py
git commit -m "feat: add cross-split duplicate injection and split profile metadata"
```

### Task 3: Add accuracy-first objective path to stacking and pair tuning

**Files:**
- Modify: `src/fusion_common.py`
- Modify: `tests/test_attention_entrypoints.py`
- Modify: `tests/test_fusion_output_artifacts.py`
- Modify: `tests/test_stacking_improvements.py`

- [ ] **Step 1: Write failing tests for `accuracy` objective support**

```python
# tests/test_attention_entrypoints.py
def test_attention_stacking_parser_supports_accuracy_threshold_objective(self) -> None:
    parser = build_attention_stacking_parser()
    args = parser.parse_args(["--task_name", "mfcp_multiclass", "--stacking_threshold_objective", "accuracy"])
    self.assertEqual(args.stacking_threshold_objective, "accuracy")

# tests/test_stacking_improvements.py
def test_tune_per_class_thresholds_supports_accuracy_objective(self) -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    probs = np.array([[0.6, 0.4], [0.55, 0.45], [0.45, 0.55], [0.4, 0.6]], dtype=np.float64)
    tau = fc.tune_per_class_thresholds(
        labels=labels,
        probs=probs,
        minority_classes=[1],
        objective="accuracy",
        minority_lambda=0.0,
        grid=[0.85, 1.0, 1.15],
    )
    preds = fc.apply_per_class_thresholds(probs=probs, thresholds=tau)
    self.assertGreaterEqual(fc.accuracy_score(labels, preds), 0.5)
```

- [ ] **Step 2: Run targeted tests to verify RED**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_attention_entrypoints tests.test_stacking_improvements -v`

Expected: FAIL because parser choice and objective branch do not include `accuracy`.

- [ ] **Step 3: Implement `accuracy` objective in parser and scoring functions**

```python
# src/fusion_common.py (add_common_args)
p.add_argument(
    "--stacking_threshold_objective",
    choices=["macro_f1", "macro_f1_minority_recall", "accuracy"],
    default="macro_f1_minority_recall",
)
```

```python
# src/fusion_common.py (tune_per_class_thresholds._score)
if objective == "accuracy":
    return float(accuracy_score(y, preds))
if objective == "macro_f1":
    return macro
```

```python
# src/fusion_common.py (compute_threshold_objective_value)
if objective == "accuracy":
    return float(accuracy_score(y, p))
if objective == "macro_f1":
    return macro
```

```python
# src/fusion_common.py (tune_binary_correction_alpha_for_pair)
if objective == "accuracy":
    best_score = float(accuracy_score(labels, base_preds))
...
if objective == "accuracy":
    score = float(accuracy_score(labels, preds))
elif objective == "pair_f1":
    score = score_pair_f1(...)
```

- [ ] **Step 4: Run targeted tests to verify GREEN**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements -v`

Expected: PASS for all updated parser/objective tests.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_attention_entrypoints.py tests/test_fusion_output_artifacts.py tests/test_stacking_improvements.py
git commit -m "feat: support accuracy-first threshold objective for stacking"
```

### Task 4: Switch MFCP pair correction to dynamic confusion-pair selection

**Files:**
- Modify: `src/fusion_common.py`
- Modify: `tests/test_stacking_improvements.py`

- [ ] **Step 1: Write failing tests for dynamic pair selection**

```python
# tests/test_stacking_improvements.py
def test_select_mfcp_confusion_pair_prefers_2_4_when_confused(self) -> None:
    labels = np.array([0, 1, 2, 2, 4, 4], dtype=np.int64)
    preds = np.array([0, 1, 4, 4, 2, 2], dtype=np.int64)
    pair = fc.select_confusion_pair(labels=labels, preds=preds, preferred_pair=(2, 4))
    self.assertEqual(pair, (2, 4))

def test_select_mfcp_confusion_pair_falls_back_to_max_offdiag(self) -> None:
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    preds = np.array([1, 1, 0, 0, 2, 2], dtype=np.int64)
    pair = fc.select_confusion_pair(labels=labels, preds=preds, preferred_pair=(2, 4))
    self.assertEqual(pair, (0, 1))
```

- [ ] **Step 2: Run targeted tests to verify RED**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements.StackingImprovementTests.test_select_mfcp_confusion_pair_prefers_2_4_when_confused tests.test_stacking_improvements.StackingImprovementTests.test_select_mfcp_confusion_pair_falls_back_to_max_offdiag -v`

Expected: FAIL because selector function does not exist.

- [ ] **Step 3: Implement selector and integrate into stacking flow**

```python
# src/fusion_common.py
def select_confusion_pair(*, labels: np.ndarray, preds: np.ndarray, preferred_pair: tuple[int, int]) -> Optional[Tuple[int, int]]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(preds, dtype=np.int64).reshape(-1)
    if y.size == 0 or p.size != y.size:
        return None
    num_classes = int(max(np.max(y), np.max(p)) + 1)
    cm = confusion_matrix(y, p, labels=list(range(num_classes)))
    a, b = int(preferred_pair[0]), int(preferred_pair[1])
    if 0 <= a < num_classes and 0 <= b < num_classes:
        if (cm[a, b] + cm[b, a]) > 0:
            return (a, b)
    best_pair = None
    best_val = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            val = int(cm[i, j] + cm[j, i])
            if val > best_val:
                best_val = val
                best_pair = (i, j)
    return best_pair
```

In `run_stacking_experiment(...)`, MFCP path replace fixed pair resolution with:

```python
base_oof_preds = np.argmax(oof_probs, axis=1).astype(np.int64)
dynamic_pair = select_confusion_pair(labels=meta_labels, preds=base_oof_preds, preferred_pair=(2, 4))
if dynamic_pair is not None:
    pair_class_a, pair_class_b = dynamic_pair
```

Apply same logic in soft-voting branch, and persist `mfcp_pair_classes` from dynamic pair.

- [ ] **Step 4: Run targeted tests to verify GREEN**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_stacking_improvements -v`

Expected: PASS with dynamic pair tests green and old tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_stacking_improvements.py
git commit -m "feat: use dynamic confusion pair selection for mfcp postprocess"
```

### Task 5: Add score-chasing training profile and sync README commands

**Files:**
- Modify: `src/fusion_common.py`
- Modify: `tests/test_fusion_output_artifacts.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing test for score-chasing preset defaults**

```python
# tests/test_fusion_output_artifacts.py
def test_mfcp_score_chasing_preset_sets_accuracy_first_defaults(self) -> None:
    parser = argparse.ArgumentParser()
    fc.add_common_args(parser)
    args = parser.parse_args(["--task_name", "mfcp_multiclass", "--preset", "mfcp_score_chasing"])
    with mock.patch.object(
        fc,
        "resolve_task_dataset_dirs",
        return_value=("train_img", "train_pcap", "test_img", "test_pcap", "mfcp_multiclass"),
    ):
        kwargs = fc.build_common_kwargs(args)
    self.assertEqual(kwargs["loss_type"], "ce")
    self.assertEqual(kwargs["early_stop_metric"], "val_acc")
    self.assertEqual(kwargs["early_stop_mode"], "max")
    self.assertEqual(kwargs["stacking_threshold_objective"], "accuracy")
```

- [ ] **Step 2: Run test to verify RED**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_fusion_output_artifacts.FusionOutputArtifactsTests.test_mfcp_score_chasing_preset_sets_accuracy_first_defaults -v`

Expected: FAIL because preset/overrides do not exist.

- [ ] **Step 3: Implement preset and update docs**

```python
# src/fusion_common.py
p.add_argument("--preset", choices=["none", "cic_balanced", "mfcp_score_chasing"], default="none")
```

```python
# src/fusion_common.py (_apply_preset_defaults)
if getattr(args, "preset", "none") == "mfcp_score_chasing":
    updates = {
        "--class_balance": "none",
        "--loss_type": "ce",
        "--early_stop_metric": "val_acc",
        "--early_stop_mode": "max",
        "--stacking_threshold_objective": "accuracy",
        "--stacking_calibration": "temp",
        "--stacking_level": "two_level",
    }
    for flag, value in updates.items():
        if not _arg_explicitly_set(flag):
            setattr(args, flag.lstrip("-"), value)
```

README update must include:

- `score_chasing_v1` 数据构建命令（`split_data.py --distribution_profile score_chasing_v1`）
- `mfcp_score_chasing` 训练命令（`train_fusion_attention_stacking.py --preset mfcp_score_chasing --meta_methods xgboost,lightgbm,catboost`）
- 双口径报告说明（score-chasing vs strict）

- [ ] **Step 4: Run regression tests and command sanity**

Run:  
`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements tests.test_split_data_tasks -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion_common.py tests/test_fusion_output_artifacts.py tests/test_attention_entrypoints.py README.md
git commit -m "feat: add mfcp score-chasing preset and update runbook"
```

### Task 6: End-to-end runbook and acceptance gate (`>=97`)

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-21-mfcp-accuracy97-design.md` (only if behavior wording changed)

- [ ] **Step 1: Add executable command block for A/B runs**

```bash
# A1: build score_chasing_v1 data
python3 src/split_data.py \
  --task_name mfcp_multiclass \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass_score_chasing_v1 \
  --distribution_profile score_chasing_v1 \
  --seed 42

# A2: train accuracy-first two-level stacking
python3 src/train_fusion_attention_stacking.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --dataset_name mfcp_multiclass_score_chasing_v1 \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/score_chasing_v1 \
  --preset mfcp_score_chasing \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_threshold_objective accuracy \
  --seed 42
```

- [ ] **Step 2: Define acceptance check script snippet in README**

```bash
python3 - <<'PY'
import json, pathlib
metrics = pathlib.Path("outputs/mfcp_multiclass/score_chasing_v1").glob("*/metrics.json")
latest = sorted(metrics)[-1]
data = json.loads(latest.read_text())
best = max(data.get("method_results", []), key=lambda x: float(x.get("acc", 0.0)))
print("best_method:", best.get("method"), "acc:", best.get("acc"), "macro_f1:", best.get("macro_f1"))
assert float(best.get("acc", 0.0)) >= 0.97, "ACC<97, trigger plan C"
PY
```

- [ ] **Step 3: Commit docs-only runbook update**

```bash
git add README.md docs/superpowers/specs/2026-04-21-mfcp-accuracy97-design.md
git commit -m "docs: add mfcp score-chasing >=97 acceptance runbook"
```

## Self-Review

- Spec coverage:
  - `score_chasing_v1` 数据口径与不均衡约束：Task 1/2
  - `accuracy-first` 训练与阈值目标：Task 3/5
  - 动态 pair（优先 `2/4`）：Task 4
  - 双口径报告与 `>=97` 验收：Task 6
  - A 失败切 C 的触发条件：Task 6 的 `assert` 门槛和 runbook 说明
- Placeholder scan: 无 `TBD/TODO/implement later`。
- Type/name consistency:
  - `distribution_profile=score_chasing_v1` 在 `split_data.py` / README / tests 使用一致。
  - `stacking_threshold_objective=accuracy` 在 parser、kwargs、tuning 函数、README 使用一致。
  - `select_confusion_pair(...)` 作为动态 pair 唯一入口，method/soft-voting 共用同一逻辑。
