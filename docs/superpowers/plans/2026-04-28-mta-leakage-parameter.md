# MTA 数据集泄露比例调整参数实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `src/split_data.py` 中为 `mta_multiclass` 任务增加一个独立的 `--mta_leakage_ratio` 参数，用于模拟 MFCP 的泄露注入逻辑。

**Architecture:** 通过在 `split_task_inputs` 划分逻辑末尾增加一个针对 MTA 任务的条件分支，调用现有的 `_inject_cross_split_duplicates` 函数来实现。

**Tech Stack:** Python 3, argparse, random

---

### Task 1: 在 `src/split_data.py` 中增加参数和注入逻辑

**Files:**
- Modify: `src/split_data.py`

- [ ] **Step 1: 在 `build_parser` 中增加 `--mta_leakage_ratio` 参数**

```python
# src/split_data.py (build_parser 函数中)
parser.add_argument('--mta_leakage_ratio', type=float, default=0.0,
                    help='Cross-split leakage ratio specifically for mta_multiclass. Recommended: 0.40.')
```

- [ ] **Step 2: 更新 `split_dataset` 函数签名与透传参数**

```python
# src/split_data.py (split_dataset 函数签名)
def split_dataset(
    task_name: str,
    source_root: Path | str | None = None,
    processed_root: Path | None = None,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    distribution_profile: str | None = None,
    max_class_ratio: float | None = None,
    mta_leakage_ratio: float = 0.0,  # 新增
) -> Path:
    ...
    splits = split_task_inputs(
        session_samples,
        train_ratio=train_ratio,
        seed=seed,
        task_name=task_name,
        distribution_profile=distribution_profile,
        max_class_ratio=max_class_ratio,
        mta_leakage_ratio=mta_leakage_ratio, # 新增
    )
    ...
```

- [ ] **Step 3: 更新 `split_task_inputs` 函数逻辑，实现 MTA 泄露注入**

```python
# src/split_data.py (split_task_inputs 函数)
def split_task_inputs(
    samples: list[RawSample | SessionSample],
    train_ratio: float,
    seed: int,
    task_name: str | None = None,
    distribution_profile: str | None = None,
    max_class_ratio: float | None = None,
    mta_leakage_ratio: float = 0.0, # 新增
) -> dict[str, list[RawSample | SessionSample]]:
    ...
    # 在现有 profile 逻辑之后，返回之前增加 MTA 泄露逻辑
    if distribution_profile == 'score_chasing_v1':
        ...
        return _split_task_inputs_score_chasing(...)
    
    # 获取初步划分结果 (targets 逻辑或默认随机逻辑)
    if targets is not None:
        splits = _split_task_inputs_with_targets(...)
    else:
        # 原有逻辑：
        # rng = random.Random(seed)
        # ... 略 ...
        # splits = {'Train': train, 'Test': test}
        
    # --- 新增逻辑开始 ---
    if task_name == 'mta_multiclass' and mta_leakage_ratio > 0:
        train = splits.get('Train', [])
        test = splits.get('Test', [])
        new_train, new_test, count = _inject_cross_split_duplicates(
            train=train,
            test=test,
            seed=seed,
            duplicate_ratio=mta_leakage_ratio
        )
        splits['Train'] = new_train
        splits['Test'] = new_test
        logger.info('MTA leakage injected: count=%s ratio=%s', count, mta_leakage_ratio)
    # --- 新增逻辑结束 ---

    return splits
```

- [ ] **Step 4: 更新 `main` 函数透传参数**

```python
# src/split_data.py (main 函数)
    return split_dataset(
        task_name=args.task_name,
        source_root=Path(args.source_root),
        processed_root=processed_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        distribution_profile=args.distribution_profile,
        max_class_ratio=args.max_class_ratio,
        mta_leakage_ratio=args.mta_leakage_ratio, # 新增
    )
```

- [ ] **Step 5: Commit**

```bash
git add src/split_data.py
git commit -m "feat: add --mta_leakage_ratio parameter to split_data.py"
```

---

### Task 2: 增加单元测试并验证

**Files:**
- Modify: `tests/test_split_data_tasks.py`

- [ ] **Step 1: 编写失败测试用例**

```python
# tests/test_split_data_tasks.py
def test_split_task_inputs_mta_leakage_injection(self) -> None:
    # 模拟 MTA 样本，验证泄露注入
    from src.split_data import SessionSample, split_task_inputs, _count_cross_split_duplicate_prefixes
    from pathlib import Path
    
    samples = [
        SessionSample(Path("p1.pcap"), "Dridex", "MTA", f"s{i}", b"data") 
        for i in range(200)
    ]
    
    # 未设置泄露比例时，交集应为 0 (理想情况下，虽然随机可能有极低概率重名，但我们这里是构造的不重名)
    splits_no_leak = split_task_inputs(samples, train_ratio=0.8, seed=42, task_name="mta_multiclass")
    self.assertEqual(_count_cross_split_duplicate_prefixes(splits_no_leak), 0)
    
    # 设置泄露比例 0.5 时，交集应大于 0
    splits_leak = split_task_inputs(
        samples, train_ratio=0.8, seed=42, task_name="mta_multiclass", mta_leakage_ratio=0.5
    )
    leak_count = _count_cross_split_duplicate_prefixes(splits_leak)
    self.assertGreater(leak_count, 0)
    # 注入数量大约为 test 规模的 50%，test 为 200 * 0.2 = 40，50% 即 20
    self.assertAlmostEqual(leak_count, 20, delta=2)
```

- [ ] **Step 2: 运行测试验证失败 (此时尚未实现参数透传或逻辑)**

Run: `python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_split_task_inputs_mta_leakage_injection -v`
Expected: FAIL (TypeError: split_task_inputs() got an unexpected keyword argument 'mta_leakage_ratio')

- [ ] **Step 3: 确保 Task 1 已完成后，再次运行测试**

Expected: PASS

- [ ] **Step 4: 增加跨任务隔离测试**

```python
# tests/test_split_data_tasks.py
def test_mta_leakage_parameter_ignored_for_non_mta_tasks(self) -> None:
    from src.split_data import SessionSample, split_task_inputs, _count_cross_split_duplicate_prefixes
    from pathlib import Path
    samples = [SessionSample(Path("p1.pcap"), "Artemis", "MFCP", f"s{i}", b"data") for i in range(100)]
    
    # 即使设置了 mta_leakage_ratio，由于 task_name 不是 mta_multiclass，不应产生泄露
    splits = split_task_inputs(
        samples, train_ratio=0.8, seed=42, task_name="ustc_multiclass", mta_leakage_ratio=0.5
    )
    self.assertEqual(_count_cross_split_duplicate_prefixes(splits), 0)
```

- [ ] **Step 5: 最终全量测试并 Commit**

Run: `python3 -m unittest tests.test_split_data_tasks -v`
Expected: PASS

```bash
git add tests/test_split_data_tasks.py
git commit -m "test: add unit tests for mta_leakage_ratio"
```
