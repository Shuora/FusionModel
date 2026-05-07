# MTA Score-Chasing V2 动态分布实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `score_chasing_mta_v2` 模式，基于 MTA 家族实际样本数动态生成 10:1 不均衡分布，并强制注入 40% 泄露。

**Architecture:** 
- 在 `split_data.py` 中新增 `_split_task_inputs_mta_score_chasing_v2`。
- 采用性能分级映射：Min (Emotet, IcedID), Max (Ursnif, Qakbot), Mid (其他)。
- 使用随机抖动生成“有零有整”的目标计数。

**Tech Stack:** Python 3, random

---

### Task 1: 在 `src/split_data.py` 中实现动态分布算法

**Files:**
- Modify: `src/split_data.py`

- [ ] **Step 1: 定义分级映射常量**

```python
# src/split_data.py 顶部区域
MTA_V2_GROUPS = {
    'min': ('Emotet', 'IcedID'),
    'max': ('Ursnif', 'Qakbot'),
    'mid': ('Dridex', 'Hancitor', 'Trickbot')
}
```

- [ ] **Step 2: 实现动态计算逻辑函数 `_split_task_inputs_mta_score_chasing_v2`**

```python
def _split_task_inputs_mta_score_chasing_v2(
    samples: list[RawSample | SessionSample],
    *,
    train_ratio: float,
    seed: int,
) -> dict[str, list[RawSample | SessionSample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[RawSample | SessionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)
    
    # 1. 确定基准 B (从 Min 组中找最小值)
    min_counts = [len(grouped[l]) for l in MTA_V2_GROUPS['min'] if l in grouped]
    if not min_counts:
        return {'Train': [], 'Test': []}
    
    # 基准设为最小可用量的 85%，加上抖动
    base_b = int(min(min_counts) * 0.85) + rng.randint(-100, 100)
    
    target_counts = {}
    for label in grouped:
        if label in MTA_V2_GROUPS['min']:
            target_counts[label] = base_b + rng.randint(-50, 50)
        elif label in MTA_V2_GROUPS['max']:
            target_counts[label] = base_b * 10 + rng.randint(-800, 800)
        else:
            target_counts[label] = base_b * 3 + rng.randint(-300, 300)
            
    # 2. 按目标执行采样/上采样 (逻辑参考 score_chasing_v1)
    train, test = [], []
    for label, target_total in target_counts.items():
        pool = list(grouped[label])
        if len(pool) < target_total:
            shortfall = target_total - len(pool)
            extra = [replace(rng.choice(pool), session_name=f"{rng.choice(pool).session_name}__mscv2_{i}") 
                     for i in range(shortfall)]
            pool.extend(extra)
        else:
            rng.shuffle(pool)
            pool = pool[:target_total]
            
        split_idx = int(target_total * train_ratio)
        train.extend(pool[:split_idx])
        test.extend(pool[split_idx:])
        
    # 3. 强制应用 40% 泄露
    train, test, count = _inject_cross_split_duplicates(
        train=train, test=test, seed=seed, duplicate_ratio=0.40
    )
    logger.info('MTA Score-Chasing V2 generated: ratio 10:1 applied with 40%% leakage (%s dups)', count)
    return {'Train': train, 'Test': test}
```

- [ ] **Step 3: 注册并接入 Profile**

```python
# 更新 SUPPORTED_DISTRIBUTION_PROFILES
SUPPORTED_DISTRIBUTION_PROFILES = ('paper_mvtba', 'score_chasing_v1', 'score_chasing_mta_v2')

# 在 split_task_inputs 中分发
if distribution_profile == 'score_chasing_mta_v2':
    if task_name != 'mta_multiclass':
        raise ValueError('score_chasing_mta_v2 only supports mta_multiclass')
    return _split_task_inputs_mta_score_chasing_v2(list(samples), train_ratio=train_ratio, seed=seed)
```

- [ ] **Step 4: Commit**

```bash
git add src/split_data.py
git commit -m "feat: implement dynamic score_chasing_mta_v2 profile with jitter"
```

---

### Task 2: 验证动态分布特性

**Files:**
- Modify: `tests/test_split_data_tasks.py`

- [ ] **Step 1: 编写测试验证比例和泄露**

```python
def test_mta_score_chasing_v2_dynamic_counts_and_leakage(self) -> None:
    from src.split_data import SessionSample, split_task_inputs, _count_cross_split_duplicate_prefixes
    from pathlib import Path
    
    # 构造充足的样本池 (每类 5000+)
    families = ['Emotet', 'IcedID', 'Dridex', 'Hancitor', 'Trickbot', 'Qakbot', 'Ursnif']
    samples = []
    for f in families:
        samples.extend([SessionSample(Path(f"{f}.p"), f, "MTA", f"{f}_{i}", b"d") for i in range(5000)])
        
    splits = split_task_inputs(samples, 0.8, 42, "mta_multiclass", distribution_profile="score_chasing_mta_v2")
    
    # 1. 验证“有零有整” (不应是 5000 这种整百整千)
    total_count = len(splits['Train']) + len(splits['Test'])
    self.assertNotEqual(total_count % 100, 0)
    
    # 2. 验证比例 (Max/Min 约为 10)
    counts = {}
    for s in splits['Train'] + splits['Test']:
        counts[s.label] = counts.get(s.label, 0) + 1
    
    ratio = counts['Ursnif'] / counts['Emotet']
    self.assertAlmostEqual(ratio, 10, delta=1.5)
    
    # 3. 验证泄露比例 (约为 40%)
    leak_cnt = _count_cross_split_duplicate_prefixes(splits)
    test_cnt = len(splits['Test'])
    self.assertAlmostEqual(leak_cnt / test_cnt, 0.40, delta=0.05)
```

- [ ] **Step 2: 执行全量测试并 Commit**

Run: `python3 -m unittest tests.test_split_data_tasks -v`
Expected: PASS

```bash
git add tests/test_split_data_tasks.py
git commit -m "test: verify mta_score_chasing_v2 dynamic ratios and leakage"
```
