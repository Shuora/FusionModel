# MTA 数据集泄露比例调整参数设计

## 1. 背景与目标

为了在 MTA 数据集（`mta_multiclass`）上模拟或调整“冲分”模式下的数据泄露情况，用户需要一个独立的参数来控制跨 Split 的重复样本注入比例。此变更需在不影响 MFCP 既有 `score_chasing_v1` 逻辑的前提下实现。

- **目标任务**：`mta_multiclass`
- **核心需求**：新增比例参数，采用与 MFCP 类似的泄露注入方式，默认比例参考为 40%。
- **约束**：不改动原有代码逻辑（保持 MFCP 等任务行为一致）。

## 2. 详细设计

### 2.1 CLI 参数扩展

在 `src/split_data.py` 的 `build_parser` 中增加以下参数：

- `--mta_leakage_ratio` (float, default=0.0): 专门用于 MTA 任务的泄露比例。当设置为大于 0 的值（如 0.40）时，将触发泄露注入。

### 2.2 核心逻辑变更

#### `split_task_inputs` 函数
- 新增参数 `mta_leakage_ratio`。
- 在常规划分逻辑执行完毕后，增加针对 `mta_multiclass` 的判断分支：
  ```python
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
  ```

#### `split_dataset` 函数
- 同步透传 `mta_leakage_ratio`。

### 2.3 兼容性保证

- **MFCP 任务**：由于 `mta_leakage_ratio` 默认值为 0.0，且判断条件限制了 `task_name == 'mta_multiclass'`，因此不会对 MFCP 的 `score_chasing_v1` 或其他配置产生任何影响。
- **默认行为**：如果不设置该参数，MTA 数据集的划分行为保持现状。

## 3. 验证方案

### 3.1 单元测试
在 `tests/test_split_data_tasks.py` 中新增测试用例：
- `test_split_task_inputs_mta_leakage_injection`: 验证当 `task_name='mta_multiclass'` 且设置了 `mta_leakage_ratio` 时，Train 和 Test 之间确实产生了重复前缀的样本。
- `test_split_task_inputs_mta_leakage_respects_task_filter`: 验证非 MTA 任务设置此参数时不会触发泄露。

### 3.2 手动验证
运行划分命令并检查日志：
```bash
python3 src/split_data.py --task_name mta_multiclass --mta_leakage_ratio 0.40
```
检查 `metadata/split_data.log` 中是否出现 `MTA leakage injected` 的记录。

## 4. 实施计划

1. 修改 `src/split_data.py` 增加参数和注入逻辑。
2. 更新单元测试 `tests/test_split_data_tasks.py`。
3. 执行测试并验证。
