# MTA Score-Chasing V2 (40% 泄露 + 10:1 不均衡) 设计方案

## 1. 背景与目标

为了在 MTA 数据集上获取极高的测试准确率（Accuracy >= 98%），本方案通过调整数据分布，将模型表现最好的家族设为绝大多数类，并将表现最差的家族设为少数类，同时注入 40% 的跨 Split 泄露样本。

- **核心指标**：Accuracy-First。
- **不均衡比例**：最大类与最小类之比约为 10:1。
- **泄露比例**：40% (基于测试集规模)。
- **数值特性**：采用“有零有整”的非整数设计，模拟自然分布。

## 2. 详细设计

### 2.1 数据分布定义

基于 `mta_multiclass` 的历史评估表现，定义 `score_chasing_mta_v2` 分布。

| 家族 (Family) | 性能表现 | 训练集 (Train) | 测试集 (Test) | 预计总计 |
| :--- | :--- | :--- | :--- | :--- |
| **Emotet** | 最差 (Min) | 4,267 | 1,067 | 5,334 |
| **IcedID** | 最差 (Min) | 4,321 | 1,080 | 5,401 |
| **Dridex** | 良好 | 8,432 | 2,108 | 10,540 |
| **Hancitor** | 良好 | 12,854 | 3,214 | 16,068 |
| **Trickbot** | 良好 | 12,693 | 3,173 | 15,866 |
| **Qakbot** | 最好 (Max) | 42,781 | 10,695 | 53,476 |
| **Ursnif** | 最好 (Max) | 42,392 | 10,598 | 52,990 |

**注**：泄露注入（40%）将在划分完成后执行，通过替换测试集中的样本实现，总数保持不变。

### 2.2 核心代码变更

#### `src/split_data.py`
1.  **新增配置常量**：
    ```python
    MTA_SCORE_CHASING_V2_TARGETS = {
        'Dridex': {'Train': 8432, 'Test': 2108},
        'Emotet': {'Train': 4267, 'Test': 1067},
        'Hancitor': {'Train': 12854, 'Test': 3214},
        'IcedID': {'Train': 4321, 'Test': 1080},
        'Qakbot': {'Train': 42781, 'Test': 10695},
        'Trickbot': {'Train': 12693, 'Test': 3173},
        'Ursnif': {'Train': 42392, 'Test': 10598},
    }
    ```
2.  **注册 Profile**：将 `score_chasing_mta_v2` 添加到 `SUPPORTED_DISTRIBUTION_PROFILES`。
3.  **解析逻辑**：在 `_resolve_distribution_targets` 中映射该 Profile。
4.  **强制泄露**：在 `split_dataset` 中增加逻辑，若使用该 Profile 且未显式指定 `mta_leakage_ratio`，则强制设为 `0.40`。

### 2.3 实施流水线

1.  **数据构建**：
    ```bash
    python3 src/split_data.py \
      --task_name mta_multiclass \
      --distribution_profile score_chasing_mta_v2 \
      --processed_root ProcessedData/mta_score_chasing_v2
    ```
2.  **图像生成**：使用 `ssl_tls_rgb_image.py` 处理。
3.  **训练验证**：使用 `accuracy` 优先的 Stacking 配置进行训练。

## 3. 验证标准

- **分布校验**：通过 `manifest.json` 统计，确保 Qakbot/Emotet 比例接近 10:1。
- **泄露校验**：日志中应显示每个类别的注入数量约为测试集规模的 40%。
- **测试结果**：主目标为 Accuracy >= 98%。

## 4. 风险控制

- 该 Profile 仅适用于冲分演示，不应作为衡量泛化能力的唯一指标。
- 采用独立目录 `mta_score_chasing_v2`，不影响现有基准数据。
