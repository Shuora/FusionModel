# Stage1 High-Score Redesign Design

## Goal

面向 `stage1 binary` 任务设计一版以 `98%+` 为目标的高分方案，允许同时调整数据协议、训练策略与融合结构，不再局限于当前 cross-attention 稳定化补丁。

## Background

当前最新稳定化 run `runs/stage1-binary-cross-attn-fix` 已经从多数类塌缩恢复到：

- `val_acc ≈ 0.9521`
- `val_macro_f1 ≈ 0.9443`

但它仍然存在两类问题：

1. 性能上限不足。距离目标 `98%+` 仍有明显差距，仅靠当前协议下的小幅调参与稳定化，无法合理承诺继续拉升到该水平。
2. 训练稳定性仍不够。第 7 个 epoch 出现了 `gradient_explosion`，说明当前配置虽然不再塌缩，但仍然不是高分稳态。

同时，既有排查已经表明：

- 当前 `stage1 binary` 存在明显类不平衡
- 不同 dataset 的组成比例不均衡
- 当前 cross-attention 虽然恢复了可训练性，但仍不适合作为唯一主判别路径

因此，本轮目标不应继续定义为“修一下当前模型”，而应定义为“为 stage1 binary 重做一版高分方案”。

## Success Criteria

### Primary

高分方案的主验收标准明确为：

1. `score_optimized` protocol 下，最终一次性 holdout test 的 `top1 / accuracy >= 0.98`
2. 同一 run 的 `val_acc >= 0.98`
3. `test_macro_f1 >= 0.97`

以上三项必须同时满足，不能只靠单个指标达标。

### Stability

训练稳定性的最低验收标准：

1. 不再出现多数类塌缩
2. 不再因 `gradient_explosion` 直接终止训练
3. 至少连续 `3` 个 epoch 内，`val_acc` 波动范围不超过 `±0.5%` 时仍保持高分区间

### Secondary

- 新方案必须保持可复现：
  - protocol mode 明确
  - 训练命令明确
  - checkpoint 选择逻辑明确
- 需要和“论文/旧协议复现”分离，避免把两个目标混成一套命令

## Non-Goals

- 本轮不强求继续维持“严格论文复现”的 protocol 作为唯一主路径
- 本轮不追求最少代码改动
- 本轮不以 preserving 当前 `stage1_binary` 默认行为为最高优先级

## Design Decision

采用“三层联动”的高分路线：

1. 重做 `stage1_binary` 协议模式
2. 重构训练阶段与选模逻辑
3. 调整 fusion 在整体判别中的角色

不推荐继续走“只在当前 protocol 下调参”的保守路线，因为该路线最多只能提供小幅收益，和 `98%+` 目标不匹配。

## Layer 1: Protocol Redesign

### Problem

当前 `stage1_binary` 更接近“论文子集近似 + 不均衡异源混合”，而不是“为高分 binary 分类优化”的协议。

这会导致：

- class balance 不理想
- dataset dominance 明显
- 验证/测试分布和训练目标不完全一致

### Proposed Change

在 `src.experiments.stage1_binary.py` 中新增一个面向高分目标的 protocol mode，例如：

- `score_optimized`

与现有 `paper_balanced` / `paper_strict` 并存，不替换它们。

### Protocol Rules

高分 mode 的核心规则：

1. Binary classes 更强制平衡
   - `normal` 与 `malicious` 的 train / val / test 不再允许严重倾斜
2. Dataset-level balance 更强
   - 避免某一来源在训练或测试中占据绝对主导
3. 显式输出 `train / val / test`
   - 不再依赖训练时从 `train` 内临时切 `val`
4. `val` 要作为调参与选模集合，`test` 只允许最终一次性评估
   - 不允许根据 test 结果反复回改 protocol / 训练策略
5. 优先控制 `val` 与预期 deployment 分布一致，而不是直接贴着 test 调参

### Why

如果目标是“稳定冲高分”，protocol 本身必须为这个目标服务，而不是把“高分优化”和“论文复现”混为一谈。

## Layer 2: Training Strategy Redesign

### Problem

当前训练仍然是单一 `fusion` 阶段直接起跑，虽然已经补了 early stopping 和结构稳定化，但还不够适合高分追求。

### Proposed Change

训练改为显式两阶段：

1. `warmup`
   - 先让单模态分支学稳
   - checkpoint 只围绕单模态可分性与基础判别稳定性
2. `fusion`
   - 在 warmup 权重基础上继续训练
   - 再打开 cross-attention / fusion 主头

### Additional Strategy

高分方案中训练端允许加入：

- 更保守的 base lr
- scheduler
- class weight 或 weighted sampler
- 更严格的 gradient stability
- 可能的 EMA / best checkpoint averaging

### Selection Logic

- 选模指标不再只靠单一 `val_acc`
- 对 binary high-score mode，允许将：
  - `val_acc`
  - `val_macro_f1`
  - threshold stability
  组合成更适合高分目标的 checkpoint 选择依据
- 其中 threshold stability 的可执行定义为：
  - 连续相邻 epoch 的 `decision_threshold` 变化不超过 `0.1`
  - 且对应 `val_acc` 不发生超过 `0.5%` 的反向跳水

## Layer 3: Fusion Model Role Redesign

### Problem

当前 cross-attention 即使经过稳定化，也仍然更适合作为“增强器”，而不适合作为唯一主判别器。

### Proposed Change

模型角色重新分层：

1. image branch 重新成为主判别支柱
2. text / timing branch 提供辅助补充
3. cross-attention 负责增益，而不是完全主导决策

### Structural Direction

推荐结构方向：

- 单模态 pooled branch 保留强 shortcut
- fusion head 更接近 residual ensemble，而不是纯 learned overwrite
- 若需要，可引入显式 branch weighting / confidence-aware combination

### Why

你当前最接近高分的方向，不是让 fusion 更“激进”，而是让它在强单模态表征之上做稳健增益。

## Validation Strategy

新的高分方案必须同时验证三件事：

1. 高分是否真实
   - train / val / test 都要看
   - 不能只看单次 val best
2. 高分是否稳定
   - 不允许靠一次偶然 lucky seed
3. 高分来自哪里
   - 要能区分 protocol 收益、训练策略收益、fusion 结构收益
4. test leakage 是否被控制
   - test 只能在每轮完整方案收尾时跑一次
   - 中间迭代只允许基于 train / val 与 ablation 结果决策

### Required Artifacts

- protocol summary
- train/val/test class distribution
- dataset composition summary
- best checkpoint metrics
- test evaluation report

## Implementation Boundary

本轮实现计划允许修改：

- `src.experiments.stage1_binary.py`
- `src.train.py`
- `src.models.fusion_model.py`
- 必要时的 `src.pipeline_data.py`
- 对应测试与实验文档

## Recommended Execution Order

1. 先新增高分 protocol mode
2. 固定 protocol baseline，并跑第一轮基线结果
3. 再实现 warmup -> fusion 训练流程
4. 固定新训练基线，并跑第二轮基线结果
5. 最后做 fusion role 调整与高分结构优化
6. 逐层记录 ablation，确保每一步都能归因

这个顺序的原因是：

- protocol 不对，后面所有调参都会浪费
- 训练流程不稳，模型结构收益难以体现
- 先稳定分布与训练，再追求结构收益，风险最低

## Risks

1. 如果 protocol 太“为高分而优化”，可能会偏离论文复现目标
   - 解决方式：把 `score_optimized` 和 `paper_*` 明确分开
2. 如果 fusion 继续承担过重职责，仍可能再次出现训练不稳定
   - 解决方式：坚持 residual / shortcut 主导
3. 如果只改一层，结果很可能停留在 `95-97%`
   - 这与当前目标不匹配
4. 如果过度重平衡 protocol，可能会让真实分布下的泛化能力下降
   - 解决方式：在高分 mode 外保留一套贴近原始分布的对照评估
5. 如果 warmup -> fusion 的迁移或冻结策略设计错误，可能会让第二阶段破坏第一阶段收益
   - 解决方式：把冻结/解冻策略写成显式实验开关
6. 如果显式重切分 train / val / test，会造成与旧 run 的横向可比性中断
   - 解决方式：文档中明确标记这是一套新 high-score protocol，而非旧 protocol 的直接续跑

## Decision Summary

为达到 `stage1 binary 98%+`，推荐路线是：

- 新增高分导向 protocol mode
- 显式两阶段训练
- 将 cross-attention 从“主判别器”降为“增益模块”

这是一版新的高分方案设计，而不是对当前 run 的延长线微调。
