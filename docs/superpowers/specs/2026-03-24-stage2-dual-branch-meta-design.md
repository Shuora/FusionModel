# Stage2 Dual-Branch Meta-Enhancement Design

## Goal

为 `stage2 multiclass` 设计一条独立于 `stage1` 的高分路线，在不牺牲 `stage1` 当前高水准的前提下，让：

- `MTA`
- `MFCP`
- `USTC-TFC2016`

三个 `stage2` 数据集任务都以各自最终 `test top1 / accuracy >= 0.96` 为目标。

本方案明确要求：

1. `stage2` 核心一级模型必须保留“双分支 + 注意力融合”
2. 允许在一级 fusion 之后增加二级增强层
3. 允许最终形成比 `stage1` 更复杂的独立 `stage2` 训练体系

## Background

当前 `stage1` 已有高水准 run，例如：

- `runs/2026-03-22/stage1-binary-attn-215753/eval_test.json`
  - `top1 ≈ 0.9610`

但当前直接复用 fusion 主模型思路到 `stage2`，效果明显不足：

- `runs/stage2-mta-residual/eval_test.json`
  - `top1 ≈ 0.6977`
- `runs/stage2-mta-residual-balanced/eval_test.json`
  - `top1 ≈ 0.6334`

同时，`MTA` 已经暴露出典型问题：

1. 单一 fusion 主模型对多类不平衡场景纠错能力不足
2. 单靠 `class_weight / best_metric / scheduler` 这类训练超参微调，无法支撑接近 `96%` 的跨数据集目标
3. 若继续强行让 `stage1` 与 `stage2` 共用一条训练主线，高概率会互相牵制

因此，本轮 `stage2` 的正确目标不是“继续调当前 fusion 模型”，而是“保留双分支注意力融合作为一级主识别器，并在其后增加可学习的二级增强体系”。

## Success Criteria

### Primary

`stage2` 三个数据集都必须满足：

1. `MTA test top1 / accuracy >= 0.96`
2. `MFCP test top1 / accuracy >= 0.96`
3. `USTC-TFC2016 test top1 / accuracy >= 0.96`

以上以最终增强后输出为准，不以一级 fusion 单独输出为准。

### Structural

必须同时满足以下结构约束：

1. 一级模型保留双分支与注意力融合
2. 二级层只能增强或纠偏一级 fusion，不能替代 fusion 主位
3. `stage1` 与 `stage2` 训练主线解耦，`stage2` 的升级不能要求同时改坏 `stage1`

### Secondary

- 保留可解释的逐层增益归因：
  - 一级 fusion baseline
  - 二级 meta-classifier 增益
  - 可选三级 router/moe 增益
- 保留统一的 `stage2` protocol runner，避免三套数据集彻底散成三份孤立脚本

## Non-Goals

- 本轮不追求 `stage1` 与 `stage2` 方法外观完全一致
- 本轮不要求最终 `stage2` 仍是单一 fusion 主模型
- 本轮不以 `macro_f1` 作为最终成功标准
- 本轮不要求三套数据集完全共享同一组超参数

## Design Decision

采用 `fusion-first, meta-second` 的两级结构：

1. 一级：双分支注意力融合主模型
2. 二级：`meta-classifier`
3. 第一版二级实现：`stacking meta-classifier`
4. 后续若需要，再在二级之后叠加 `router/moe`

不推荐的路线：

- 只继续硬调当前共享 fusion 主线
- 退化成纯单模态专家拼装
- 一开始就直接上重型 `moe` 而没有稳定二级基线

## Architecture

### Level 1: Fusion Backbone

一级主模型继续使用双分支多模态识别：

- image branch
- text branch
- attention fusion backbone

一级模型的主职责是输出一个强的 fusion 判别结果，同时暴露足够的高层信息供二级器学习纠偏。

一级模型至少需要导出：

1. `fusion logits`
2. `image logits`
3. `text logits`
4. 融合置信度统计
5. 分支一致性统计
6. 轻量 attention / fusion 摘要特征

### Level 2: Meta-Classifier

二级增强器的角色不是重新看原始模态，而是学习：

- 什么时候应该相信一级 fusion
- 什么时候一级 fusion 的错误是可预测的
- 哪类样本更适合根据分支冲突或置信度重新修正

设计层面统一称为 `meta-classifier`。

第一版实现采用：

- `stacking meta-classifier`

即：

- 训练时使用 OOF / validation 风格特征
- 推理时输入一级 fusion 模型导出的高层信号
- 输出最终修正后的类别预测

### Optional Level 3: Router / MOE

在 `v1` 跑通之后，若二级 `stacking` 仍不能把三套数据集全部推到 `96%`，允许再叠加一个更强的后置增强层：

- router
- moe
- learned gating over correction heads

但这个三级增强层只能建立在已有稳定的二级基线上，不应作为第一版起步。

## Components

建议将 `stage2` 体系拆成 4 个清晰模块：

### 1. Fusion Backbone

负责：

- 原始 `rgb + text tokens` 输入
- 双分支注意力融合前向
- 输出一级判别与高层摘要特征

### 2. Feature Dump / OOF Generator

负责：

- 在 train/val 上导出二级训练样本
- 形成 OOF 风格的 meta features
- 避免二级层直接吃一级模型的训练集自拟合输出

### 3. Meta Enhancer

负责：

- 读取一级导出的 meta features
- 训练二级 `meta-classifier`
- 输出最终修正后的类别

### 4. Stage2 Protocol Runner

负责串起完整流程：

1. 一级 fusion 训练
2. OOF / validation 特征导出
3. 二级 meta-classifier 训练
4. 最终 test 评估

## Meta Features

第一版二级输入特征固定为轻量、高信号组合，不直接引入整段高维 token 序列。

### Group 1: Primary Logits

- `fusion logits`
- `image logits`
- `text logits`

这是最核心的 stacking 输入。

### Group 2: Confidence Features

对三路输出分别计算：

- softmax max probability
- `top1 - top2 margin`
- entropy

这些特征用于表达样本级不确定性。

### Group 3: Branch Agreement Features

包括：

- `fusion top1 == image top1`
- `fusion top1 == text top1`
- `image top1 == text top1`
- 三路概率分布之间的距离统计
  - `L1`
  - cosine
  - 必要时可扩展 KL

这些特征是二级纠偏的关键依据。

### Group 4: Fusion Summary Features

只保留轻量摘要，不直接喂整段 token：

- attention mean / std
- fusion representation norm
- image/text contribution summary

目标是提供“融合过程是否稳定、是否偏某一分支”的压缩信号。

### Group 5: Training-Time Helper Signals

只用于训练二级器时的样本构造与分析：

- OOF correctness flag
- dataset id
- optional class prior stats

这些信号是否进入最终推理输入，可在实现时单独配置。

## Training Protocol

### Stage2 v1 Flow

第一版固定采用以下四步，不额外开放复杂分支：

1. 训练一级 `fusion backbone`
2. 导出 OOF / validation meta features
3. 训练二级 `stacking meta-classifier`
4. 用“一级 fusion + 二级增强”的最终输出跑 test

### Train Level 1

- 每个数据集单独训练自己的一级 fusion 模型
- 不强求三套数据共享一组最佳超参数
- 优先把一级 fusion 本体推到各自最优

### Build OOF Features

二级训练必须使用 OOF / validation 风格特征，而不是直接使用一级模型在训练集上的自拟合预测。

否则二级 stacking 容易学成“记住一级训练误差”，不能泛化到 test。

### Train Level 2

- 每个数据集单独训练自己的 `stacking meta-classifier`
- 一级 fusion 是固定输入来源
- 二级器只负责纠偏，不回传梯度去重写一级 fusion

### Inference

test 时的数据流固定为：

`raw sample -> fusion backbone -> meta features -> stacking meta-classifier -> final prediction`

## Validation Strategy

### Metrics

最终主指标只看：

- `test top1 / accuracy`

但研发阶段仍需要同时记录：

- 一级 fusion 的 test top1
- 二级增强后的 test top1
- 各类 confusion matrix
- per-class classification report

这样才能确认增益究竟来自哪里。

### Required Ablation

至少要保留三层结果：

1. 一级 fusion baseline
2. `fusion + stacking meta-classifier`
3. `fusion + stacking + optional router/moe`

任何更重结构若不能超过上一层，就不应进入主线。

## Execution Order

推荐实施顺序：

1. 先把 `stage2` protocol 独立出来
2. 先实现 `fusion + stacking meta-classifier` 的最小闭环
3. 先用 `MTA` 打通闭环并确认二级层能带来真实增益
4. 再扩到 `MFCP / USTC-TFC2016`
5. 若仍不足，再叠加 `router/moe`

这个顺序的原因是：

- 先验证“二级纠偏”是否对当前 fusion 有真实收益
- 再决定是否需要更重的三级增强
- 避免一开始就把问题复杂化到难以归因

## Risks

### Risk 1: 一级 Fusion 太弱，二级层无从纠偏

如果一级 fusion 本体对三套数据集都过弱，则二级 stacking 只能放大噪声，无法带来实质收益。

### Risk 2: 二级层过拟合 OOF 特征

如果 OOF 构造不严格，二级器会学成训练集记忆器，test 不增反降。

### Risk 3: 特征过多导致二级层学不稳

第一版若直接引入大块高维 hidden states，二级层很容易在小数据集上失稳，因此本设计明确要求“先轻量摘要，再逐步增加”。

### Risk 4: 三套数据集最优 recipe 分化过大

若 `MTA / MFCP / USTC` 的最优路径差异明显，可能需要保留统一框架下的 dataset-specific config，而不是强求一组超参通吃。

## Implementation Boundary

本轮后续 implementation plan 允许修改：

- `src/train.py`
- `src/experiments/stage2_multiclass.py`
- `src/stacking.py`
- `src/moe.py`
- `src/models/fusion_model.py`
- 必要时新增 `stage2` 相关 feature dump / meta-classifier 模块
- 对应测试与实验文档

## Summary

本方案的核心判断是：

- `stage1` 与 `stage2` 必须解耦
- `stage2` 必须保留双分支注意力融合作为一级主识别器
- 想要把三个 `stage2` 数据集都推进到 `96%`，必须引入后置二级增强层
- 第一版最合理的落地方式是：
  - `dual-branch attention fusion`
  - `stacking meta-classifier`
  - 必要时再上 `router/moe`
