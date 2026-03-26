# Stage2 Unified Cross-Attention Redesign

## Goal

重做 `stage2` 主路径，使其从当前效果不稳定、外置增强退化明显的 `fusion -> stacking -> moe` 方案，切换到一套统一的端到端双分支模型：

- 保留 `图像分支 + 时序/序列分支`
- 引入显式 `cross-attention`
- 保持“一套模型架构”而不是三套独立模型
- 允许不同数据集使用不同训练配方
- 逐个数据集验收，目标是把 `MTA / MFCP / USTC-TFC2016` 最终准确率推进到 `90%+`

## Non-Negotiable Constraints

### Must Keep

1. 必须保留双分支：
   - image branch
   - sequence branch
2. 必须是一套统一模型主架构。
3. 允许按数据集配置不同训练 recipe，但不允许拆成三套完全不同的模型。
4. 文档必须落盘，避免后续再次遗忘为什么旧主线被放弃。

### Must Stop Doing

1. 不能再把当前 `stacking` 作为 `stage2` 主路径。
2. 不能再把当前 `moe` 作为 `stage2` 主路径。
3. 不能继续把“统一默认命令 + 外置后处理增强”当成高分路线。

## Why The Current Main Path Is Being Retired

当前真实运行结果已经说明旧主线不适合作为继续投入的基础：

- `MTA`
  - level1 `top1 ≈ 0.5884`
  - stacking `top1 ≈ 0.4212`
- `MFCP`
  - level1 `top1 ≈ 0.6167`
  - stacking `top1 ≈ 0.6070`
- `USTC-TFC2016`
  - level1 `top1 ≈ 0.8554`
  - stacking `top1 ≈ 0.3553`

这不是“增益有限”，而是主路径在三个数据集上都不能稳定提分，且在 `USTC` 上是灾难性退化。继续围绕这条主线打补丁没有意义。

另外，`--level3-router moe` 的实际运行还暴露了额外问题：

- 会复用已有 `run_dir`
- 会污染根 `config.yaml`
- 没有稳定留下最终 `moe` 产物

因此本轮设计明确将：

- `stacking`
- `moe`

从 `stage2` 推荐主路径中移除。

## Proposed Architecture

采用统一的 `dual-branch multimodal transformer`：

### Branch Encoders

#### 1. Image Encoder

- 输入图像 patch / feature map
- 输出 `image tokens`
- 继续保留现有视觉 backbone 的可迁移部分

#### 2. Sequence Encoder

- 输入当前时序/文本/序列特征
- 输出 `sequence tokens`
- 继续保留现有时序分支的编码能力

### Fusion Core

融合核心不再是浅层拼接或单步 attention，而是显式包含三类交互：

#### 1. Branch-Local Self-Attention

- image tokens 自身建模
- sequence tokens 自身建模

#### 2. Bidirectional Cross-Attention

- `image -> attend sequence`
- `sequence -> attend image`

要求 cross-attention 是模型主干的一部分，而不是实验性附加件。

#### 3. Shared Multimodal Transformer Trunk

- 经过 cross-attention 后，把两个分支投到共享 multimodal token space
- 通过若干层共享 transformer block 做联合建模

### Output Head

统一模型只保留一套，但允许 classifier 接收 `dataset embedding`：

- fused pooled representation
- image branch summary
- sequence branch summary
- dataset id embedding

最终输出为一套统一分类头范式，而不是三套独立 head 代码分支。

## Model Principle

这套设计要解决的不是“再加一个后处理器”，而是把纠偏能力收回到模型内部：

- 什么时候相信图像分支
- 什么时候相信时序分支
- 分支冲突时如何融合
- 不同数据集的类别边界如何在统一架构内表达

旧主线把这些问题丢给外置 `stacking`。新主线要求模型本体学会这些事。

## Training Design

### Core Rule

模型结构统一，但训练 recipe 可以按数据集不同。

### Training Stages

#### Stage A: Shared Stabilization

目的：

- 先把双分支编码器和 multimodal transformer trunk 训练稳定
- 让 cross-attention 学到有意义的跨模态对齐

特点：

- 使用统一结构
- 不追求单个数据集极限分数
- 更偏表示学习稳定性

#### Stage B: Dataset-Aware Fine-Tune

目的：

- 在统一结构不变的前提下，按数据集优化最终分类边界

每个数据集允许不同：

- `epochs`
- `lr`
- `scheduler`
- `class weighting`
- `sampling strategy`
- `early stopping`
- `freeze / unfreeze schedule`
- `loss coefficient`

### What Is Dataset-Specific vs Shared

#### Shared

- model architecture
- tokenization / branch contracts
- fusion trunk
- eval/report format
- run protocol

#### Dataset-Specific

- training recipe
- acceptance target status
- best checkpoint criterion if needed
- imbalance handling strategy

## Protocol Redesign

### Keep One Main Entry

`src.experiments.stage2_multiclass` 继续保留为唯一 `stage2` 主入口。

### Redefine Its Main Path

默认主路径改成：

`preprocessed data -> unified dual-branch cross-attention model -> evaluate -> report`

不再默认执行：

- external stacking
- external moe

### Decommission Old Path

本轮不再引入 `stage2_v2` 并存。

原因：

- 旧主线已经被证明会误导后续实验
- 如果继续共存，后续很容易再次跑回旧路径
- 用户明确要求“之前的删了吧，没用了，效果太差了”

### Recommended Retirement Strategy

为了降低一次性删除过多代码的风险，退场分两层：

#### Layer 1: Immediate Retirement

- 从主流程移除 `stacking/moe`
- 从命令文档移除其“推荐路径”身份
- 从 protocol tests 主线移除对其正向依赖

#### Layer 2: Physical Deletion

在新主线通过最小验收后，再删除：

- `stage2` 默认路径相关旧编排
- 旧 `stacking` 主线路由
- 旧 `moe` 主线路由

## Acceptance Strategy

### Rule

逐个数据集验收，不做“一次三套一起验收”的模糊推进。

### Acceptance Order

1. `MTA`
2. `MFCP`
3. `USTC-TFC2016`

### Why This Order

- `MTA` 目前最弱，且已有历史 `residual` 路线作为可比基线
- `MFCP` 当前中低，需要确认统一 trunk 是否能显著抬升
- `USTC` 当前 level1 不差，但需要避免新 trunk 反而拉低

### Per-Dataset Acceptance Record

每个数据集都必须记录：

- 当前目标
- 当前使用的 recipe
- best validation
- final test result
- 是否达到阶段门槛
- 下一步是否继续

这些结果必须落到文档里，而不是只保存在会话上下文里。

## Documentation Requirements

### Must Write

1. 本设计文档
2. 后续 implementation plan
3. `docs/commands/session-full-experiments.md`
4. 逐个数据集验收记录

### Must Explicitly State In Docs

1. 旧 `stage2` 外置增强主线为什么被放弃
2. 新主线的唯一推荐入口是什么
3. 当前验收到哪个数据集
4. 哪些数据集已经过线，哪些还没过线

## Files Expected To Change

### High Priority

- `src/models/fusion_model.py`
- `src/train.py`
- `src/evaluate.py`
- `src/experiments/stage2_multiclass.py`
- `tests/models/test_fusion_model.py`
- `tests/pipeline/test_protocol_execution.py`
- `tests/pipeline/test_stage2_multiclass_protocol.py`
- `docs/commands/session-full-experiments.md`

### Likely Removed From Main Path

- `src/stacking.py`
- `src/moe.py`
- `tests/pipeline/test_stacking_pipeline.py`
- `tests/pipeline/test_moe_pipeline.py`

注：
这些文件是否“物理删除”取决于 Layer 2 收尾阶段；但它们必须先从主路径退出。

## Risks

### 1. A Single Architecture May Still Be Insufficient

即使允许 dataset-specific recipe，统一架构也未必足以让三个数据集同时达到 `90%+`。

### 2. Validation / Test Coupling May Still Mislead

必须警惕当前 `MTA` 已经出现过“val 更好但 test 没同步”的情况。

### 3. Large Refactor Blast Radius

本轮是项目级重构，不是局部修补，必须按阶段控制变更范围和验收顺序。

## Recommendation

采用以下实施原则：

1. 直接退役旧 `stacking/moe` 主路径，不再修补。
2. 用统一双分支 cross-attention multimodal transformer 重建 `stage2` 主线。
3. 架构统一，recipe 可按数据集不同。
4. 按 `MTA -> MFCP -> USTC` 顺序逐个验收，并把结果持续写入文档。

## Definition Of Success

### Phase Success

- 新主线可稳定训练与评估
- 旧主线退出主路径
- 验收文档机制建立完成

### Performance Success

- `MTA / MFCP / USTC-TFC2016`
- 在统一架构前提下
- 逐个推进到 `90%+`

本设计的目标是为此建立正确的工程路径，而不是在设计阶段虚假承诺“一次实现就必达成”。
