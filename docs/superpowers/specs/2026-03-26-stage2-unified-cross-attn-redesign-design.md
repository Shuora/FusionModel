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

### Label Space Contract

这是实现层面的硬契约，不能留空。

#### Problem

三个数据集的类别数与类别语义不对齐：

- `MTA`: 7 类
- `MFCP`: 6 类
- `USTC-TFC2016`: 10 类

因此不能把三者当作一个天然共享的全局 label space。

#### Required Solution

采用 `shared pre-classifier + dataset-conditioned projection head`：

1. 共享部分：
   - fused representation
   - shared pre-classifier MLP
   - dataset embedding
2. 条件化输出部分：
   - 统一 head 模块内部维护一组按 `dataset id` 选择的 projection
   - 每次前向只激活当前 dataset 对应的输出维度

#### What This Means In Practice

- 仍然是一套统一模型
- 不是三套独立模型
- 但输出 logits 的最后一层允许按 dataset id 选择不同输出维度

#### Training / Eval Contract

- `train.py`
  - loss 只在当前 dataset 对应的 logits 上计算
- `evaluate.py`
  - 推理时只读取当前 dataset 对应的 logits
- `checkpoint`
  - 必须同时保存：
    - dataset id vocabulary
    - 每个 dataset 的 output dim
    - head projection schema

#### Dataset Vocabulary Authority

- `dataset id vocabulary` 必须由单一权威来源定义
- 第一实现中固定来自 `stage2` 任务注册表，顺序固定为：
  - `MTA`
  - `MFCP`
  - `USTC-TFC2016`
- 训练、保存 checkpoint、加载 checkpoint、推理与报告都必须使用同一顺序

#### Explicit Non-Goals

- 不采用“全局并集类别 + 大 mask”方案作为第一实现
- 不把不同数据集的类别 id 强行映射到同一语义空间

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

#### Stage A Execution Protocol

Stage A 不是抽象概念，固定采用以下协议：

1. 训练数据
   - 三数据集联合训练
2. sampler
   - dataset-balanced round-robin sampler
   - 每个 step 从单一 dataset 取一个 batch
   - 三个数据集按近似 `1:1:1` 的 dataset-level 频率轮转
3. loss
   - 使用当前 batch 所属 dataset 的条件化输出 head
   - 不做跨 dataset label 混算
4. checkpoint 选择
   - 每个 epoch 结束后，对三个数据集各自的 validation split 评估
   - 计算 `mean_normalized_val_top1`
   - 作为 Stage A 唯一 checkpoint selection metric
5. normalization rule
   - 先记录每个数据集当前主线 baseline 的 validation / test 参考值
   - Stage A normalization 只用于多数据集联合选择 checkpoint，不用于最终报告
   - 硬定义：
     - `normalized_top1(dataset) = current_val_top1(dataset) / max(reference_top1(dataset), 1e-6)`
     - `mean_normalized_val_top1 = mean(normalized_top1(MTA), normalized_top1(MFCP), normalized_top1(USTC-TFC2016))`
6. stop condition
   - 达到配置 epoch 上限
   - 或 `mean_normalized_val_top1` 连续 `patience` 未提升

#### Stage A Success Definition

Stage A 只要求满足：

- 训练稳定，无 NaN / run 污染 / artifact 缺失
- cross-attention trunk 学到可复用表示
- 三数据集 validation 都优于随机/塌陷状态

Stage A 不以 `90%+` 为目标。

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

#### Stage B Protocol

Stage B 固定为按数据集逐个执行：

1. 从 Stage A best checkpoint 初始化
2. 每个数据集单独 fine-tune
3. 允许该数据集专属 recipe
4. best checkpoint 以该数据集自己的主指标选择

## Acceptance Gates

### Final Goal

- `MTA / MFCP / USTC-TFC2016`
- 最终目标都是 `test top1 >= 0.90`

### Intermediate Gates

为了避免项目长期失控，必须设置中间门槛：

#### Gate 0: Protocol Gate

新主线只有在满足以下条件后，才允许宣布“可取代旧主线”：

- 独立 run 目录
- 独立 config
- 独立 checkpoint
- 独立 eval/report artifact
- 不污染其他 run

#### Gate 1: MTA Gate

在进入 `MFCP` 主攻前，`MTA` 必须满足：

- `test top1 >= 0.70`
- 且不低于当前已知历史强 baseline `0.6977`
- 判定口径：单次固定 seed 主 run 的 best checkpoint 对应 test 结果

#### Gate 2: MFCP Gate

在进入 `USTC` 主攻前，`MFCP` 必须满足：

- `test top1 >= 0.70`
- 判定口径：单次固定 seed 主 run 的 best checkpoint 对应 test 结果

#### Gate 3: USTC Safety Gate

在宣布统一 trunk 成功前，`USTC` 必须满足：

- `test top1 >= 0.86`
- 且不得明显低于当前已知 Level1 baseline `0.8554`
- 判定口径：单次固定 seed 主 run 的 best checkpoint 对应 test 结果

### Why These Gates Exist

- 它们不是最终目标
- 它们是项目推进与是否继续重构的阶段门槛
- 只有阶段门槛明确，旧主线退场才有客观依据

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
- 但暂不物理删除代码文件
- 只有在 `Gate 0 + Gate 1` 达成后，才允许进入 Layer 2

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

## Run Hygiene Requirements

新协议必须把旧主线暴露出的工程问题转成硬约束。

### Required Run Isolation

1. 每次运行必须生成独立 `run_dir`
2. 不得复用旧 run 根目录执行不同 stage
3. 不得覆盖已有 `config.yaml`
4. 不得把后续阶段产物写回 level1 根配置

### Required Artifacts

每个 run 至少稳定落盘：

- `config.yaml`
- `metrics.csv`
- `checkpoints/best.ckpt`
- `checkpoints/last.ckpt`
- `eval_test.json`
- `report.md`
- `figures/learning_curve.png`

若未来引入额外阶段或分析产物，也必须写入独立子目录，不能污染根层配置语义。

### Required Test Coverage

必须新增协议测试锁定：

1. run_dir 唯一性
2. config 不被后续阶段覆盖
3. artifact 完整性
4. dataset-conditioned head 的 label-space 契约

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
