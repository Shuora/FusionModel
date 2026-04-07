# Two-Layer Cost-Sensitive Stacking Design (All Tasks)

## Goal

在不替换现有 `MobileViT + CharBERT + attention fusion` 主干的前提下，将当前 `attention_stacking` 从“单层 stacking + soft-voting”为主，升级为“二层 cost-sensitive stacking”为主，面向四个任务统一生效，优先提升弱类召回与 `macro_f1`。

目标优先级：

1. 弱类 `recall` 提升（最高优先级）
2. `macro_f1` 稳定提升
3. 降低 OOF-test gap
4. 保留现有流程兼容性（命令入口、输出目录、日志结构）

## Scope

- 覆盖四个任务：
  - `binary_benign_vs_malicious`
  - `ustc_multiclass`
  - `mta_multiclass`
  - `mfcp_multiclass`
- 升级目标文件：`src/fusion_common.py` 与 `src/train_fusion_attention_stacking.py` 参数层。
- 保留 `attention` 训练流程不变，仅升级 `attention_stacking`。
- 保留当前元学习器生态（`xgboost/lightgbm/catboost/mlp`），但调整为二层结构。
- `soft_voting` 保留为 baseline/backup，不再作为主结果。

## Why Current Pipeline Is Not Enough

当前实现（`src/fusion_common.py:2831` 起）已经具备 OOF、多元学习器、weighted soft-voting、任务后处理，但仍有三个结构性短板：

1. 仍是单层元学习，模型互补关系只做一次映射；
2. voting 权重基本是全局静态，难以处理弱类与难样本局部差异；
3. 后处理以规则增强为主，缺少统一的“代价敏感 + 校准 + 阈值优化”主链路。

这会导致“看起来有集成，但在强评审语境下方法新颖性和严谨性不足”。

## Alternatives Considered

### A. Two-Layer Cost-Sensitive Stacking (Recommended)

- Level-1 生成多元学习器 OOF 概率并做校准；
- Level-2 学习器在 Level-1 输出上再学习，显式引入类不平衡代价；
- 最终做 per-class 阈值优化（目标偏弱类召回 + macro_f1）。

优点：方法链条完整，可解释性和可答辩性强，和现有代码兼容性好。  
代价：训练耗时增加约 1.5x-3x。

### B. Dynamic Ensemble Selection (DES)

为每个样本动态选择最优元学习器。

优点：理论上对边界样本更强。  
代价：实现与调参复杂度高，稳定性风险较大。

### C. Single-Layer + Strong Post-Processing

保持单层 stacking，不改架构，仅强化校准与阈值。

优点：改动小。  
代价：提升上限有限，难满足“复杂集成方法”叙事。

本设计采用 **A**。

## Target Architecture

### Stage 0: Base Fusion Model (Unchanged)

先按当前流程训练 attention 融合基模型，得到 train/test 元特征输入基础。

### Stage 1: Level-1 Meta Learners

对每个 method（`xgboost/lightgbm/catboost/mlp`）执行：

1. 使用 K-fold 生成 OOF 概率 `P_m_oof`；
2. 在全训练集上训练完整模型 `M_m_full`；
3. 对测试集输出 `P_m_test`；
4. 对 `P_m_oof` 进行概率校准器拟合，再映射到 `P_m_test_calibrated`。

校准策略：

- 默认 `temperature scaling`（多分类统一温度）；
- 可选 `isotonic`（样本足够时）；
- 记录 `ECE/Brier`，用于判断校准是否有效。

### Stage 2: Level-2 Cost-Sensitive Blender

构建二层输入特征：

- 所有 `P_m_oof` 拼接；
- 每个 method 的 entropy、margin、top2 gap；
- method 间一致性统计（vote entropy、pairwise KL 近似）；
- 样本难度信号（预测置信度与标签偏差）。

二层学习器默认使用 class-weighted XGBoost（可切换 LR）：

- `sample_weight = inverse_freq_weight * hard_sample_factor`；
- objective 仍为多分类概率输出；
- 训练目标偏向弱类召回。

### Stage 3: Per-Class Threshold Optimization

在 Level-2 OOF 概率上搜索每类阈值向量 `tau_c`：

- 默认目标：`macro_f1 + lambda * minority_recall`；
- 用坐标下降或网格搜索；
- 推理时按 `score_c = p_c / tau_c` 再 argmax。

说明：这一步是提升弱类召回的关键，不等价于简单 voting。

## Data Flow

1. 训练 base fusion model；
2. 提取 train/test 元特征（延续 deterministic meta loader）；
3. Level-1 多 method OOF + full fit + calibration；
4. 训练 Level-2 blender；
5. 在 OOF 调阈值，在 test 应用阈值；
6. 输出主结果（two-layer）、对照结果（single-layer / soft-voting）。

## Error Handling and Safety

- 任一 method 缺依赖或训练失败：跳过该 method，但不中断全流程；
- Level-1 可用 method 少于 2 个：自动降级到 single-layer stacking；
- 校准器失败：回退未校准概率，并记录 warning；
- 阈值搜索失败：回退全 1.0 阈值；
- 保留现有 NaN/Inf 防护和 early-stop 机制，不改变 attention 主训练安全逻辑。

## Evaluation and Reporting

新增并强制落盘：

- `oof_macro_f1`、`test_macro_f1`、`oof_test_gap`（Level-1/Level-2 分开）；
- 弱类 recall（按任务自动识别 minority set）；
- 校准指标：`ECE`、`Brier`；
- 阈值向量与目标函数值；
- 对照组：`soft_voting`、`single-layer stacking`、`two-layer stacking`。

输出到 `metrics.json` 的新增字段建议：

- `stacking_level`: `single` / `two_level`
- `calibration`: method-level calibrator config + metrics
- `thresholds`: per-class tau
- `minority_metrics`: per-class recall before/after

## Defensibility in Review (答辩可讲点)

1. 不是“多模型堆叠”，而是“cross-fit + calibration + cost-sensitive + threshold optimization”的闭环；
2. 使用 OOF 避免信息泄漏，并显式报告 OOF-test gap；
3. 目标函数与任务目标一致（弱类召回优先）；
4. 给出完整消融：
   - 去掉校准会怎样；
   - 去掉二层会怎样；
   - 去掉阈值优化会怎样；
5. 复杂度可控，有降级路径，不是高风险黑盒。

## Commands and Compatibility

- 保持入口脚本不变：`src/train_fusion_attention_stacking.py`。
- 新增参数建议：
  - `--stacking_level {single,two_level}` 默认 `two_level`
  - `--stacking_calibration {none,temp,isotonic}` 默认 `temp`
  - `--stacking_threshold_objective {macro_f1,macro_f1_minority_recall}` 默认后者
  - `--stacking_minority_lambda` 默认 `0.3`
  - `--stacking_oof_folds` 默认 `5`

旧命令不传新参数时应直接可运行。

## Test Plan

### Unit Tests

- Level-1 OOF 与 full-fit 输出维度一致性；
- 校准器数值稳定性与概率归一性；
- Level-2 特征构建正确性；
- 阈值搜索在 toy 数据上可提升 minority recall；
- method 缺失/失败时降级路径可运行。

### Integration Tests

- 现有 `tests/test_stacking_improvements.py` 扩展为 two-level 场景；
- `tests/test_attention_entrypoints.py` 增加新参数透传；
- 小样本 smoke 运行四任务，验证产物与日志字段完整。

### Acceptance Criteria

- 四任务中至少 3 个任务 `macro_f1` 不低于当前实现；
- `mta_multiclass` 和 `mfcp_multiclass` 弱类 recall 至少一个显著提升；
- 无新增训练崩溃类型（NaN/Inf 防线不退化）。

## Documentation Updates Required

实现落地时必须同步：

- `README.md`：新增 two-level stacking 参数与四任务独立命令示例；
- `AGENTS.md`：仅当协作规则变化时更新（本次方法升级本身不强制）；
- `task_plan.md`、`findings.md`、`progress.md`：记录设计与后续推进状态。

## Literature Anchors (Foreign + Chinese Search Strategy)

可作为方法论锚点的英文方向：

- Stacked Generalization / Super Learner
- Cost-Sensitive Learning under Class Imbalance
- Probability Calibration for Deep/Ensemble Classifiers
- Dynamic/Adaptive Ensemble Weighting

中文文献检索按 CNKI 执行（AI 不直接访问 CNKI）：

- 关键词建议：
  - `网络流量分类 集成学习 类别不平衡`
  - `堆叠集成 多分类 阈值优化`
  - `恶意流量检测 概率校准 宏平均F1`
  - `代价敏感 学习 弱类召回`
- 检索条件建议：近 5 年 + 核心期刊/CSSCI/EI。

## Non-Goals

- 不修改 `attention` 主训练逻辑；
- 不引入全新深层神经二层融合网络（先保持可解释的树模型/线性模型）；
- 不在本轮改动数据预处理管线。
