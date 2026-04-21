# MFCP `accuracy>=97%` 冲分方案设计（Score-Chasing 口径）

## 1. 背景与目标

当前 `mfcp_multiclass` 最新稳定 run（`attention_stacking_20260421_131834`）总体 `accuracy≈0.88`，主要短板为 `class 2/4` 混淆。  
本设计按用户明确约束执行：

- 硬性目标：总体 `accuracy >= 97%`
- 允许调整全链路（数据、训练、后处理）
- 保留样本不均衡，目标 `max:min = 2.5~3.0`
- 先执行方案 A（宽松划分），未达标再切方案 C（session 指纹）
- 单次实验预算：`12~24h`

## 2. 方案对比与选型

### 方案 A（先执行，已确认）

通过宽松划分口径（允许近重复跨 `Train/Test`）拉高可达上限，同时保持轻度不均衡并转为 `accuracy-first` 优化目标。

- 优点：在预算内最有机会快速接近 `97%`
- 风险：泛化意义下降，需要与严格口径并排报告

### 方案 B（不采用）

仅改模型与后处理，不改数据划分。

- 优点：实现简单，实验口径更干净
- 风险：按现有基线，难以从 `~88%` 提升到 `97%`

### 方案 C（A 失败后启用）

在 A 基础上增加 session 指纹类特征，强化“记忆近重复样本”能力。

- 优点：冲 `97%+` 概率更高
- 风险：评估可信度进一步下降

**结论：先 A，未达标再 C。**

## 3. 体系结构与组件边界

### 3.1 数据构建层（新增 score-chasing 数据口径）

- 新增 MFCP 专用分布档位：`score_chasing_v1`
- 允许近重复样本跨 split
- 类别分布约束：`max:min` 保持在 `2.5~3.0`
- 不人为固定 `2` 或 `4` 为高频类，由搜索流程自动决定

### 3.2 训练层（accuracy-first）

- 训练监控指标：`val_acc`
- early stop 与 best checkpoint 选择：`val_acc`
- loss：优先 `CrossEntropy`（弱化 focal 倾向）

### 3.3 Stacking 层（two-level 保留）

- 继续使用 `two_level` 与 `xgboost,lightgbm,catboost`
- 阈值优化目标从 `macro_f1_minority_recall` 改为 `accuracy`
- MFCP pair 后处理由固定类名改为“按验证混淆矩阵动态选 top-confusion pair（优先 `2/4`）”，并将调参目标切到 `accuracy`

### 3.4 报告层（双口径并排）

- 主报告：`score_chasing` 口径结果（用于 97% 验收）
- 对照报告：严格口径结果（仅对照，不作本轮失败判定）

## 4. 数据流设计

1. 原始/现有 MFCP 样本输入  
2. 生成 `score_chasing_v1` 划分（含跨 split 近重复）  
3. 输出 `ProcessedData` 新目录（不覆盖严格口径目录）  
4. 运行 `attention_stacking`（accuracy-first 配置）  
5. 产出 `metrics.json/report/confusion_matrix`  
6. 汇总双口径评估并做达标判定

## 5. 失败处理与风险控制

### 5.1 验收规则

- 主门槛：`score_chasing` 测试集 `accuracy >= 97.0%`
- 辅助门槛：`2/4` 混淆项不能连续两轮恶化

### 5.2 运行节奏

- Run-1：A 基线
- Run-2：仅高影响参数微调（`lr/weight_decay/temperature/threshold`）
- 若 Run-2 仍 `<97`：触发方案 C

### 5.3 可追溯性

每次 run 必须落盘：

- 类别计数与 `max:min`
- 近重复跨 split 比例
- seed 与关键参数
- 两套评估口径结果

## 6. 测试与验证策略

### 6.1 配置正确性验证

- 校验 `score_chasing_v1` 的类分布确实满足 `2.5~3.0`
- 校验 split 目录命名、样本计数、日志元数据完整

### 6.2 训练与后处理验证

- 确认 early stop/best model 由 `val_acc` 驱动
- 确认 two-level 与三种 meta learner 全部执行
- 确认 pair 后处理目标已切为 `accuracy`，且目标 pair 选择与当轮混淆诊断一致

### 6.3 结果验证

- 读取最终 `metrics.json` 与 `report*.md`
- 先看主门槛（97%），再看 `2/4` 混淆趋势

## 7. 回滚与隔离

- 新口径目录、参数与报告采用独立命名（`score_chasing_v1`）
- 不覆盖当前严格口径产物
- 任意时点可回退到现有严格流程

## 8. 实施范围声明

本设计服务于“硬性冲分”目标，默认接受宽松评估口径。  
若后续转回泛化优先目标，应重新定义口径并走新的 spec/plan 流程。
