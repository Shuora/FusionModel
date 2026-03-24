# Cross-Attention Fusion Plan (2026-03-23)

## Stage1 98+ High-Score Redesign Plan (2026-03-24)

## Goal

围绕 `stage1 binary` 冲击 `98%+` 指标，允许同时重做数据协议、训练策略与 fusion 结构设计，并先输出经过评审的正式方案文档。

## Status

- Completed on 2026-03-24

## Scope

- `src/experiments/stage1_binary.py`
- `src/train.py`
- `src/models/fusion_model.py`
- `src/pipeline_data.py`
- `docs/commands/session-full-experiments.md`
- `docs/superpowers/specs/2026-03-24-stage1-high-score-redesign-design.md`
- `docs/planning-with-files/{task_plan,findings,progress}.md`

## Plan

1. 基于当前 `95.2%` run 与既有排查结论，定义“冲 98+”所允许的协议/训练/结构调整边界。
2. 输出正式 spec，明确新 protocol mode、训练阶段设计、模型角色分工与验证标准。
3. 对 spec 做 reviewer 审查，修正遗漏或自相矛盾项。
4. 将最终 spec 交给用户确认，再进入 implementation planning。

## Notes

- 用户已确认进入 implementation planning 阶段。
- 代码层基础设施已落地，当前剩余主任务是高分 baseline 与最终验收实验。

## Cross-Attention Stabilization + Early Stopping Plan (2026-03-23)

## Goal

修复当前 cross-attention 融合模型在 `stage1 binary` 上塌缩到多数类的问题，并为训练入口补充通用 early stopping 机制。

## Status

- Completed on 2026-03-23

## Scope

- `src/models/fusion_model.py`
- `src/train.py`
- `src/experiments/stage1_binary.py`
- `tests/models/test_fusion_model.py`
- `tests/pipeline/test_train_eval_report.py`
- `tests/pipeline/test_protocol_execution.py`
- `docs/superpowers/specs/2026-03-23-cross-attention-stabilization-design.md`
- `docs/superpowers/plans/2026-03-23-cross-attention-stabilization.md`
- `docs/planning-with-files/{task_plan,findings,progress}.md`

## Plan

1. 将本轮修复设计固化为 spec，并完成 spec review。
2. 将实现拆成 TDD 任务计划，明确模型修复、early stopping 与测试覆盖范围。
3. 创建隔离 worktree，先补红灯测试，再最小实现模型稳定路径与 early stopping。
4. 跑定向回归，确认模型接口、训练流程与 stage1 协议透传不回退。

## Cross-Attention Stage1 Acc Regression Investigation (2026-03-23)

## Goal

只读排查当前 cross-attention 融合实现里，与旧 gate 模型相比最可能导致 `stage1 acc` 降到约 `70%` 的结构性问题，不改代码，只输出结论与证据路径。

## Status

- Completed on 2026-03-23

## Scope

- `src/models/fusion_model.py`
- `src/models/mobilevit_backbone.py`
- `src/models/etbert_backbone.py`
- `src/train.py`
- `src/evaluate.py`
- `src/experiments/stage1_binary.py`
- `tests/models/test_fusion_model.py`
- `tests/models/test_pretrained_backbones.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/superpowers/specs/2026-03-23-bidirectional-fusion-encoder-design.md`
- `docs/superpowers/specs/2026-03-21-stage1-accuracy-training-design.md`
- `docs/superpowers/plans/2026-03-23-bidirectional-fusion-encoder-plan.md`
- `docs/superpowers/plans/2026-03-21-stage1-accuracy-training-plan.md`
- `runs/stage1-binary/config.yaml`

## Plan

1. 核对 cross-attention 设计文档与测试约束，提取 warmup、fusion token、aux heads、loss 的预期行为。
2. 阅读当前 `fusion_model` 与两个 backbone，定位 token 数、信息瓶颈、随机初始化模块和与旧 gate 的结构差异。
3. 阅读 `train/evaluate/stage1_binary`，确认 warmup 是否真的绕过 fusion，以及实际 stage1 run 是否用到了 warmup。
4. 收敛最可能伤害 `stage1 acc` 的结构性问题，并记录证据路径。

## Stage1 Evaluate OOM Fix Plan (2026-03-23)

## Goal

修复 `runs/stage1-binary` 在评估阶段因为整批上 GPU 导致的 CUDA OOM，同时保持现有 `stage1_binary -> evaluate -> report` 协议接口可继续直接使用。

## Status

- Completed on 2026-03-23

## Scope

- `src/evaluate.py`
- `tests/pipeline/test_train_eval_report.py`
- `tests/pipeline/test_protocol_execution.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 用单测复现“evaluate 需要按 batch 多次前向”的行为约束。
2. 在 `src/evaluate.py` 中引入评估批大小参数，并将整批推理改为逐批聚合 logits。
3. 回归 `train_eval_report` 与 `protocol_execution`，确认 stage1 协议执行链路不回退。

## Stage1 Evaluate Failure Investigation (2026-03-23)

## Goal

只读排查 `runs/stage1-binary` 在 stage1 协议评估阶段的失败根因，确认 OOM 与 `state_dict` mismatch 是否属于同一问题。

## Status

- Completed on 2026-03-23

## Scope

- `src/evaluate.py`
- `src/pipeline_data.py`
- `src/models/fusion_model.py`
- `src/experiments/stage1_binary.py`
- `runs/stage1-binary/config.yaml`
- `runs/stage1-binary/checkpoints/best.ckpt`
- `runs/stage1-binary/train.log`
- `runs/2026-03-21/stage1-binary-195511/config.yaml`
- `runs/2026-03-22/stage1-binary-131314/config.yaml`

## Plan

1. 核对 `stage1_binary -> evaluate` 调用链，确认报错发生位置。
2. 检查 `runs/stage1-binary` 当前 config 与 checkpoint key，判断是否为新 cross-attention run。
3. 对比旧 run 的 checkpoint 结构，确认 `Missing key / Unexpected key` 的真实来源。
4. 核对 test split 样本量与 `evaluate.py` 的张量搬运方式，判断 OOM 是否由全量一次性评估触发。

## Goal

将当前 `MobileViTETBertFusionClassifier` 从门控特征融合改为 cross-attention 融合，同时尽量保持训练/评估主链路接口稳定。

## Status

- Completed on 2026-03-23

## Scope

- `src/models/fusion_model.py`
- `src/models/mobilevit_backbone.py`
- `src/models/etbert_backbone.py`
- `tests/models/test_fusion_model.py`
- `README.md`
- `docs/planning-with-files/task_plan.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 明确 cross-attention 的目标形态与接口约束，产出设计并获得用户确认。
2. 在改代码前补测试，先让“cross-attention 输出契约”失败，避免直接凭感觉改实现。
3. 新建 worktree，在隔离工作区实现最小可行 cross-attention 融合，并尽量保持训练/评估链路兼容。
4. 跑模型相关定向测试，必要时补文档并同步 planning 文件。

## Notes

- 设计文档已写入：`docs/superpowers/specs/2026-03-23-bidirectional-fusion-encoder-design.md`
- 实现计划已写入：`docs/superpowers/plans/2026-03-23-bidirectional-fusion-encoder-plan.md`
- 当前设计结论：
  - 保留 `MobileViT + ET-BERT`
  - 删除旧 gate fusion
  - 升级为 `2-layer bidirectional fusion encoder`
  - 优先目标为性能上限，而非旧接口兼容
- 已完成实现与定向验证：
  - `MobileViTBackbone` 新增多尺度 `forward_features`
  - `ETBertBackbone` 新增 `forward_features`
  - `fusion_model` 已切换为 bidirectional fusion encoder
  - `train/evaluate/stacking/moe` 已移除 `gate` 依赖
- 当前新增文档收尾目标：
  - 将实验命令更新为新模型参数口径
  - 删除 `docs/commands/stage2-multiclass-e2e.sh`
  - 将其命令并入 `docs/commands/session-full-experiments.md`

---

# Evaluate Checkpoint 兼容性排查计划（2026-03-23）

## Goal

只读排查 `src/evaluate.py` 在加载 checkpoint 时出现 `Missing key(s)` / `Unexpected key(s)` 的根因，并区分它与同轮出现的 CUDA OOM 是否属于同一问题。

## Status

- Completed on 2026-03-23

## Scope

- `src/evaluate.py`
- `src/models/fusion_model.py`
- `src/models/mobilevit_backbone.py`
- `src/experiments/stage1_binary.py`
- `runs/stage1-binary/config.yaml`
- `runs/stage1-binary/checkpoints/best.ckpt`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 核对当前 `evaluate.py` 实例化的模型结构与配置来源。
2. 读取 `runs/stage1-binary` 的配置与 checkpoint key，确认它期望的架构特征。
3. 对比当前代码与历史 `gate` 版本模型，判断报错来自 run 目录不匹配、代码升级，还是两者叠加。
4. 记录本次只读排查结论，并向用户说明 OOM 与 state_dict mismatch 的区别。

# Documentation Sync Plan (MobileViT + ET-BERT Adapter)

## Session Full 命令重写计划（2026-03-22）

## Goal

基于仓库当前最新实现，删除并重写 `docs/commands/session-full-experiments.md`，让实验命令、参数说明、执行顺序与现有代码严格对齐。

## Status

- Completed on 2026-03-22

## Scope

- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/task_plan.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`
- `src/data/preprocess_runner.py`
- `src/experiments/stage1_binary.py`
- `src/experiments/stage2_multiclass.py`
- `src/train.py`
- `src/evaluate.py`
- `src/report.py`

## Plan

1. 核对仓库当前实验入口、CLI 参数、推荐路径与已有辅助脚本，确认旧文档哪些内容已过期。
2. 提炼“当前可执行的最小完整流程”，统一环境、预处理、stage1、stage2、evaluate、report 的命令风格。
3. 直接删除旧版 `session-full-experiments.md` 内容并按现状重写。
4. 同步更新 findings/progress，记录这次命令文档重构的依据与限制。

## Goal

将项目文档与当前代码实现对齐，重点覆盖架构口径、环境准备、实验命令和验证记录。

## Status

- Completed on 2026-03-18（documentation sync）

## Scope

- `README.md`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 核对当前代码中的 MobileViT 与 ET-BERT 相关实现细节（主干类型、checkpoint/vocab/config 接入能力、能力边界）。
2. 更新 `README.md` 的架构与环境说明，并修正路径与依赖口径。
3. 更新 `session-full-experiments.md` 命令与描述，保持与当前 pipeline 行为一致。
4. 更新 findings/progress，记录当前状态与验证范围（46 项测试）。

---

# 模型结构调研计划（2026-03-22）

## Goal

梳理仓库当前模型结构，明确主模型、两个 backbone、特征融合方式，以及它们与训练/评估管线的连接关系，形成可直接给用户阅读的代码导览。

## Status

- In progress on 2026-03-22

## Scope

- `src/models/fusion_model.py`
- `src/models/mobilevit_backbone.py`
- `src/models/etbert_backbone.py`
- `src/train.py`
- `tests/models/test_fusion_model.py`
- `tests/models/test_pretrained_backbones.py`
- `README.md`

## Plan

1. 定位模型入口与目录结构，确认核心实现文件与测试覆盖位置。
2. 阅读 `fusion_model` 与两个 backbone，梳理输入、输出、融合与分类头的职责分工。
3. 补读 `train.py`、README 与模型测试，确认模型如何被实例化、训练与验证。
4. 更新 findings/progress，并向用户输出结构化说明。

---

# Runtime Support Plan (CUDA + num-workers)

## Goal

为 `train/evaluate` 增加可用的 CUDA 运行时支持与 `--num-workers` 参数，并把推荐参数和命令文档同步到实验文档。

## Status

- In progress on 2026-03-19

## Scope

- `src/train.py`
- `src/evaluate.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 先补测试，覆盖 `device=auto/cpu/cuda` 解析、训练配置持久化、评估阶段 CUDA 不可用时的安全回退。
2. 在 `src.train` 中增加 `--device` 与 `--num-workers`，实现自动选择设备并将模型/张量迁移到目标设备。
3. 在 `src.evaluate` 中复用保存配置中的设备偏好，但当 CUDA 不可用时回退到 CPU，避免评估阶段崩溃。
4. 更新实验命令文档，补充 CUDA / `num-workers` 命令示例，以及针对 `RTX 4060 Laptop 8GB + i7-13700 + 8GB RAM` 的推荐参数。
5. 运行针对性测试并记录结果。

---

# Stage1 Paper Protocol Plan

## Goal

将 `src/experiments/stage1_binary.py` 改为按论文 MVTBA 表 1-3 严格构造 stage1 binary manifest，而不是继续使用近似白名单筛选。

## Status

- Completed on 2026-03-20

## Scope

- `src/experiments/stage1_binary.py`
- `tests/pipeline/test_stage1_binary_protocol.py`
- `tests/pipeline/test_protocol_execution.py`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`
- `docs/superpowers/specs/2026-03-20-stage1-paper-protocol-design.md`
- `docs/superpowers/plans/2026-03-20-stage1-paper-protocol.md`

## Plan

1. 将论文表 1-3 的类别/家族与 train/test 配额整理成协议配置，并记录到 spec 与 findings。
2. 先补 `stage1_binary` 协议测试，覆盖：
   - `torrent` 与 `PUA` 被纳入论文协议
   - ISCX / MTA / MFCP 的精确裁样
   - 样本不足时报错
3. 在 `src/experiments/stage1_binary.py` 中实现论文表驱动的裁样逻辑，移除旧的近似 fallback 行为。
4. 更新 stage1 命令文档，说明当前是“论文类别与数量严格复现”，不是原作者原始 session 列表逐条还原。
5. 运行相关 pytest 回归并记录结果。

---

# Stage1 Binary Result Investigation Plan

## Goal

只读排查当前 `runs/stage1-binary` 的 `0.9642` 是否存在明显实现/配置问题，明确该数值代表的指标、当前协议下是否异常，以及是否存在会压低性能或导致误解的训练/评估因素。

## Status

- Completed on 2026-03-22

## Scope

- `src/experiments/stage1_binary.py`
- `src/train.py`
- `src/evaluate.py`
- `src/pipeline_data.py`
- `src/data/preprocess.py`
- `src/data/dataset_inventory.py`
- `runs/stage1-binary/config.yaml`
- `runs/stage1-binary/eval_test.json`
- `runs/stage1-binary/metrics.csv`
- `runs/stage1-binary/train.log`
- `runs/stage1-binary/report.md`
- `runs/stage1-binary/figures/confusion_matrix_test.csv`
- `outputs/protocol/stage1_binary_manifest.csv`

## Plan

1. 追踪 `eval_test.json` 中 `0.9642` 的生成路径，确认它对应的评估指标与 checkpoint 来源。
2. 核对 `stage1_binary` manifest 构造、`train` 的 train/val 协议和 `evaluate` 的 test 协议，确认 run 的真实实验口径。
3. 汇总 manifest 与混淆矩阵，检查类别分布、样本缺口、潜在 leakage 与可能导致误解的配置因素。

---

# Metrics Parity Check Plan (MVTBA paper vs repo)

## Goal

核对仓库当前“各种指标”的计算方式是否与论文 `MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification` 一致，并明确指出一致项、差异项与原因。

## Status

- Completed on 2026-03-21

## Scope

- `docs/paper/MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification.pdf`
- `src/evaluate.py`
- `src/train.py`
- `src/report.py`
- `src/stacking.py`
- `src/moe.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 提取论文实验评估章节中对指标的定义与展示方式，确认主指标、辅助指标及是否使用 confusion matrix。
2. 检查仓库训练、评估、stacking、moe 的指标计算实现，确认公式、平均方式与零除边界处理。
3. 逐项对照论文与实现，判断“完全一致 / 部分一致 / 不一致”。
4. 将结论同步到 findings/progress，并向用户给出带文件定位的说明。

---

# Paper-Compatible Metrics Plan

## Goal

为当前评估结果增加一套与 MVTBA 论文更兼容的指标输出，同时保留现有 sklearn 工程口径作为主口径。

## Status

- In progress on 2026-03-21

## Scope

- `src/evaluate.py`
- `src/report.py`
- `src/ablation.py`
- `tests/pipeline/test_train_eval_report.py`
- `tests/pipeline/test_ablation_plan.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 先补测试，约束新增 `paper_*` 指标字段与 `paper_macro_f1` 公式。
2. 在评估阶段实现双口径输出，不改变训练选模逻辑。
3. 更新报告与 ablation 汇总，展示兼容指标。
4. 运行针对性测试并同步 findings/progress。

---

# 日志中文化计划（2026-03-22）

## Goal

将项目中用户可见的日志输出尽可能切换为中文表达，同时保留必要英文技术术语与 event code，避免破坏现有检索与测试。

## Status

- Completed on 2026-03-22

## Scope

- `src/common/structured_logging.py`
- `src/experiments/stage1_binary.py`
- `src/ablation.py`
- `tests/common/test_structured_logging.py`
- `tests/data/test_preprocess_pipeline.py`
- `tests/pipeline/test_stage1_binary_protocol.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 统一结构化日志模板：level/module/event 展示中文化。
2. 事件名采用“中文说明 + 英文 event code”双显示，兼容历史检索与断言。
3. 翻译 `stage1_binary` 与 `ablation` 中的直出日志字符串。
4. 更新受影响测试断言并执行针对性回归。

---

# Evaluation Report Tables Fix Plan

## Goal

补齐 `evaluate/report` 的分类明细产物与 Markdown 表格展示，让 stage1 binary 的 run 报告直接包含 confusion matrix 和 classification report，而不只是文件路径。

## Status

- Completed on 2026-03-21

## Scope

- `src/evaluate.py`
- `src/report.py`
- `tests/pipeline/test_train_eval_report.py`
- `tests/pipeline/test_protocol_execution.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`
- `docs/superpowers/plans/2026-03-21-eval-report-tables.md`

## Plan

1. 先补 `train_eval_report` 相关失败测试，锁定缺失的 classification report artifact 与 `report.md` 表格渲染行为。
2. 在 `src/evaluate.py` 中输出 `classification_report_<split>.csv/json`，保留现有 `eval_*.json` 与 confusion matrix 产物。
3. 在 `src/report.py` 中读取 confusion matrix / classification report artifact，并将其渲染为 Markdown 表格。
4. 跑针对性回归，确认 `stacking/moe` fallback 与 stage1 execute 相关测试不回退。

---

# Latest Run Result Review Plan

## Goal

只读核对最新一次 `stage1-binary` 训练/验证/测试产物，解释“为什么只有 96%”对应的真实指标，以及当前协议下更可能的限制因素。

## Status

- Completed on 2026-03-21

## Scope

- `runs/2026-03-21/stage1-binary-195511/config.yaml`
- `runs/2026-03-21/stage1-binary-195511/metrics.csv`
- `runs/2026-03-21/stage1-binary-195511/train.log`
- `runs/2026-03-21/stage1-binary-195511/eval_test.json`
- `runs/2026-03-21/stage1-binary-195511/report.md`
- `outputs/protocol/stage1_binary_manifest.csv`
- `src/train.py`
- `src/experiments/stage1_binary.py`

## Plan

1. 确认最新 run 的训练、验证、测试分别对应哪些指标，避免把 `Top-1`、`Macro-F1`、`weighted avg F1` 混为一谈。
2. 核对训练端的 `val` 产生方式、best checkpoint 选择策略，以及 loss 是否考虑类不平衡。
3. 汇总 manifest 的类别分布、数据集占比和 `capture_id` overlap，判断 96% 更像是模型上限、协议口径，还是明显实现问题。

---

# Accuracy-Oriented Training Plan

## Goal

在不改当前 stage1 数据协议的前提下，通过训练策略优化优先提升当前协议下的 test `accuracy`。

## Status

- In progress on 2026-03-21

## Scope

- `src/train.py`
- `src/evaluate.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/task_plan.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`
- `docs/superpowers/specs/2026-03-21-stage1-accuracy-training-design.md`
- `docs/superpowers/plans/2026-03-21-stage1-accuracy-training-plan.md`

## Plan

1. 先补训练/评估测试，约束分层 validation、可配置 `best_metric` 和二分类 threshold calibration。
2. 在 `src/train.py` 中把派生 val 改为按 label 分层抽样，并支持按 `val_acc` 选择 best checkpoint。
3. 在 `src.train` / `src.evaluate` 中增加二分类 decision threshold 的持久化与复用。
4. 更新实验文档与 planning-with-files 记录，并运行针对性回归。

---

# Protocol Dated Runs Default Plan

## Goal

让 `src.experiments.stage1_binary` 与 `src.experiments.stage2_multiclass` 在默认 `--run-root runs` 下执行时，也自动把产物写到 `runs/YYYY-MM-DD/<run-id>`，无需用户手工显式传日期路径；同时让 stage2 汇总结果记录实际日期分区信息。

## Status

- In progress on 2026-03-24

## Scope

- `src/experiments/stage1_binary.py`
- `src/experiments/stage2_multiclass.py`
- `src/run_dir.py`
- `tests/pipeline/test_protocol_execution.py`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/task_plan.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 先补协议执行层的红灯测试，锁定 stage1/stage2 默认产物落到日期分区，以及 stage2 summary 带出实际 run 目录。
2. 在共享 run-dir helper 中补“按日期构造协议 run 目录”的能力，并让 stage1/stage2 执行层统一复用。
3. 调整 stage2 summary 的落盘位置和字段，保证既能追溯日期分区，也不影响后续按 `runs/<run-id>` 解析最新 run。
4. 更新实验文档与 planning-with-files 记录，并跑协议执行相关回归。
