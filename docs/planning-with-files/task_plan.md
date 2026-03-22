# Documentation Sync Plan (MobileViT + ET-BERT Adapter)

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

- Completed on 2026-03-21

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

- In progress on 2026-03-21

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

# Stage1 Binary Acc 可解释性修复计划（2026-03-22）

## Goal

修复“二分类一直 95%”的观测误导问题，确保训练日志和报告口径与配置一致、可对齐。

## Status

- Completed on 2026-03-22

## Scope

- `src/train.py`
- `src/report.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 按 TDD 新增失败测试：`report` 应按 `config.best_metric` 选择 best epoch；`train` 应输出阈值口径的 `val_acc` 指标。
2. 在训练评估环节增加 `val_acc_at_decision_threshold` 计算与日志/metrics 落盘。
3. 调整 `report` 的 best row 选择逻辑：优先使用 `config.best_metric` 对应列，缺失时回退 `val_macro_f1`。
4. 运行针对性测试与一次端到端 stage1 binary 训练+评估+报告，确认结果可复现并与日志一致。
5. 提交分支、合并到 `dev`，删除 worktree 与分支。

---

# Attention Fusion 改造计划（2026-03-22）

## Goal

将当前二分类主干从门控线性融合（`gate * img + (1-gate) * tls`）改为注意力层面的融合，同时保持训练与下游 `stacking/moe` 接口兼容。

## Status

- Completed on 2026-03-22

## Scope

- `src/models/fusion_model.py`
- `src/train.py`
- `src/evaluate.py`
- `src/stacking.py`
- `src/moe.py`
- `tests/models/test_fusion_model.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 在 `fusion_model` 中引入 attention 融合层（learnable query + cross-attention）。
2. 保持输出字典兼容：保留 `logits_fuse/logits_img/logits_tls/gate`，其中 `gate` 改为 attention 到 image token 的权重。
3. 同步模型构造参数 `num_heads` 到 train/evaluate/stacking/moe。
4. 增加/更新单测，确保融合逻辑为 attention 路径并维持输出形状约束。
5. 运行针对性测试与最小训练验证，记录结果到 findings/progress。

---

# ETBERT Nested Tensor Warning 修复计划（2026-03-22）

## Goal

消除 `ETBertBackbone` 在 PyTorch `TransformerEncoder` 上触发的 nested tensor prototype warning，不改变模型前向语义。

## Status

- Completed on 2026-03-22

## Scope

- `src/models/etbert_backbone.py`
- `tests/models/test_pretrained_backbones.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 先补失败测试，锁定 `TransformerEncoder.enable_nested_tensor` 必须关闭。
2. 在 `ETBertBackbone` 构造 encoder 时显式传 `enable_nested_tensor=False`。
3. 运行 backbone 相关测试与语法校验，确认不改前向输出约束。

---

# 训练早停计划（2026-03-22）

## Goal

为训练过程增加按 `val_acc` 监控的 early stopping，减少无效 epoch。

## Status

- Completed on 2026-03-22

## Scope

- `src/train.py`
- `src/common/structured_logging.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 先补失败测试，锁定 `patience + min_delta` 配置和提前停止行为。
2. 在 `train.py` 增加 `--early-stopping-patience` / `--early-stopping-min-delta` 参数，并按 `val_acc` 做停训判断。
3. 增加 `early_stopping_triggered` 日志事件与配置落盘。
4. 运行针对性测试与语法校验。
