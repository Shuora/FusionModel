# Progress

## 2026-03-18

- 读取并核对了 `README.md`、`docs/commands/session-full-experiments.md` 与当前实现代码（MobileViT / ET-BERT / stage1 / stage2）。
- 更新 `README.md`：
  - 改为“当前架构 + 环境 + 最小流程命令”口径。
  - 明确 MobileViT 本地 checkpoint 复用行为与 ET-BERT adapter 能力边界。
  - 修正仓库路径为 `/home/shuora/Traffic/FusionModel`，补充 `pip install -r requirements.txt`。
- 更新 `docs/commands/session-full-experiments.md`：
  - 命令参数与当前 `stage1_binary.py` / `stage2_multiclass.py` 保持一致。
  - 明确当前管线为 MobileViT + ET-BERT Adapter，并注明“非原始 UER ET-BERT 完整实现”。
- 更新 `docs/planning-with-files/findings.md`，记录本轮文档同步结论与当前 46 项测试验证范围。
- 执行并通过指定回归命令：`python -m pytest -q tests/data/test_etbert_feature_encoder.py tests/models/test_pretrained_backbones.py tests/models/test_fusion_model.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_stage2_multiclass_protocol.py tests/pipeline/test_protocol_execution.py -q`，结果 `46 passed`。

## 2026-03-19

- 读取并更新了 `docs/planning-with-files/task_plan.md`，为本轮运行时能力改造补充计划。
- 按 TDD 增加并跑红了两类测试：
  - `train` 记录 `device` / `num_workers`
  - `evaluate` 在 CUDA 不可用时记录 CPU fallback
- 新增共享运行时解析模块：`src/runtime_device.py`。
- 更新 `src/train.py`：
  - 增加 `--device {auto,cpu,cuda}`
  - 增加 `--num-workers`
  - 训练与验证 batch 按解析后的 device 迁移
  - 将 `device_requested` / `device` / `device_fallback` / `num_workers` 写入 `config.yaml`
- 更新 `src/evaluate.py`：
  - 增加 `--device`
  - 默认复用训练配置中的设备偏好
  - CUDA 不可用时自动回退到 CPU，并在 `eval_*.json` 中记录
- 更新 `src/stacking.py` 与 `src/moe.py`，使其支持 `device` / `num-workers`。
- 更新 `src/experiments/stage1_binary.py` 与 `src/experiments/stage2_multiclass.py`，使 `--execute` 路径也能透传 `device` / `num-workers` 到 train/evaluate。
- 更新 `docs/commands/session-full-experiments.md`：
  - 主命令显式加入 `--device auto`
  - 主训练命令显式加入 `--num-workers 4`
  - 增加针对 `RTX 4060 Laptop 8GB + i7-13700 + 8GB RAM` 的推荐参数说明
- 执行并通过回归命令：`pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_protocol_execution.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py`，结果 `23 passed`。

## 2026-03-20

- 读取并核对论文原文 `docs/paper/MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification.pdf` 第 10-12 页。
- 确认论文 stage1 / Exp. I 协议为：
  - `ISCX + MTA + MFCP`
  - 不含 `USTC`
  - 按 Table 1-3 的类别/家族与 train/test 配额构造数据
- 确认当前 `src/experiments/stage1_binary.py` 仍是近似论文的白名单实现，尚未严格按表 1-3 构造 manifest。
- 新增规格文档：
  - `docs/superpowers/specs/2026-03-20-stage1-paper-protocol-design.md`
- 新增实现计划：
  - `docs/superpowers/plans/2026-03-20-stage1-paper-protocol.md`
- 更新 `src/experiments/stage1_binary.py`：
  - 引入论文 Table 1-3 驱动的协议配置
  - 按 `capture_id` / `family` 与 `split=train|test` 精确裁样
  - 样本不足时直接报错并输出缺口信息
  - 删除“paper subset 匹配不到则 fallback 到全量”的行为
- 更新 `tests/pipeline/test_stage1_binary_protocol.py`：
  - 新增 `PUA` 与精确配额测试
  - 使用最小论文协议夹具更新旧测试
- 更新 `tests/pipeline/test_protocol_execution.py`：
  - 将 stage1 execute smoke tests 改为最小论文协议配额
  - 显式传入 `--num-workers 0`，避免无关的多进程 worker 干扰执行测试
- 执行并通过回归命令：
  - `pytest -q tests/pipeline/test_stage1_binary_protocol.py`，结果 `10 passed`
  - `pytest -q tests/pipeline/test_protocol_execution.py -k 'stage1_binary_execute_runs_train_eval_report or stage1_binary_execute_stacking_reports_stacking_metrics or stage1_binary_execute_moe_reports_moe_metrics'`，结果 `3 passed`
