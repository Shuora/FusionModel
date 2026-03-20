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
