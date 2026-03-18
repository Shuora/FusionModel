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
