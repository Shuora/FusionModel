## Progress

- 2026-03-29: 读取 `using-superpowers` 与 `systematic-debugging` 技能，开始按排查流程收集证据。
- 2026-03-29: 定位到最新训练日志 `outputs/binary_attention/logs/attention_dim256_20260328_234045.log`。
- 2026-03-29: 确认首个高指标来自完整 `Epoch 1` 汇总，不是第一个 mini-batch。
- 2026-03-29: 检查 `split_data.py` 和 `manifest.json`，确认 Train/Test 在 `raw_path` 维度完全重叠。
