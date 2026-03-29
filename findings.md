## Findings

- 最新 attention 训练日志显示，首个记录点是完整 `Epoch 1` 结束后的汇总，不是第一个 batch。
- `AttentionFusionModel` 使用 `MobileViTConfig()` 和 `build_model(cfg, ...)` 直接实例化编码器，未见加载外部预训练权重。
- `split_data.py` 当前是在 session 级别随机划分 Train/Test，同一原始 pcap 的不同 session 会同时进入 Train 和 Test。
- `binary_benign_vs_malicious/metadata/manifest.json` 中 Train/Test 的 `raw_path` 完全重叠，说明评估存在源文件级泄漏。
