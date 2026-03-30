## Progress

- 2026-03-30: 读取 `run_all_modes.py`、`train_fusion_attention.py`、`train_fusion_attention_stacking.py`、`fusion_common.py`，确认训练入口与通用参数集合。
- 2026-03-30: 读取 `split_data.py` 与 `ssl_tls_rgb_image.py`，确认原始 pcap 到 `ProcessedData/<task>` 的标准数据处理流程。
- 2026-03-30: 检查 `requirements.txt`、测试文件和 `.gitignore`，确认依赖、任务名范围与仓库忽略目录。
- 2026-03-30: 运行 `python3 ... --help` 验证主要 CLI 可用，确认 README 应统一使用 `python3` 而不是 `python`。
- 2026-03-30: 收集到 `tools/sort_outputs_by_mode.py` 与 `tools/split_concat_log.py` 含硬编码 Windows 路径，不适合写成通用命令。
- 2026-03-30: 将训练默认参数统一调整为 `batch_size=32`、`num_workers=4`，并同步收敛 `prefetch_factor=2` 及相关文档说明。
