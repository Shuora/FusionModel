## Findings

- 当前 V4 仓库的标准实验入口是 `src/run_all_modes.py`、`src/train_fusion_attention.py` 和 `src/train_fusion_attention_stacking.py`，且都要求传入 `--task_name`。
- 支持的任务名仅有 `binary_benign_vs_malicious`、`ustc_multiclass`、`mta_multiclass`、`mfcp_multiclass`。
- 标准数据流程依赖固定目录链路：
  `SourceData/<dataset>` -> `ProcessedData/<task>/pcap_data/{Train,Test}` -> `ProcessedData/<task>/image_data/{Train,Test}`。
- 根目录 `requirements.txt` 未包含 `dpkt`，但 `split_data.py` 实际依赖它解析 pcap，因此 README 需要单独提示安装。
- 当前环境没有 `python` 命令，CLI 校验需使用 `python3`。
- `.gitignore` 会忽略 `/SourceData`、`/ProcessedData`、`/outputs`、`/dataset` 等目录，README 需要明确这些数据与结果不会随仓库提供。
- `tools/sort_outputs_by_mode.py` 和 `tools/split_concat_log.py` 不是可直接复用的跨平台标准脚本，因为内部写死了历史路径。
- 当前仓库的训练默认值已统一为 `batch_size=32`、`num_workers=4`，相关预取参数保持为 `prefetch_factor=2`，以保持代码默认值与文档示例一致。
