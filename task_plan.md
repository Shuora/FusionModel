# Task Plan

## Goal
为 `prepare_dataset.py` 增加 SplitCap 可恢复执行能力（断点续跑），并修复“扩展名为 .pcap 但实际为 pcapng”导致的 SplitCap 失败。

## Steps
1. 增加失败测试：误标后缀 pcapng 自动转码、SplitCap checkpoint 跳过。
2. 在数据准备脚本中实现：
   - 基于魔数检测 pcapng；
   - 对每个 raw capture 的 SplitCap 输出目录写入 `.splitcap.done`；
   - 重跑时跳过已完成目录；
   - SplitCap 失败时继续处理其他文件并汇总警告。
3. 更新 `run_prepare_binary.sh` 透传/默认 `--resume-splitcap`。
4. 运行相关 pytest 验证。
5. 回传可直接使用的运行命令与注意事项。
