# Task Plan

## SplitCap Resume

### Goal
为 `prepare_dataset.py` 增加 SplitCap 可恢复执行能力（断点续跑），并修复“扩展名为 .pcap 但实际为 pcapng”导致的 SplitCap 失败。

### Steps
1. 增加失败测试：误标后缀 pcapng 自动转码、SplitCap checkpoint 跳过。
2. 在数据准备脚本中实现：
   - 基于魔数检测 pcapng；
   - 对每个 raw capture 的 SplitCap 输出目录写入 `.splitcap.done`；
   - 重跑时跳过已完成目录；
   - SplitCap 失败时继续处理其他文件并汇总警告。
3. 更新 `run_prepare_binary.sh` 透传/默认 `--resume-splitcap`。
4. 运行相关 pytest 验证。
5. 回传可直接使用的运行命令与注意事项。

## Preprocess Workers

### Goal
为数据预处理增加细粒度路径过滤、后处理断点续跑、进度日志和多进程后处理能力。

### Steps
1. 写设计与实现计划文档，固定功能边界。
2. 先补失败测试，覆盖路径过滤、后处理跳过和并行 worker 入口。
3. 实现 `prepare_dataset.py` 的参数、过滤、并行后处理和日志。
4. 更新入口脚本透传新参数。
5. 运行相关测试并汇总结果。

## Preprocess Planning Logs

### Goal
为 `prepare_cached_rows(...)` 增加规划阶段进度日志，避免全量二分类在去重扫描阶段长时间静默。

### Steps
1. 复盘 `Ctrl+C` 栈和当前实现，确认瓶颈在父进程串行 `read_session_bytes(...)` 去重。
2. 补测试，覆盖规划阶段日志输出。
3. 在不改变去重语义的前提下实现 `[plan]` 周期性日志。
4. 运行目标 pytest 验证，并给出新的运行预期。

## Preprocess Planning Performance

### Goal
并行化 `prepare_cached_rows(...)` 中的 payload 读取与 fingerprint 计算，在不改变去重顺序与结果的前提下缩短规划阶段耗时。

### Steps
1. 为规划阶段并行 payload 检查补测试。
2. 实现有序消费的 payload inspection worker。
3. 保持 `[plan]` 日志、empty/duplicate/cache 统计语义不变。
4. 跑回归测试并记录结果。
