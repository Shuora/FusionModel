## Task
修复 `src/split_data.py` 在读取尾部残缺但主体可用的 `.pcap` 文件时直接报错并跳过整文件的问题。

## Plan
1. 在隔离 worktree 中检查当前 `split_data.py` 的 pcap 读取路径与现有测试覆盖。
2. 先新增一个尾部截断 `pcap` 的回归测试，确认当前行为失败。
3. 仅对 `.pcap` 的尾部不完整 packet header/data 增加容错，保留前面已经读出的包。
4. 运行相关 `unittest` 验证修复，并同步更新本次排查发现与进度记录。

## Constraints
- 不运行 `mvn test`。
- 只放宽 `.pcap` 尾部截断场景，不吞掉其他真实格式错误。
- 先写失败测试，再写生产代码。
