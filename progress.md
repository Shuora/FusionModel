## Progress

- 2026-03-30: 读取 `using-superpowers`、`systematic-debugging`、`test-driven-development`、`using-git-worktrees` 技能，并按仓库约束在项目内创建 `.worktrees/codex-pcap-tail-tolerance`。
- 2026-03-30: 复核 `split_data.py` 的异常路径，确认 `expand_raw_samples_to_sessions()` 会在读取异常时跳过整个 `pcap`。
- 2026-03-30: 对 `SourceData/MFCP/Cobalt/Cobalt.pcap` 手工解析，确认文件主体正常，仅文件尾部剩余 `2` 个字节导致下一条 packet header 不完整。
- 2026-03-30: 在 worktree 中运行 `python3 -m unittest tests.test_split_data_tasks`，当前基线通过，准备补回归测试。
- 2026-03-30: 新增 `test_iter_packets_tolerates_truncated_tail_in_pcap`，先运行单测得到失败，确认当前 `.pcap` 读取路径无法处理该场景。
- 2026-03-30: 将 `.pcap` 读取改为内建顺序解析，并在尾部 packet header/data 不完整时保留前序数据包、记录 warning 后返回。
- 2026-03-30: 补充 nanosecond `pcap` magic 支持与对应回归测试，避免本次修复缩窄原有 `.pcap` 兼容范围。
- 2026-03-30: 重新运行 `python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_iter_packets_tolerates_truncated_tail_in_pcap` 与 `python3 -m unittest tests.test_split_data_tasks`，结果通过。
