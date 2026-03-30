## Progress

<<<<<<< Updated upstream
- 2026-03-30: 读取 `using-superpowers`、`systematic-debugging`、`test-driven-development`、`using-git-worktrees` 技能，并按仓库约束在项目内创建 `.worktrees/codex-pcap-tail-tolerance`。
- 2026-03-30: 复核 `split_data.py` 的异常路径，确认 `expand_raw_samples_to_sessions()` 会在读取异常时跳过整个 `pcap`。
- 2026-03-30: 对 `SourceData/MFCP/Cobalt/Cobalt.pcap` 手工解析，确认文件主体正常，仅文件尾部剩余 `2` 个字节导致下一条 packet header 不完整。
- 2026-03-30: 在 worktree 中运行 `python3 -m unittest tests.test_split_data_tasks`，当前基线通过，准备补回归测试。
- 2026-03-30: 新增 `test_iter_packets_tolerates_truncated_tail_in_pcap`，先运行单测得到失败，确认当前 `.pcap` 读取路径无法处理该场景。
- 2026-03-30: 将 `.pcap` 读取改为内建顺序解析，并在尾部 packet header/data 不完整时保留前序数据包、记录 warning 后返回。
- 2026-03-30: 重新运行 `python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_iter_packets_tolerates_truncated_tail_in_pcap` 与 `python3 -m unittest tests.test_split_data_tasks`，结果通过。
=======
- 2026-03-30: 检查仓库根目录，确认项目当前没有 `README.md` 和 `AGENTS.md`。
- 2026-03-30: 读取 `split_data.py`、`ssl_tls_rgb_image.py`、`task_config.py`、`train_fusion_attention.py`、`train_fusion_attention_stacking.py`、`fusion_common.py`，梳理预处理与训练命令来源。
- 2026-03-30: 确认四个任务名与公共训练参数，准备生成完整 README 与 AGENTS.md。
- 2026-03-30: 新增 `README.md`，按项目结构补充四个任务各自的预处理命令与训练命令。
- 2026-03-30: 新增 `AGENTS.md`，要求后续 AI 修改代码、命令或流程时同步更新 `README.md` 与 `AGENTS.md`。
- 2026-03-30: 运行 `python3 -m unittest tests.test_attention_entrypoints tests.test_split_data_tasks tests.test_task_config tests.test_run_all_modes tests.test_ssl_tls_rgb_image tests.test_fusion_task_resolution`，结果通过。
>>>>>>> Stashed changes
