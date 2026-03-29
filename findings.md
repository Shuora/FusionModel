## Findings

- 当前实现位于 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，流程是 `discover raw -> extract all sessions -> split sessions`。
- 现有 `split_dataset()` 先调用 `expand_raw_samples_to_sessions(raw_samples)`，再对 `session_samples` 做 `split_task_inputs()`，这会让同一原始 `pcap` 的 session 同时进入 Train/Test。
- 现有测试 [tests/test_split_data_tasks.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py) 已经覆盖 raw-level split，但还没有覆盖“单个 raw capture 需要 time-blocked split”的场景。
- worktree 基线测试 `python3 -m unittest tests.test_split_data_tasks -v` 通过，说明当前改动可以从干净基线开始。

## Final Findings

- 已在 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py) 新增 `iter_session_payloads()`，并让 `extract_sessions()` 复用该迭代器，避免重复解析逻辑。
- 已新增 `split_single_capture_by_time()`：使用 `boundary = min_ts + (max_ts - min_ts) * train_ratio` 做时间边界切分。
- 同一五元组若在边界两侧都有 payload，会在 `split_single_capture_by_time()` 中被统计为 `dropped_cross_boundary` 并从双侧删除。
- `split_dataset()` 现为 hybrid：label 下 `len(raw)>1` 走 raw-level split 后 sessionize，`len(raw)==1` 走时间切分后 sessionize。
- 新增/更新测试覆盖三类关键行为：单 raw 时间切分、跨边界 session 双侧丢弃、多 raw 保持 raw-level split。

## Code Quality Fix Findings

- 已修复 singleton 路径容错：`split_dataset()` 对 `split_single_capture_by_time()` 加了 `try/except`，坏 capture 仅记录 `Error reading ...` 并跳过，不再中断整次任务。
- 已修复 `min_ts == max_ts` 的空侧问题：`split_single_capture_by_time()` 在同时间戳场景回退为 `packet-order` 切分，保证可切时 Train/Test 都有数据。
- 已消除全量 `sorted(packet_items)` 内存退化：改为两遍流式扫描。
  - 第一遍只统计 `packet_count/min_ts/max_ts`；
  - 第二遍做 payload 聚合与 session 归属判定；
  - 不再把所有 `(timestamp, key, payload)` 常驻内存。

## Final Reviewer Fix Findings

- 已修复 session 命名碰撞：`build_session_name()` 现在使用 `build_raw_capture_token(raw_path)`，格式为 `raw_stem-hash`，再拼接五元组。即使同 label 下不同 raw capture 同 stem，也不会覆盖。
- 已修复 rerun 脏数据：成功重跑时 `pcap_data/metadata` 会被完整替换，不保留旧 `.bin`。
- 新增测试确认：
  - 同 label 同 stem 多 raw 的 `manifest.bin_path` 全部唯一；
  - 同 `processed_root` 连续运行两次时，第一次残留 `.bin` 不会保留到第二次结果中。

## Transactional Output Findings

- 已移除“先删后写”流程，改为事务式输出切换：
  - 本次结果先写到 `processed_root/.split_data_staging/{pcap_data,metadata}`；
  - 仅当 staging 完整后才执行替换发布；
  - 发布失败会回滚恢复旧 `pcap_data/metadata`。
- 若检测到上次发布异常中断留下 `.split_data_backup_*` 且对应 final 缺失，会先自动恢复再继续本次发布。
- 中途异常（例如 unknown task、写出异常、manifest 异常）不会清空旧结果，满足“失败不破坏最后一次成功输出”。
- 仍然保留 rerun 去脏能力：成功发布后旧的 `pcap_data/metadata` 会被整体替换，不会残留旧 `.bin`。

## Recovery Timing Fix Findings

- 已将“检测并恢复 `.split_data_backup_*` 且 final 缺失”的逻辑前移到 `split_dataset()` 最开始，确保 discovery 前即恢复到最近一次提交态。
- `promote` 阶段仍保留同一恢复逻辑，作为发布前防御，不影响现有 staging/rollback 机制。
- 新增测试验证：手工制造 backup-only 损坏状态后，用 `unknown_task` 在 promote 前失败，函数会先恢复旧输出再抛出预期异常。
