# Time-Blocked Split Design

## Goal

修复 `split_data.py` 的评估泄漏问题，让“每个标签只有一个原始 `pcap`”的数据集也能在 session 提取前完成 Train/Test 隔离。

## Design

- 当某个标签拥有多个原始 `pcap` 时，继续在 raw-capture 级别做 seed-driven split，然后对各自子集独立提取 session。
- 当某个标签只有一个原始 `pcap` 时，在该 capture 内按时间边界切成 Train/Test 两段，再分别累积 payload 并生成 session `.bin`。
- 若 singleton capture 读取失败或损坏，按多 raw 路径一致的容错语义记录错误并跳过该 capture，不中断整个任务。
- 如果同一五元组 session 同时跨越时间边界，则从 Train/Test 两边同时丢弃，避免同一连接被拆到两个集合。
- 保持输出目录不变，仍写到 `pcap_data/{Train,Test}/{label}`，下游训练与图像生成接口无需修改。
- 为避免同 label 目录下覆盖，session 文件名必须纳入 raw capture 身份（而不仅是 `raw_stem + five_tuple`）。
- 输出发布需要事务化：先写 staging，再原子替换最终 `pcap_data`/`metadata`，失败时保持旧结果。
- 事务恢复检查必须在 `split_dataset()` 开始即执行，而不是仅在 promote 阶段执行。

## Implementation Notes

- 时间边界计算：`boundary_ts = min_ts + (max_ts - min_ts) * train_ratio`。
- 当 `min_ts == max_ts`（同时间戳）或 ratio 不在 `(0,1)` 时，回退到 `packet-order` 切分，按包序号切 Train/Test。
- session 命名：`{raw_stem}-{sha1(raw_path)[:10]}.{proto}_{src}_{sport}_{dst}_{dport}`（可读且稳定唯一）。
- 事务发布：
  1. preflight 恢复：`split_dataset()` 启动即检查 `.split_data_backup_*`，若 final 缺失先恢复；
  2. 写入 `processed_root/.split_data_staging/{pcap_data,metadata}`；
  3. staging 完整后再执行替换发布；
  4. 发布失败时回滚恢复旧 `pcap_data/metadata`；
  5. 发布前再次做恢复检查（防御性）；
  6. 成功发布后旧输出整体替换，避免 rerun 残留。
- 单 capture 处理顺序：
  1. 第一遍流式扫描：统计 `packet_count/min_ts/max_ts`；
  2. 选择 `time-boundary` 或 `packet-order` 策略；
  3. 第二遍流式扫描：按策略将 payload 归入 Train/Test 并聚合 session；
  4. 若某 five_tuple 两侧都出现，则整条 session 双侧丢弃；
  5. 剩余 session 写入既有目录结构。
- 多 capture 标签不走时间边界逻辑，仍按 raw 文件切分后再做 `extract_sessions`。

## Verification

- 为单 capture 时间切分补测试。
- 为跨边界 session 丢弃补测试。
- 保留并验证现有 raw-level split 测试。
- 为 singleton 读包失败容错补测试。
- 为同时间戳 fallback 到 packet-order 补测试。
- 为同 label 同 stem 多 raw 的文件名碰撞补测试。
- 为同 processed_root 连续运行的残留污染补测试。
- 为“首次成功后第二次失败仍保留旧输出”补测试。
- 为“backup-only 损坏 + promote 前失败时仍先恢复旧输出”补测试。
- 运行命令：`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
  - RED：`FAILED (failures=1, errors=2)`。
  - 第一轮 GREEN：`Ran 10 tests ... OK`。
  - 第二轮 RED（质量问题复现）：`FAILED (failures=1, errors=1)`。
  - 第二轮 GREEN（质量修复后）：`Ran 12 tests ... OK`。
  - 第三轮 RED（碰撞/残留复现）：`FAILED (failures=2)`。
  - 第三轮 GREEN（本轮修复后）：`Ran 14 tests ... OK`。
  - 第四轮 RED（事务问题复现）：`FAILED (errors=1)`。
  - 第四轮 GREEN（事务修复后）：`Ran 15 tests ... OK`。
  - 第五轮 RED（恢复时机缺口复现）：`FAILED (errors=1)`。
  - 第五轮 GREEN（恢复前移后）：`Ran 16 tests ... OK`。
