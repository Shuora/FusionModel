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

---

## Task (2026-03-31 MFCP 样本一致性排查)
排查 `mfcp_multiclass` 处理结果中 `Cobalt` 家族缺失的根因，确认是否与论文口径不一致及其成因。

## Plan
1. 对齐三种口径：论文统计、`SourceData/MFCP` 原始文件统计、`ProcessedData/mfcp_multiclass` 处理后统计。
2. 检查 `src/split_data.py` 在截断 pcap 上的异常处理逻辑，并结合历史提交确认行为变化。
3. 用原始包级统计验证 `Cobalt.pcap` 是否实际包含可提取会话，避免误判为“无有效流量”。
4. 输出根因结论与可复现修复步骤（重建处理数据）。


## Task (2026-03-31 Early Stopping 严谨化)
将融合训练早停默认耐心轮次统一为 8，并审查/加固早停监控指标逻辑，避免模式与指标方向不一致或 NaN 指标导致误停。

## Plan
1. 统一 `EarlyStopping` 与 `train_fusion_model` 的默认 `patience=8`，消除默认值漂移。
2. 增加早停指标方向解析与校验（`auto` 自动推断、手动模式不一致时 fail-fast）。
3. 增加监控值有限性检查，遇到 NaN/Inf 时跳过该轮 early stop 与 ReduceLROnPlateau 更新。
4. 补充单元测试并同步 README、findings、progress。

## Task Status (2026-04-01 early stop 遇到 NaN 未停训)
- [x] 复核日志与训练循环，确认根因是 NaN 分支跳过 early-stop 更新。
- [x] 新增 NaN 场景回归测试。
- [x] 最小改动修复早停逻辑（非有限值按未改善处理并可触发停止）。
- [ ] 运行测试验证并回报结果。
- [x] 运行测试验证并回报结果。
- [x] 增加训练 batch 非有限 loss 保护与回归测试，并通过验证。

## Task (2026-04-02 全量训练改进项 1-5 落地)
根据最新全量训练日志审计结果，实施 5 项改进：
1) 稳定性默认参数；
2) 梯度/参数有限性保护与 fail-fast；
3) `run_all_modes` 子模式重置 seed；
4) `mta/mfcp` 任务默认不均衡策略；
5) `metrics.json` 增加训练健康字段。

## Plan
1. 先补回归测试（默认参数、任务默认策略、seed 重置、fail-fast、metrics 健康字段）并确认红灯。
2. 修改 `src/fusion_common.py` / `src/run_all_modes.py` 完整实现 1-5。
3. 运行目标单测验证行为，并同步 README、findings、progress。
