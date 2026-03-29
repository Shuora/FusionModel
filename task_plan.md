## Task
修复 `split_data.py` 的数据泄漏问题：当某个标签只有一个原始 `pcap` 时，不再先抽取全部 session 再随机划分 Train/Test，而是先在原始抓包内按时间切分，再分别进行 session 提取；当某个标签有多个原始 `pcap` 时，保持 raw-capture 级划分。

## Plan
1. 为“单个 raw capture 的时间阻断切分”补充失败测试，覆盖正常切分和跨边界 session 丢弃。✅
2. 重构 `src/split_data.py`，引入按时间切分单个 capture 的逻辑，并保留多 capture 标签的 raw-level split。✅
3. 更新 manifest 和任务文档，明确当前 split policy 与单 capture 的处理方式。✅
4. 运行相关 `unittest` 验证行为正确，再做代码复核。✅
5. 修复 code review 的 3 个质量问题（singleton 容错、同时间戳 fallback、流式多遍扫描）。✅
6. 修复最后一轮 review 问题（session 文件名碰撞、rerun 脏数据残留）。✅
7. 修复高危问题：输出切换改为事务式 staging->commit，失败时保持旧结果不变。✅
8. 修复事务恢复缺口：启动即恢复遗留 backup/final 缺失状态。✅

## Final Strategy
- 多 raw capture 标签：先按 raw capture split（seed + ratio），再分别 sessionize。
- 单 raw capture 标签：先按时间边界切 Train/Test，再在各自集合内累积 session payload。
- 五元组跨越时间边界：该 session 从 Train/Test 同时丢弃，避免泄漏。
- 输出结构保持 `pcap_data/{Train,Test}/{label}`，下游接口不变。
- session 文件名使用 `raw_stem + raw_path_hash + five_tuple`，保证同 split/label 目录下稳定唯一。
- 成功重跑时通过事务发布整体替换 `pcap_data/metadata`，不会残留旧 bin。
- 输出发布改为事务式切换：先写 `.split_data_staging`，仅在产物完整后替换最终 `pcap_data/metadata`。
- 若 discovery/write/manifest 任一步失败，旧 `pcap_data/metadata` 保持不变。
- `split_dataset()` 启动即执行 preflight 恢复：若存在 `.split_data_backup_*` 且 final 缺失，先恢复到最近提交态再继续。

## Verification
- RED（先失败）：`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
  - 结果：`FAILED (failures=1, errors=2)`，失败点符合预期（缺少 `iter_session_payloads` 和 multi-raw 行为不符）。
- GREEN（实现后通过）：同一命令复跑。
  - 结果：`Ran 10 tests ... OK`。
- 第二轮 RED（质量修复前）：同一命令复跑。
  - 结果：`FAILED (failures=1, errors=1)`，失败点为 singleton 读包异常中断与同时间戳全部落入 Train。
- 第二轮 GREEN（质量修复后）：同一命令复跑。
  - 结果：`Ran 12 tests ... OK`。
- 第三轮 RED（碰撞/脏数据问题复现）：同一命令复跑。
  - 结果：`FAILED (failures=2)`，失败点为 bin_path 碰撞与 rerun 残留。
- 第三轮 GREEN（本轮修复后）：同一命令复跑。
  - 结果：`Ran 14 tests ... OK`。
- 第四轮 RED（事务问题复现）：同一命令复跑。
  - 结果：`FAILED (errors=1)`，失败点为失败 rerun 后旧 manifest 丢失。
- 第四轮 GREEN（事务修复后）：同一命令复跑。
  - 结果：`Ran 15 tests ... OK`。
- 第五轮 RED（恢复时机缺口复现）：同一命令复跑。
  - 结果：`FAILED (errors=1)`，失败点为 promote 前失败时未恢复旧输出。
- 第五轮 GREEN（恢复前移后）：同一命令复跑。
  - 结果：`Ran 16 tests ... OK`。

## Constraints
- 不运行 `mvn test`。
- 修改代码前必须在 git worktree 中进行。
- 优先保持下游目录结构和训练入口不变。

## Execution Update (2026-03-29, attention outputs)
- [x] RED：新增 `tests/test_fusion_output_artifacts.py`，覆盖 `metrics.json` / `epoch_metrics.csv` 导出与 run 目录隔离。
- [x] GREEN：在 `src/fusion_common.py` 新增统一 helper：
  - `prepare_run_output_dir()`：`output_dir/<run_name>` 自动防重名（`_2/_3` 后缀）。
  - `build_run_artifact_paths()`：固定文件名路径映射。
  - `export_metrics_artifacts()`：落盘 `metrics.json` + `epoch_metrics.csv`。
- [x] `run_fusion_experiment()`/`run_stacking_experiment()` 改为每次 run 独立子目录，固定文件名保存核心产物。
- [x] `collect_attention_diagnostics()` 支持固定文件名参数，attention 诊断图稳定输出 `attention_curve.png`（可用时）。
- [x] 兼容 `run_all_modes.py`：`mode=all` 同根目录执行时，attention 与 stacking 通过 run 子目录隔离。
