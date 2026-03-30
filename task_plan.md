## Task
<<<<<<< Updated upstream
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
>>>>>>> e01b7ed (fix: 容忍 pcap 尾部截断并保留已解析数据包)
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
<<<<<<< HEAD
=======
补充项目级 README 和 AGENTS 文档，完整覆盖数据预处理、四个实验任务的独立训练命令，以及后续 AI 改动时的文档同步要求。

## Plan
1. 读取训练、预处理和任务配置入口，确认实际可执行命令与参数来源。
2. 按项目目录结构编写 README，覆盖环境、目录、数据流、预处理步骤、四个任务各自的 attention 与 attention_stacking 命令。
3. 编写 AGENTS.md，约束 AI 在修改代码、脚本、数据流程或命令时同步更新 README 与 AGENTS.md。
4. 做一次基础校验，确认文档中的脚本名、参数名、任务名与仓库实现一致。

## Constraints
- 不运行 `mvn test`。
- 本次以文档更新为主，不改训练逻辑。
- README 中四个任务必须分别给出独立命令，不能只给合并入口。
>>>>>>> Stashed changes
=======
<<<<<<< HEAD
为当前 V4 仓库补写 `README.md`，提供完整命令手册和项目介绍，覆盖环境准备、数据处理、训练、测试、输出说明与常见注意事项。

## Plan
1. 读取训练入口、数据切分脚本、图像生成脚本、测试与依赖定义，确认真实命令与默认目录。
2. 汇总项目背景、任务定义、目录结构、数据流和输出产物，整理适合 README 的说明顺序。
3. 编写 `README.md`，覆盖从零开始的完整命令链路与常用参数示例。
4. 更新 `findings.md` 与 `progress.md`，记录文档整理过程中的关键发现与限制。
5. 运行 CLI `--help` 进行文档验证，确保 README 中的命令与当前 V4 代码一致。

## Constraints
- 不运行 `mvn test`。
- 当前环境命令解释器中不存在 `python`，文档应以 `python3` 为准。
- `tools/` 下部分历史脚本包含硬编码路径，不作为标准流程命令推荐。
=======
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

## Execution Update (2026-03-29, attention plotting crash)
- [x] RED：在 `tests/test_fusion_task_resolution.py` 增加 `pad_mask` 分支 warning 回归用例，锁定 `np.where(..., np.log(...), ...)` 仍会触发 `divide by zero encountered in log`。
- [x] RED：在 `tests/test_fusion_output_artifacts.py` 增加 `load_pyplot_headless()` 用例，要求绘图后端为 `Agg` 且可成功 `savefig`。
- [x] GREEN：在 `src/fusion_common.py` 新增统一 headless `pyplot` helper，并让训练曲线、混淆矩阵、attention curve 全部经由该 helper 落图。
- [x] GREEN：将 entropy 计算改为 `np.log(..., out=..., where=...)`，避免 padding 位置零值触发运行时 warning。
>>>>>>> c926dfcf8bb829c579b702d527601a20ba85ca45
>>>>>>> b23c1ae5f224cc4b17dd1dd42703f3536fdf21d9
=======
>>>>>>> e01b7ed (fix: 容忍 pcap 尾部截断并保留已解析数据包)
=======
补充项目级 README 和 AGENTS 文档，完整覆盖数据预处理、四个实验任务的独立训练命令，以及后续 AI 改动时的文档同步要求。

## Plan
1. 读取训练、预处理和任务配置入口，确认实际可执行命令与参数来源。
2. 按项目目录结构编写 README，覆盖环境、目录、数据流、预处理步骤、四个任务各自的 attention 与 attention_stacking 命令。
3. 编写 AGENTS.md，约束 AI 在修改代码、脚本、数据流程或命令时同步更新 README 与 AGENTS.md。
4. 做一次基础校验，确认文档中的脚本名、参数名、任务名与仓库实现一致。

## Constraints
- 不运行 `mvn test`。
- 本次以文档更新为主，不改训练逻辑。
- README 中四个任务必须分别给出独立命令，不能只给合并入口。
>>>>>>> Stashed changes
