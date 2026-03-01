# Task Plan: 恶意 TLS 家族分类实现执行

## Goal
按 `doc/plans/2026-02-23-tls-family-classification-delivery-plan.md` 实现可运行的数据预处理与训练工程骨架，当前已完成 Day1~Day6 最小可运行版（含 stacking/moe/消融清单），并补齐训练日志与进度条规范实现。
在此基础上，新增并落地 `session_full` 论文口径链路（Session PCAP 落盘、非 TLS 保留并标记、两阶段评估编排）。

## Current Phase
Phase 6

## Phases
### Phase 1: Requirements & Discovery
- [x] 审阅执行计划与约束
- [x] 识别实现前风险（工作区隔离、无测试基线）
- **Status:** complete

### Phase 2: Planning & Structure
- [x] 创建隔离 worktree：`feat/tls-family-impl`
- [x] 拆分首批 3 个实现任务（日志/TLS过滤/build_dataset）
- **Status:** complete

### Phase 3: Implementation
- [x] Task 1: 结构化日志模块与测试
- [x] Task 2: TLS strict/relaxed 过滤核心与测试
- [x] Task 3: 数据集输出路径与 TLS/non-TLS 切分骨架与测试
- [x] Task 4: 数据盘点、capture split、泄漏检查与落盘
- [x] Task 5: pcap 会话抽取器（真实输入）与 TLS/non-TLS 分类
- [x] Task 6: preprocess 主流程（进度条 + 结构化日志 + manifest 输出）
- [x] Task 7: RGB 与 TLS token 产物生成器
- [x] Task 8: strict/full 双策略 preprocess 入口
- [x] Task 9: train/evaluate/report CLI 骨架
- [x] Task 10: 融合模型分支（image/tls/fusion）最小可训练版
- [x] Task 11: stacking 元特征导出与 meta-learner 训练
- [x] Task 12: MoE router 训练与评估产物输出
- [x] Task 13: 消融实验矩阵清单自动生成
- [x] Task 14: 训练日志规范（git commit/config 摘要/数据集统计/epoch macroF1/异常显式处理）
- [x] Task 15: 训练进度条规范（train/val 双 tqdm）与 `--stage stacking|moe` 调度入口
- [x] Task 16: 消融结果自动汇总（`runs/ablation/<group>/<run_id>` -> summary CSV）
- **Status:** complete

### Phase 4: Testing & Verification
- [x] 首批 3 任务单测通过
- [x] 集成测试（从 pcap 到 manifest）
- [x] 集成测试（从 pcap 到 rgb/seq shard）
- [x] 训练链路 smoke test（train->evaluate->report）
- [x] stacking/moe/ablation 流程测试
- [x] 全量回归测试通过（31 passed）
- **Status:** complete

### Phase 5: Delivery
- [x] 提交批次总结与待评审点
- [x] 根据反馈执行下一批（如需 Day7 统一复现实验与封版）
- **Status:** complete

### Phase 6: Session Full 论文口径规划与实施
- [x] 完成 brainstorming 决策确认（命名、数据流、自动清理、抽检图保留、两阶段协议）
- [x] 输出设计文档：`docs/plans/2026-02-23-session-full-mvtba-design.md`
- [x] 输出实施计划：`docs/plans/2026-02-23-session-full-mvtba-implementation-plan.md`
- [x] 进入 `executing-plans` 按任务逐条编码与验收
- **Status:** complete

### Phase 7: 分支收尾与交付确认
- [ ] 汇总变更清单与风险点
- [ ] 根据用户意图执行 `finishing-a-development-branch`
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 在 `.worktrees/tls-family-impl` 执行 | 避免污染主工作区未提交内容 |
| 先实现 Day1 + Day2 前半 | 尽快打通真实 pcap 到 manifest 的闭环 |
| 全部采用 TDD 红绿循环 | 提升后续迭代可靠性 |
| parquet 无引擎时回退 CSV | 当前环境无 `pyarrow/fastparquet`，避免流程阻塞 |
| `policy` 与 `filter_mode` 解耦 | 支持 `strict/full` 双目录与同一过滤策略 |
| Day3 训练先用线性头做可运行基线 | 先保证端到端链路与日志/产物完整，再替换为融合模型 |
| Day4 切换到 `TinyFusionClassifier` | 先满足双分支+gate+辅助头训练流程，再扩展更重模型 |
| 新策略命名为 `session_full` | 避免 `paper_mode` 语义不清，便于 CLI/文档理解 |
| `session_full` 默认提特征后自动清理 `tmp_sessions` | 降低磁盘占用，避免接近 2x 原始数据长期膨胀 |
| 保留抽检 RGB 图像 | 支撑可视化审计，不保存全量 PNG 防止 I/O 过大 |
| 阶段1缺任一数据集直接失败 | 保证论文口径一致，避免 partial-run 指标失真 |
| `session_full` 采用 `PCAP -> Session PCAP -> classify_pcap_sessions` | 与论文切分链路一致，并复用现有特征编码入口 |
| 预处理入口与 runner 默认 `cleanup_sessions=True`，`--keep-sessions` 可关闭 | 满足用户“默认自动清理”的要求，同时保留调试开关 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `pytest` 无法导入 `src` | 1 | 增加 `tests/conftest.py` 注入项目根路径 |
| 根目录默认无测试 | 1 | 改为任务级目标测试集验证 |
| 当前无 parquet 引擎 | 1 | 统一实现 parquet->csv fallback 策略 |
| `policy=full` 与过滤模式不一致风险 | 1 | 新增 `filter_mode` 参数并由 runner 显式映射 |
| matplotlib 与 torch 在本机触发非阻塞 warning | 1 | 保留 warning，不影响测试通过与主流程执行 |
| 融合模型接入后需兼容 warmup/fusion 两种损失路径 | 1 | 用 `_loss_and_logits` 统一分支逻辑 |
| 训练入口 `stage=stacking|moe` 未实际调度子流程 | 1 | 在 `train.py` 增加 stage dispatch，自动调用 `src/stacking.py` 或 `src/moe.py` |

---

# Task Plan: 仓库运行标准体检（2026-03-01）

## Goal
核验当前仓库是否满足“可运行标准”，覆盖环境、数据集、代码入口、测试与最小执行链路，并给出结论与修复建议。

## Current Phase
Phase 1

## Phases
### Phase 1: 基线盘点
- [x] 读取技能规范并启用 planning-with-files
- [x] 盘点仓库结构、README 运行命令与入口脚本
- [ ] 同步数据集命名与目录一致性
- **Status:** in_progress

### Phase 2: 环境与依赖核验
- [ ] 核验 Python/关键库可导入
- [ ] 核验 CLI 入口可启动（--help）
- **Status:** pending

### Phase 3: 代码可执行性核验
- [ ] 运行测试集（至少全量 pytest）
- [ ] 运行最小链路 smoke（preprocess/train/eval/report 中至少一段）
- **Status:** pending

### Phase 4: 结论与整改建议
- [ ] 输出通过/风险分级
- [ ] 给出优先级修复项
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 以 README 作为“运行标准”主准绳 | 仓库中未发现 requirements/environment 锁定文件 |
| 先做只读核验，再做命令执行核验 | 降低对现有数据和产物的扰动 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| N/A | N/A | N/A |

## 2026-03-01 体检结论更新

### Phase 1: 基线盘点
- [x] 读取技能规范并启用 planning-with-files
- [x] 盘点仓库结构、README 运行命令与入口脚本
- [x] 同步数据集命名与目录一致性
- **Status:** complete

### Phase 2: 环境与依赖核验
- [x] 核验 Python/关键库可导入
- [x] 核验 CLI 入口可启动（--help）
- **Status:** complete

### Phase 3: 代码可执行性核验
- [x] 运行测试集（至少全量 pytest）
- [x] 运行最小链路 smoke（preprocess/train/eval/report 中至少一段）
- **Status:** complete

### Phase 4: 结论与整改建议
- [x] 输出通过/风险分级
- [x] 给出优先级修复项
- **Status:** complete

## Decisions Made (2026-03-01)
| Decision | Rationale |
|----------|-----------|
| 以 MTA 做 smoke 数据集 | 体量适中，能快速验证真实链路 |
| 判定“部分达标而非完全达标” | 关键阶段1流程存在 ISCX 命名阻塞风险 |

## Errors Encountered (2026-03-01)
| Error | Attempt | Resolution |
|-------|---------|------------|
| 阶段1要求 `ISCX`，而原始目录为 `ISCX-VPN-NonVPN-2016` | 1 | 定位为代码与数据命名契约不一致，需目录别名或代码映射修复 |

## 2026-03-01 修复闭环

### Phase 5: 运行标准修复项执行
- [x] 修复 stage1 的 ISCX 数据集命名兼容
- [x] 补充环境锁定文件并更新 README
- [x] 完成回归验证
- **Status:** complete

## Decisions Made (修复项)
| Decision | Rationale |
|----------|-----------|
| 采用代码别名映射，而非强制重命名目录 | 避免改动大体积原始数据目录，兼容历史产物 |
| `environment.yml` 采用 `python+pip` 最小锁定方案 | 当前环境主要包来自 pip，先保证可复现与可安装 |

### Phase 6: review反馈收敛（2026-03-01）
- [x] 忽略 torch 环境项（按用户确认）
- [x] 补充 stage1 输出字段文档说明
- [x] 增加 ISCX 主目录优先级测试覆盖
- [x] 完成回归验证
- **Status:** complete
