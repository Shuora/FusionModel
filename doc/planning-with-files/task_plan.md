# Task Plan: 恶意 TLS 家族分类实现执行

## Goal
按 `doc/plans/2026-02-23-tls-family-classification-delivery-plan.md` 实现可运行的数据预处理与训练工程骨架，当前已完成 Day1~Day6 最小可运行版（含 stacking/moe/消融清单），并补齐训练日志与进度条规范实现。

## Current Phase
Phase 5

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
- [ ] 根据反馈执行下一批（如需 Day7 统一复现实验与封版）
- **Status:** in_progress

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
