# Task Plan: 恶意TLS家族分类任务细化与执行蓝图

## Goal
基于 `doc/恶意tls家族分类方案计划书.md` 输出可直接执行的端到端详细计划，覆盖数据、建模、训练、评估、交付与风险控制。

## Current Phase
Phase 8（blocked）

## Phases
### Phase 1: 需求提取与约束确认
- [x] 读取初步方案并提取硬约束
- [x] 对齐仓库目录与当前资产
- [x] 记录发现到 `findings.md`
- **Status:** complete

### Phase 2: 详细执行计划设计
- [x] 将初步方案拆解为可执行工作流
- [x] 补齐每个工作流的输入/输出与验收标准
- [x] 定义里程碑、依赖关系与风险缓解
- **Status:** complete

### Phase 3: 数据工程落地（TLS定向）
- [ ] 实现 TLS 过滤、会话切分与分组切分
- [ ] 产出 Full-TLS / Leakage-reduced 双轨数据集
- [ ] 验证 `dataset/<name>/image_data` 与 `pcap_data` 成对生成
- **Status:** pending

### Phase 4: 特征与样本表示实现
- [ ] 实现 TLS-RGB（R/G/B 固定语义编码）
- [ ] 实现 TLS-Field-BERT token 构建与词表
- [ ] 统一样本索引与标签映射
- **Status:** pending

### Phase 5: 模型训练与集成
- [ ] 实现三阶段训练（分支热身→融合→集成）
- [ ] 实现 cross-attn + gating + 辅助监督
- [ ] 实现增强 stacking（可选 MoE）与推理导出
- **Status:** pending

### Phase 6: 评估、消融与报告
- [ ] 输出主指标、每类指标、混淆矩阵、学习曲线
- [ ] 完成 6 类消融矩阵（分支/融合/RGB/集成等）
- [ ] 自动生成 run 报告与结果归档
- **Status:** pending

### Phase 7: 交付与复现实验
- [x] 整理 configs、命令与默认行为说明
- [ ] 完成 smoke 级全链路复现
- [x] 输出最终交付清单与后续迭代建议
- **Status:** in_progress

### Phase 8: 执行计划（executing-plans 批次化落地）
- [x] 读取 `doc/plans/2026-02-21-malicious-tls-family-classification.md`
- [x] 完成 Step 1 审查并识别阻塞项
- [ ] 按 Task 1-3 完成首批实现与验证
- [ ] 输出批次报告并等待 review 反馈
- **Status:** blocked（等待 worktree 目录与基线确认）

## Key Questions
1. 当前 `SourceData` 中最终纳入的恶意家族名单与最小样本阈值是否固定（建议先锁定家族名单后再生成最终 split）。
2. 训练资源上限（单卡显存、可用时长）是多少（将直接影响 BERT 序列长度与集成复杂度）。
3. 交付优先级是“最高精度”还是“可部署轻量模型”（影响是否强制做蒸馏阶段）。

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 以 `SourceData` 作为唯一数据来源进行切分与评估 | 与初步方案一致，避免引入外部数据导致不可复现 |
| 计划先落地主线（cross-attn fusion + stacking），MoE 与蒸馏作为增强阶段 | 控制交付风险，先确保主成果可达成 |
| 将任务拆成 7 个阶段并定义每阶段验收产物 | 保证推进可跟踪，便于并行与回归 |
| 详细实施计划文档落位到 `doc/plans/2026-02-21-malicious-tls-family-classification.md` | 与仓库 `doc/` 结构一致，便于团队查阅 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 无 | 1 | - |

## Notes
- 每完成一个阶段立即更新 `task_plan.md`、`findings.md`、`progress.md`。
- 涉及数据切分、标签映射、泄漏控制的结论必须写入 `findings.md`。
