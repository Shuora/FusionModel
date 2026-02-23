# Session Full 论文口径复现实验设计

## 1. 背景与目标

用户希望基于论文《MVTBA: A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification》重构本项目的数据实验口径，并继续沿用当前项目已有的 `RGB + 时序` 特征与训练框架。

核心目标：

1. 引入新的数据预处理策略 `session_full`，采用“SplitCap 会话切分 + 全量会话保留 + TLS/SSL 标记”。
2. 不再强制 strict 过滤非 TLS/SSL 流量，而是保留所有会话并打标。
3. 采用论文风格两阶段评估：
   - 阶段1：`ISCX + MFCP + MTA + USTC` 混合二分类（normal vs malicious）
   - 阶段2：`MTA`、`MFCP`、`USTC` 三个数据集各自独立多分类
4. 缺失数据集时，阶段1必须报错终止（严格模式）。

## 2. 设计决策（已确认）

1. 新策略命名：`session_full`（不用 `paper_mode`）。
2. 会话处理链路：`PCAP -> SplitCap -> Session PCAP -> 特征提取`。
3. Session PCAP 需要真实落盘。
4. 默认在特征提取完成后自动清理 Session PCAP 临时文件。
5. 保留抽检 RGB 图像输出。
6. 阶段1切分采用论文口径（train/test 口径优先）。
7. 阶段1标签定义：
   - `normal = ISCX`
   - `malicious = MFCP + MTA + USTC`
8. 阶段2是 3 个独立多分类模型：
   - MTA-7类
   - MFCP-6类
   - USTC-10类
9. 阶段1缺任一数据集直接失败退出。

## 3. 系统架构与数据流

### 3.1 预处理新增策略

在现有 `strict/full/relaxed` 之外新增 `session_full`：

1. 输入源：`SourceData/<dataset>/*.pcap`。
2. 先执行 Session 级拆分（SplitCap 同类逻辑）并落盘临时会话文件。
3. 对每个会话生成元信息并提取特征。
4. 特征提取后自动清理临时 Session PCAP。

### 3.2 样本定义

`session_full` 样本单位为 session，不做 strict TLS 过滤删除；每个样本增加字段：

1. `is_tls_ssl: bool`
2. `tls_ssl_reason: str`（命中规则或 non-tls 原因）
3. `dataset`, `family`, `capture_id`, `split`
4. `label_binary`（normal/malicious）
5. `label_multiclass`（家族类别）

### 3.3 产物目录

在现有输出结构下新增 `session_full` 口径目录（示例）：

1. `outputs/processed/<dataset>/session_full/tmp_sessions/`（临时，默认自动清理）
2. `outputs/processed/<dataset>/session_full/manifest/`
3. `outputs/processed/<dataset>/session_full/rgb/rgb_shard_00000.npz`
4. `outputs/processed/<dataset>/session_full/seq/seq_shard_00000.npz`
5. `outputs/processed/<dataset>/session_full/debug/preview_png/...`（保留）

## 4. 实验设计

### 4.1 阶段1：混合二分类

数据源：ISCX + MFCP + MTA + USTC

1. 标签映射：
   - ISCX -> normal
   - MFCP/MTA/USTC -> malicious
2. 切分：
   - 按论文口径构建 train/test（优先使用论文给定数量或可复现实验比例）
3. 数据完整性校验：
   - 缺任一数据集则 `error + exit`，不允许 partial run。

### 4.2 阶段2：三数据集独立多分类

分别独立训练评估：

1. `MTA`: 7-class
2. `MFCP`: 6-class
3. `USTC`: 10-class

每个任务均使用 `session_full` 预处理产物，不跨数据集混训。

### 4.3 指标与报告

统一输出：

1. Accuracy
2. Precision
3. Recall
4. Macro-F1
5. confusion matrix（csv/png）
6. 结果汇总表（阶段1 + 阶段2）

## 5. 训练框架复用策略

不重写主训练流程，复用现有：

1. `src.train`
2. `src.evaluate`
3. `src.report`
4. `src.stacking`
5. `src.moe`

改动聚焦在：

1. 数据预处理策略扩展（`session_full`）
2. 数据装载口径（binary/multiclass 标签处理）
3. 实验编排入口（阶段1/阶段2）

## 6. 错误处理与可观测性

### 6.1 强制失败场景

1. 阶段1任一必需数据集缺失。
2. Session 切分失败或结果为空。
3. 特征输出缺失或 manifest 不一致。

### 6.2 日志增强

沿用结构化日志与 tqdm，补充：

1. session 总数
2. `is_tls_ssl=True/False` 分布
3. 临时 session 文件清理数量
4. 抽检 RGB 图数量（按类统计）

## 7. 测试与验收

### 7.1 单元测试

1. `session_full` 会话切分与索引正确性
2. `is_tls_ssl` 标记逻辑
3. 临时 Session 自动清理
4. 抽检 RGB 保留行为

### 7.2 集成测试

1. 单数据集从 PCAP 到 train/eval/report 跑通
2. 阶段1缺数据集时硬失败
3. 阶段2三套多分类均产出指标

### 7.3 验收结果

1. 能复现实验口径与样本规模趋势（相对 strict 口径显著增加）
2. 生成可复现命令、配置、产物报告

## 8. 实施边界

1. 当前轮次只定义设计，不直接改训练模型结构。
2. ISCX 未下载完成时，阶段1按严格规则报错退出。
3. 不影响现有 strict/full/relaxed 链路，可并行保留作对照。

