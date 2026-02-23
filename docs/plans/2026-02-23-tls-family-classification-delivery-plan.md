# 恶意 TLS 家族分类端到端落地计划（7天，8GB 显存，三数据集并行）

## 摘要
目标是在不解密 TLS 的前提下，完成“恶意家族多分类”可复现实验闭环，主榜单采用严格防泄漏口径（按 capture 分组 + leakage-reduced），首轮硬验收线 `Top-1 >= 95%`，冲刺 `98%`。  
技术路线锁定：`RGB(TLS语义化)` + `TLS-Field-BERT(BERT-like)` + `双向Cross-Attn+Gate`，并保留 `BiLSTM-Att` 对照与 `MoE(P1增强)`。

## 范围与边界
1. 数据范围：`SourceData` 下 3 个原始数据集只读，不移动不改写。
2. 任务范围：仅恶意家族多分类，不做 benign/malicious 二分类。
3. 信息约束：仅用 TLS 可见侧信道（握手字段、record 头、长度/方向/时间），不解密 payload。
4. 评估策略：双榜单输出，但主榜单固定 `strict leakage-reduced`，`full-TLS` 仅对照。
5. 工期与资源：7 天；硬件 `i7-13700 + 10GB RAM + RTX 4060 Laptop 8GB`。

## 规划文件工作流（先执行）
1. 阅读技能文档：`~/.codex/skills/planning-with-files/SKILL.md`（已完成）。
2. 创建并持续维护：
   - `doc/planning-with-files/task_plan.md`
   - `doc/planning-with-files/findings.md`
   - `doc/planning-with-files/progress.md`
3. 规则：每完成一个阶段更新状态；每次关键发现写入 findings；每次实验写入 progress（含命令、配置、指标、异常）。

## 代码与目录落地设计
1. 原始数据：`SourceData/<dataset_name>/*.pcap`（只读）。
2. 处理索引与 schema：`src/data/processed/`（轻量元数据、split 清单、字段映射）。
3. 大体积中间产物：`outputs/processed/`（session parquet/csv、缓存张量）。
4. 训练产物：`runs/{run_id}/`，包含：
   - `config.yaml`
   - `train.log`
   - `checkpoints/`
   - `metrics.csv`
   - `figures/`
   - `report.md`
5. 入口脚本：
   - `src/train.py --stage warmup|fusion|stacking|moe`
   - `src/evaluate.py --split test --policy strict|full`
   - `src/report.py --run_id ...`

## 数据预处理产物（回答问题 1）
1. 图像主产物（训练输入，不默认落 PNG）：
   - `outputs/processed/<dataset>/<policy>/rgb/rgb_shard_00000.npz`
   - 数组字段：`session_id[str]`、`label[int]`、`rgb[uint8, N, 3, 28, 28]`
2. 时序主产物（训练输入）：
   - `outputs/processed/<dataset>/<policy>/seq/seq_shard_00000.npz`
   - 数组字段：`session_id[str]`、`token_ids[int32, N, L]`、`attention_mask[uint8, N, L]`、`segment_ids[uint8, N, L]`
3. 元数据索引：
   - `outputs/processed/<dataset>/<policy>/manifest/session_manifest.parquet`
   - 字段：`session_id,dataset,family,capture_id,split,policy,flow_stats`
4. 可视化抽检图（仅调试采样）：
   - `outputs/processed/<dataset>/<policy>/debug/preview_png/<family>/<session_id>.png`
   - 默认每类抽样 20 张，不做全量落盘，避免 I/O 膨胀。

## 非 TLS 过滤程序设计（回答问题 2）
1. 过滤入口：
   - `src/data/tls_filter.py`
   - `src/data/build_dataset.py --tls-filter strict|relaxed`
2. 会话聚合：
   - 使用 5-tuple 双向合并形成 session，按时间窗口归并。
3. TLS 判定规则（strict）：
   - `TCP payload >= 5` 且可解析 TLS record header。
   - `content_type in {20,21,22,23}`。
   - `version in {0x0300,0x0301,0x0302,0x0303,0x0304}`。
   - `record_length in [1, 18432]`。
   - 在首段负载中至少命中 2 条合法 record，并满足以下之一：
     - 解析到 `ClientHello/ServerHello/Certificate`；
     - 或中途流（无 hello）但合法 appdata record 连续命中 >= 6。
4. 明确剔除：
   - 非 TCP 会话。
   - 明显明文协议特征（HTTP 方法、DNS 文本头等）且 TLS header 不成立。
   - 仅端口命中但 record 校验失败的流量。
5. 过滤输出：
   - 保留会话：`outputs/processed/<dataset>/<policy>/manifest/tls_sessions.parquet`
   - 丢弃记录：`outputs/processed/<dataset>/<policy>/manifest/non_tls_dropped.parquet`
   - 丢弃原因统计：`bad_header, invalid_version, no_handshake_evidence, non_tcp, cleartext_signature`

## 公共接口与类型（决策完成）
1. 数据样本统一结构 `Sample`：
   - `rgb_tensor`: `float32 [3, 28, 28]`
   - `tls_tokens`: `int64 [L<=256]`
   - `tls_mask`: `int64 [L]`
   - `label`: `int64`
   - `meta`: `{dataset,capture_id,family,policy}`
2. 模型前向输出 `ModelOutput`：
   - `logits_fuse`, `logits_img`, `logits_tls`
   - `gate_value`
   - `embeddings`（用于 stacking/moe）
3. 评估输出 `EvalResult`：
   - `top1`, `macro_f1`, `macro_recall`
   - `per_class_metrics`
   - `confusion_matrix_raw`, `confusion_matrix_norm`
4. OOF 元特征 `MetaFeatures`：
   - `logits_img/tls/fuse`
   - `entropy_*`, `margin_*`
   - `gate_value`, `norm_img/tls`

## 模型方案（锁定）
1. 图像分支：`MobileViT-XXS`，输入 `28x28x3`。
2. 时序分支：`TLS-Field-BERT`（4 层、hidden=256、heads=8、max_len=256）。
3. 融合：双向 cross-attn + sigmoid gate。
4. 监督：`CE(fuse) + 0.3*CE(img) + 0.3*CE(tls)`（可网格微调）。
5. 对照与增强：
   - 必做对照：`BiLSTM-Att`
   - P1 增强：`MoE router`（专家：img/tls/fuse）

## 训练日志与进度条规范（回答问题 3）
### 控制台日志（中文 + icon，允许保留英文术语）
1. 日志格式（控制台与文件同构）：
   - `{time} | {level_icon}{level} | {module_icon} {module} | {event} | key=value ...`
2. 等级与图标：
   - 成功：`✅`
   - 警告：`⚠️`
   - 错误：`❌`
3. 模块图标约定：
   - `🧱 Data`、`🧠 Model`、`🧪 Eval`、`💾 Save`、`⏱️ Time`、`📈 Metric`
4. 启动必打日志：
   - git commit（可用则打印 short hash）
   - config 摘要（关键超参与路径）
   - 数据集统计（样本数、家族数、类别分布）
5. epoch 必打日志：
   - `train/val loss, acc, macroF1, lr, epoch_time`
6. 保存必打日志：
   - 保存路径、best 指标、checkpoint SHA256 前 8 位
7. 异常处理必打日志：
   - `NaN`, `gradient explosion`, `empty dataset`, `file missing`
   - 严重异常默认终止；可恢复异常执行降级策略并记录。

### 进度条实时展示（epoch/batch 级）
1. 使用 `tqdm`（或 `rich progress`）实现多进度条。
2. 训练阶段：
   - `position=0` 显示 train loop，`position=1` 显示 val loop。
   - 实时字段：`epoch/batch, ETA, loss, acc, macroF1(optional), lr`
3. 预处理阶段：
   - 显示 pcap 文件级与 session 级双层进度。
   - 实时输出 `accepted_tls`, `dropped_non_tls`, `drop_ratio`。
4. 日志落盘：
   - 进度条简化文本同步写入 `train.log`，避免全是控制字符。

## 七天执行里程碑（可直接分工）
1. Day 1：三数据集盘点、family 映射表、capture 分组切分器（strict/full 双策略）与泄漏检查脚本。
2. Day 2：TLS/非TLS 过滤流水线 + R/G/B 与 tokenizer 生成；产出样本统计与可视化抽检。
3. Day 3：warmup 训练（img-only、tls-only、BiLSTM-Att）并产出基线指标。
4. Day 4：fusion 主模型训练稳定化（AMP、grad accumulation、early stopping）。
5. Day 5：OOF stacking（LightGBM）训练与测试；对照主模型收益分析。
6. Day 6：MoE(P1) + 论文必做消融批量实验（短程统一 budget）。
7. Day 7：统一复现实验、报告生成、交付封版（命令、配置、权重、图表、结论）。

## 消融实验清单（论文必做，回答问题 4）
### 时序分支消融
1. `BiLSTM-Att`（基线）
2. `TLS-Field-BERT`（主线）
3. `Byte-BERT`（增强）

### 融合机制消融
1. 线性 `w` 融合
2. learnable weight（旧）
3. cross-attn + gating（新）
4. 有/无辅助损失（验证防塌缩）

### RGB 通道贡献
1. `R-only`
2. `G-only`
3. `B-only`
4. `RGB`
5. `G 去 SNI`（泄漏控制）

### 集成复杂度
1. 简单拼概率 + XGBoost（旧）
2. 增强 stacking（新 A）
3. MoE router（新 B）

### 消融执行规范（保证可比）
1. 固定 split、seed、训练 epoch 上限与 early stop 规则。
2. 统一报告 `Top-1, MacroF1, MacroRecall, 参数量, 单 epoch 时间`。
3. 每项消融输出到 `runs/ablation/<group>/<run_id>/`，自动汇总总表。

## 测试与验收场景
1. 数据正确性：
   - TLS record header 合法性校验通过。
   - split 无 capture 级泄漏（同 capture 不跨集合）。
2. 训练稳定性：
   - 无 NaN/梯度爆炸。
   - 单卡 8GB 可完成至少 30 epoch（或 early stop）。
3. 指标验收（strict leakage-reduced 主榜）：
   - 硬门槛：`Top-1 >= 95%`
   - 冲刺：`Top-1 ~= 98%`
   - 同时输出 macro 指标和每类召回。
4. 报告完整性：
   - learning curves、双混淆矩阵、per-class F1、错分对、gating/router 分布图齐全。
5. 可复现性：
   - 固定 seed 与 config 下，重复两次波动在预设阈值内（例如 Top-1 差异 <= 0.5%）。

## 风险与兜底
1. 风险：strict 口径下 98% 不稳定。
   - 兜底：先保 `>=95%` 交付线，再通过特征桶化/调参/stacking 冲刺。
2. 风险：三数据集标签体系不一致。
   - 兜底：先做统一 family 映射层，冲突类单独列入 `other_rare`。
3. 风险：8GB 显存不足。
   - 兜底：减小 L/K、启用梯度累积、冻结底层、降低 batch。

## 明确假设与默认值
1. 三个原始数据集会在 `SourceData/` 下可访问且具备合法 pcap。
2. 首轮以 `strict leakage-reduced` 为主榜单，`full-TLS` 仅作对照。
3. 时序主线采用 `TLS-Field-BERT`，`CharBERT` 不作为首轮主干，仅可做扩展消融。
4. MoE 定位为 `P1 增强`，不阻塞首轮硬交付。
5. 最终交付以“可运行命令 + 可复现实验报告 + 权重与图表”定义完成。
