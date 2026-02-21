# 恶意 SSL/TLS 加密流量家族分类：详细 Plan

## 0. 任务定义与边界
- **任务**：仅聚焦 **恶意 SSL/TLS 加密流量**（不做良性/恶性二分类），输出 **恶意家族多分类**。
- **不解密**：不依赖明文 payload；允许使用 TLS 握手阶段可见字段、TLS record 头、包长/方向/时间等侧信道信息。
- **输出**：
  1) 一个端到端主模型（RGB 图像分支 + TLS-BERT 时序分支 + 跨模态注意力融合）
  2) 一个更强的集成（MoE/增强 stacking）
  3) 可部署版本（蒸馏为单模型，可选）

---

## 1. 数据与标注设计（TLS 定向）

### 1.1 数据集选择与切分原则
- 训练/验证/测试：按 **pcap/场景来源分割**，避免同一 capture 的会话泄漏（同源泄漏会让指标虚高）。
- 建议至少两类数据来源：
  - **SourceData（仓库内数据集）**：作为唯一训练、验证与测试数据来源，所有实验基于仓库 `SourceData` 目录下的 TLS 恶意家族数据进行构建与评估。
  - 数据划分原则：基于 `SourceData` 中不同 capture 文件 / 家族 / 场景来源进行分组切分，确保训练集与测试集之间无 capture 级数据泄漏，并支持家族级多分类评估。
- 切分策略：
  - **按 capture 文件分组**做 GroupKFold 或 train/val/test 分离。
  - 每个家族保持最小测试样本数阈值（例如 ≥ 500），不足的家族作为 few-shot 分组单独报告。

### 1.2 TLS 过滤与会话切分
- 入口强约束：只保留 TLS/SSL 会话。
  - 识别条件（不解密）：端口不可信，必须检查 TLS record header：
    - ContentType ∈ {20,21,22,23}
    - Version 合法（0x0301~0x0304 / SSLv3 0x0300 视数据情况）
    - Record length 合理
- 切分粒度：session（5-tuple + 双向聚合），按时间窗口归并。

### 1.3 去偏与清洗
- 随机化/移除：IP、MAC、五元组显式字段；删除空/重复会话。
- 去泄漏检查：
  - **SNI/证书指纹**可能强泄漏（如果某家族只连固定域名/证书），需要两套设置：
    1) **Full-TLS features**（包含 SNI/证书统计）
    2) **Leakage-reduced**（SNI hash / 只保留长度/熵等统计，不保留具体字符串）
  - 两套结果都报告。

---

## 2. 输入表示（保留 RGB + 引入 TLS-BERT Token）

### 2.1 RGB 图像（保留）— TLS-RGB 语义化编码
输出固定尺寸：建议继续 **28×28×3**（若 MobileViT 已适配），否则可升级到 32×32×3。

#### R 通道：TLS Record 形态（替代“原始字节头部 512B”的更 TLS 定向版本）
- 取前 K 条 record（例如 K=64）按顺序编码：
  - record_type（1 byte）
  - record_version（2 bytes，可拆分或映射）
  - record_length（2 bytes，分桶/归一化）
  - direction（c2s/s2c 作为 0/1）
  - Δt（相对时间差，log 分桶）
- 将上述序列展平填充到 784 长度；不足补 0，超出截断。

#### G 通道：Handshake 语义（握手阶段可见字段）
- 基于 ClientHello/ServerHello/Certificate 的可见信息：
  - cipher_suites_count、extensions_count
  - ALPN presence、SNI length、supported_groups_count
  - certificate_chain_length、cert_count、cert_total_len
  - handshake->appdata 时间差（可放 B 或 G，二选一）
- 具体编码：
  - 先写入少量关键统计（固定位置），再写入握手字节片段/字段长度序列，填满 784。

#### B 通道：行为统计（会话级）
- 推荐至少 12–24 个统计特征（再展开/重复填充）：
  - duration、pkt_count、byte_count
  - c2s_bytes/s2c_bytes、c2s_pkt/s2c_pkt
  - pkt_len_mean/std/max/min（分方向）
  - inter-arrival mean/std（分方向）
  - burstiness 指标（如 coefficient of variation）
  - record_len_mean/std（分方向）
- 编码方式：固定位置写入归一化数值（0–255），剩余空间用长度序列/方向序列填充。

> 备注：RGB 的核心是“通道语义固定且可解释”，保证论文叙事。

### 2.2 TLS-BERT Token 序列（老师要求的 BERT 时序分支）
主线方案采用 **TLS-Field-BERT（结构化 token）**：

#### Token Schema（示例）
- [CLS]
- TLS_VERSION=1.2
- CH_CIPHER_CNT=xx
- CH_EXT_CNT=yy
- SNI_LEN_BIN=b
- ALPN=0/1
- CERT_CHAIN_LEN_BIN=b
- CERT_CNT_BIN=b
- REC_LEN_BIN_1=b … REC_LEN_BIN_K=b（前 K 条 record 长度分桶）
- REC_TYPE_1=t … REC_TYPE_K=t
- DIR_1=d … DIR_K=d
- DT_BIN_1=b … DT_BIN_K=b
- [SEP]

#### 关键设计
- 分桶：长度/时间做 log 分桶；桶数 32/64。
- 最大序列长度：建议 256–512（视 K 而定）。
- Embedding：token embedding + position embedding + segment embedding（handshake vs appdata）。

#### 训练方式
- 若有能力：先在无标签 TLS 会话做 MLM 预训练（可选），再 finetune 家族分类。
- 若资源有限：直接 supervised finetune。

---

## 3. 模型架构

### 3.1 图像分支：MobileViT / MViT
- 输入：28×28×3
- 输出：
  - **patch tokens**（用于跨模态注意力）
  - pooled embedding f_img ∈ R^d（用于辅助头）
- 维度建议：d=256 或 512。

### 3.2 时序分支：TLS-BERT（或改造 BERT）
- 输入：TLS token 序列
- 输出：
  - token embeddings（用于 cross-attn）
  - pooled embedding f_tls（[CLS]）

### 3.3 融合：双向 Cross-Attention + 门控（替代权重融合）
目的：解决“权重倾斜导致某分支失效”。

#### 流程
1) Image→TLS cross-attn：Z_img = Attn(Q=img_tokens, K=tls_tokens, V=tls_tokens)
2) TLS→Image cross-attn：Z_tls = Attn(Q=tls_tokens, K=img_tokens, V=img_tokens)
3) Pool：p_img = pool(Z_img), p_tls = pool(Z_tls)
4) 门控：g = sigmoid(MLP([p_img; p_tls]))
5) 融合向量：F = g * p_img + (1-g) * p_tls

#### 防塌缩辅助监督（强制两分支都可用）
- 主头：Head_fuse(F) → y
- 辅助头：Head_img(pool(img_tokens)) → y
- 辅助头：Head_tls(pool(tls_tokens)) → y
- Loss：L = CE(fuse) + α CE(img) + β CE(tls)
  - 建议 α=0.3, β=0.3 起步，网格搜索。

---

## 4. 集成学习升级（老师要求“更复杂”）

### 4.1 方案 A（推荐主线）：增强 Stacking（多视角 meta-features）
**Base learners**：
- 模型1：图像分支单独分类器（MobileViT only）
- 模型2：TLS-BERT 单独分类器
- 模型3：融合主模型（cross-attn fusion）

**Meta-features（每个样本）**：
- logits_img, logits_tls, logits_fuse（拼接）
- entropy_img/entropy_tls/entropy_fuse
- margin(top1-top2)
- gating g（融合门控输出）
- embedding norms（||f_img||, ||f_tls||）

**Meta-learner**：
- LightGBM 或 XGBoost（比“仅拼概率”更复杂且更强）
- 训练方式：用验证集/交叉验证 out-of-fold 生成 meta 特征，避免泄漏。

### 4.2 方案 B（更高级）：Mixture-of-Experts（MoE）决策级路由
- Experts：img-only, tls-only, fusion
- Router：小 MLP，输入可用 [f_img; f_tls; g; entropy]，输出 r_i
- 最终概率：p = Σ r_i * p_i
- 加正则：router entropy reg，避免只选一个 expert。

### 4.3 部署（可选）：蒸馏
- Teacher：MoE/stacking 最强输出
- Student：仅融合主模型或轻量版本
- 目标：推理成本下降，精度接近 teacher。

---

## 5. 训练策略与工程细节

### 5.1 训练阶段（建议三阶段）
1) 预训练/热身：
   - 图像分支、TLS-BERT 分支分别训练到可用（各自单独 head）。
2) 融合训练：
   - 解冻融合模块 + 两分支后几层
   - 启用辅助损失（α、β）
3) 集成训练：
   - 生成 OOF meta-features
   - 训练 meta-learner 或 MoE router

### 5.2 类不平衡
- loss：Class-balanced focal / reweighting
- sampler：WeightedRandomSampler
- 报告：macroF1、per-class recall（少样本家族重点）。

### 5.3 评估指标与报告方式
- 主指标：macroF1、macroRecall、Top-1 Acc
- 次指标：per-family precision/recall、混淆矩阵
- 泛化：跨数据集测试（训练 SourceData → 测试 SourceData 的 held-out captures），并可选加入“按场景/按时间”的外推切分。

### 5.4 训练日志与可视化要求（新增）
> 目标：训练过程可观测、可复现实验、输出可直接用于组会/论文。

#### 5.4.1 控制台日志（中文 + icon，允许保留英文术语）
- 统一采用结构化日志格式，示例：
  - ✅/⚠️/❌ 表示成功/警告/错误
  - 🧱 Data、🧠 Model、🧪 Eval、💾 Save、⏱️ Time、📈 Metric 等模块 icon
- 日志内容要求：
  - 启动时打印：git commit（若可用）、config 摘要、数据集统计（样本数/家族数/类别分布）
  - 每个 epoch：train/val 的 loss、acc、macroF1、lr、耗时
  - 每次保存：保存路径、best 指标、checkpoint hash
  - 异常：NaN/梯度爆炸/数据为空等必须显式报错并终止或降级处理

#### 5.4.2 进度条实时展示（epoch/batch 级）
- 采用 tqdm（或 rich progress）实现：
  - 实时显示：当前 epoch、batch、ETA
  - 实时更新：loss、acc（以及可选 macroF1）、lr
  - 支持多进度条：train 与 val 分别显示

#### 5.4.3 实验追踪与落盘（本地保存）
- 每次实验生成唯一 run_id（时间戳 + 简短 hash），输出目录：
  - runs/{run_id}/
    - config.yaml
    - train.log（完整控制台日志落盘）
    - checkpoints/（last、best、以及可选的 epoch_x）
    - metrics.csv（每个 epoch 的指标表）
    - figures/（所有图像）
    - report.md 或 report.html（最终报告）

#### 5.4.4 训练完成后的详细报告（自动生成）
报告内容（report.md / report.html）：
1) 实验信息：run_id、时间、硬件、随机种子、关键超参
2) 数据统计：每类样本数、train/val/test split 方式、是否 leakage-reduced
3) 最佳 checkpoint：best epoch、best 指标、对应路径
4) 测试集详细指标：
   - accuracy、macroP/macroR/macroF1
   - per-class precision/recall/F1/support
5) 错分分析：top confusion pairs、每类最易混淆的前 N 类
6) 推理成本：params、FLOPs（可选）、单样本 latency（可选）

#### 5.4.5 必须输出并保存的图像（figures/）
- 📈 learning curves：train/val loss、train/val acc、macroF1 vs epoch
- 🔥 confusion matrix（归一化 + 非归一化两张）
- 📊 per-class F1 bar chart（按家族）
- 🧭 reliability diagram / calibration curve（可选，用于概率质量分析）
- 🧩 gating/router 分布图（若使用 cross-attn gating 或 MoE router）

---

## 6. 消融实验清单（写论文必须）

### 6.1 时序分支消融
- BiLSTM-Att（基线） vs TLS-Field-BERT（主线） vs Byte-BERT（增强）

### 6.2 融合机制消融
- 线性 w 融合 vs learnable weight（旧） vs cross-attn + gating（新）
- 有/无辅助损失（验证是否防塌缩）

### 6.3 RGB 通道贡献
- R-only / G-only / B-only / RGB
- G 通道去 SNI（泄漏控制）

### 6.4 集成复杂度
- 简单拼概率 + XGBoost（旧）
- 增强 stacking（新A）
- MoE router（新B）

---

