# MobileViT + 改动版 ET-BERT MVTBA 重构设计

## 1. 背景与目标

当前项目已经具备较完整的 `session_full` 论文口径实验编排能力，但核心模型与文本特征表达仍停留在轻量自定义方案：

1. 图像侧使用浅层 CNN。
2. 文本侧使用 TLS record 摘要的哈希 token。
3. 融合侧虽然具备双分支 + gate 的基本框架，但与更强 backbone 的结合不足。

本轮重构目标是：

1. 在实验组织上尽量贴近 MVTBA 论文的两阶段协议。
2. 在模型结构上替换为用户指定的 `MobileViT + 改动版 ET-BERT`。
3. 整体推翻旧的 hashed token 文本链路，使 ET-BERT 预训练真正可用。
4. 保留 `session` 级样本定义与整体工程化训练/评估能力。

## 2. 已确认设计决策

1. 视觉 backbone 采用用户已验证过的 `MobileViT` 路线。
2. 文本 backbone 不使用 CharBERT，改为 ET-BERT。
3. 允许整体推翻旧文本预处理，不强求兼容旧 `seq_shard` 格式。
4. 实验流程尽量靠近 MVTBA：
   - 阶段1：二分类
   - 阶段2：多分类
5. 具体模型结构不是照搬论文原模，而是采用 `MobileViT + 改动版 ET-BERT + 融合头`。

## 3. 总体系统架构

新系统分为 4 层：

1. 预处理层：
   `PCAP -> Session -> RGB 图像特征 + ET-BERT 文本特征 + Manifest`
2. 数据加载层：
   读取 session 级产物，组装多模态 batch。
3. 模型层：
   `MobileViT backbone + 改动版 ET-BERT + 融合分类头`
4. 协议执行层：
   阶段1二分类、阶段2多分类、评估与报告。

## 4. 数据预处理设计

### 4.1 样本单位

样本单位继续定义为 session，不退回 capture 级别。

每个 session 至少保存：

1. `session_id`
2. `dataset`
3. `family`
4. `capture_id`
5. `split`
6. `is_tls_ssl`
7. `tls_ssl_reason`

此处延续当前 `session_full` 的合理抽象，不再推翻 session 粒度。

### 4.2 图像特征

图像侧保留当前会话图像生成思路的主体框架：

1. 输入仍为 `3 x 28 x 28`
2. 每个 session 生成一张 3 通道图
3. 默认继续输出抽检 PNG 以便人工检查

设计原因：

1. 当前项目已有稳定图像构造逻辑。
2. 用户已有单分支 `MobileViT` 在类似输入上验证有效。
3. 首轮重构优先解决 backbone 与文本预训练对齐问题，而不是同时更换图像编码范式。

### 4.3 文本特征：ET-BERT 对齐链路

旧方案的问题在于：

1. 当前 token 由 `VER_* / RT_* / RL_*` 经哈希映射到固定词表。
2. 这与 ET-BERT 的词表和预训练输入语义不一致。
3. 因而无法有效继承 ET-BERT 预训练 embedding 与上下文表示。

新方案：

1. 废弃旧 hashed token 文本链路作为主输入。
2. 新增 ET-BERT 对齐编码器，从 session payload / datagram 序列生成文本 token 序列。
3. 直接输出：
   - `input_ids`
   - `attention_mask`
   - `token_type_ids`
4. 词表对齐 ET-BERT 使用的 `encryptd_vocab.txt`。
5. 序列长度第一版建议 `128` 或 `256`，保持训练成本可控。

### 4.4 预处理产物目录

新的主产物目录建议为：

1. `outputs/processed/<dataset>/session_full/manifest/`
2. `outputs/processed/<dataset>/session_full/rgb/`
3. `outputs/processed/<dataset>/session_full/etbert/`

其中：

1. `rgb` 保存图像张量 shard。
2. `etbert` 保存文本张量 shard：
   - `session_id`
   - `label`
   - `input_ids`
   - `attention_mask`
   - `token_type_ids`

旧 `seq/` 可不再作为主链路输出。

## 5. 数据加载层设计

`pipeline_data` 重构为新多模态输入装载器：

1. 从 `rgb/` 读取图像张量。
2. 从 `etbert/` 读取文本张量。
3. 通过 `session_id` 对齐图像与文本样本。
4. 结合 manifest 生成：
   - 标签
   - split
   - dataset
   - family

输出字段建议为：

1. `rgb`
2. `input_ids`
3. `attention_mask`
4. `token_type_ids`
5. `y`
6. `session_id`
7. `split`
8. `dataset`

## 6. 模型结构设计

### 6.1 图像分支：MobileViT

视觉侧采用用户现有 `MobileViT` 路线：

1. backbone 使用 Hugging Face / transformers 风格 `MobileViT`
2. 加载预训练权重
3. 不直接使用最终分类头
4. 仅提取图像 embedding 作为融合输入

图像分支结构：

`rgb -> MobileViT backbone -> img_feature -> head_img`

### 6.2 文本分支：改动版 ET-BERT

文本分支目标不是“结构参考”，而是尽可能保留 ET-BERT 预训练收益。

设计原则：

1. 保持 ET-BERT 词表与输入语义对齐。
2. 允许对 encoder 深度做轻量化裁剪。
3. 尽量避免修改 hidden size，以保住预训练权重可加载性。

推荐实现：

1. 使用 ET-BERT 预训练参数作为初始化来源。
2. 采用“截断版 encoder”：
   - 第一版优先考虑保留前 4 层或前 6 层。
3. 保持 embedding 维度与主干 hidden size 不变。
4. 最终通过 pooling 得到 `tls_feature`。

文本分支结构：

`input_ids/token_type_ids/attention_mask -> truncated ET-BERT -> tls_feature -> head_tls`

### 6.3 融合层

融合层不使用最简单的 late fusion，而保留当前项目中更有价值的多头输出思想。

建议结构：

1. `img_feature`
2. `tls_feature`
3. 可选轻量 cross-attention / 双塔交互
4. gate 融合
5. fused 分类头

输出保持为：

1. `logits_fuse`
2. `logits_img`
3. `logits_tls`
4. `gate`

这样既方便训练，也便于继续支持 stacking / moe。

## 7. 训练与阶段语义

### 7.1 warmup 阶段

`warmup` 保留，但重新解释为：

1. 稳定图像支路与文本支路各自分类能力
2. 降低一开始融合训练带来的震荡

损失延续当前思路：

1. `img` 分支损失
2. `tls` 分支损失
3. warmup 预测时可取双分支平均

### 7.2 fusion 阶段

`fusion` 为主训练阶段：

1. 启用融合头
2. 继续保留分支辅助损失
3. 主指标以 `logits_fuse` 为准

### 7.3 stacking / moe

第一阶段保留接口，不作为最先落地重点。

原因：

1. 主模型与新预处理先稳定更重要。
2. stacking / moe 依赖主模型输出质量。
3. 只要输出契约仍保留，这两条扩展实验链后续可恢复。

## 8. 实验协议设计

### 8.1 阶段1：混合二分类

按当前 MVTBA 风格组织：

1. `ISCX -> normal`
2. `MFCP + MTA + USTC-TFC2016 -> malicious`

目标：

1. 优先验证新 backbone 在粗粒度判别上的稳定性
2. 建立与旧方案可对照的第一层结果

### 8.2 阶段2：多分类

三数据集分别独立训练：

1. `MTA`：7 类
2. `MFCP`：6 类
3. `USTC-TFC2016`：10 类

每个数据集：

1. 独立训练
2. 独立评估
3. 独立报告

### 8.3 指标输出

统一输出：

1. Accuracy
2. Precision
3. Recall
4. Macro-F1
5. confusion matrix
6. protocol summary

## 9. 代码重构范围

### 9.1 重点重写

1. `src/data/feature_encoder.py`
2. `src/pipeline_data.py`
3. `src/models/fusion_model.py`
4. `src/train.py`
5. `src/evaluate.py`

### 9.2 条件保留并适配

1. `src/experiments/stage1_binary.py`
2. `src/experiments/stage2_multiclass.py`
3. `src/report.py`
4. `src/stacking.py`
5. `src/moe.py`

### 9.3 可删除或降级的旧逻辑

1. 旧 hashed token 编码主路径
2. 依赖旧 `seq_shard` 的主读取逻辑

## 10. 测试策略

### 10.1 预处理测试

新增或重写以下验证：

1. ET-BERT 文本特征文件正确产出
2. 词表映射后 token 非空
3. `session_id` 在 rgb / etbert / manifest 间一致
4. 预处理后数据能被新 loader 正确装载

### 10.2 模型测试

1. 新模型 forward shape 正确
2. `gate` 仍在 `[0, 1]`
3. ET-BERT 输入 mask 正常工作
4. MobileViT 图像输入与文本输入可联合前向

### 10.3 协议集成测试

1. 阶段1二分类最小数据跑通
2. 阶段2多分类最小数据跑通
3. train -> evaluate -> report 路径可闭环

## 11. 风险与缓解

### 11.1 ET-BERT 预训练权重接入风险

风险：

1. ET-BERT 不是 Hugging Face 原生模型。
2. 权重加载和模块裁剪可能需要额外适配。

缓解：

1. 第一阶段先保证词表与输入格式对齐。
2. 第二阶段实现截断版 encoder 加载。
3. 必要时增加权重转换脚本。

### 11.2 训练成本上升

风险：

1. `MobileViT + ET-BERT` 显著重于当前模型。

缓解：

1. 首版优先使用截断 ET-BERT。
2. 降低 batch size。
3. 先完成阶段1二分类验证，再做阶段2多分类。

### 11.3 与论文“完全一致”之间的差距

风险：

1. 实验协议可接近 MVTBA，但主模型不是论文原始结构。

缓解：

1. 在文档中明确：这是“沿论文协议组织的重构版实验框架”，不是论文原模型复刻。

## 12. 推荐实施顺序

1. 重写 ET-BERT 对齐文本预处理。
2. 重写 `pipeline_data`。
3. 实现 `MobileViT + 改动版 ET-BERT` 主模型。
4. 接通 `train/evaluate`。
5. 跑通阶段1二分类。
6. 跑通阶段2多分类。
7. 最后恢复并验证 stacking / moe。
