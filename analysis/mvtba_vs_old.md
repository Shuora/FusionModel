# MVTBA论文预处理流程 vs old目录流程差异

## 论文（MVTBA）中的预处理要点
根据论文文本提取结果，预处理包含：
1. **Traffic Segmentation**：按五元组切分，并以**session**为粒度（而非flow），使用 SplitCap 切分。  
2. **Data Cleaning**：随机化可能引入偏置的信息（如 MAC/IP），删除重复或空会话。  
3. **Length Unification**：统一为 **784 bytes**，超长截断、短于长度零填充。  
4. **Format Conversion**：将 0~255 字节映射为像素，转换为 **28×28×1 灰度图**；同一 784-byte 序列也可作为 BiLSTM-Att 输入。

## old 目录当前流程要点
1. `split_data.py`：从 pcap 中提取会话并按 `(proto, src_ip, sport, dst_ip, dport)` 聚合 payload，再按 8:2 划分 Train/Test。  
2. `ssl_tls_rgb_image.py`：将每个 `.bin` 生成 **28×28×3 RGB 图**：
   - R：头部字节（512B优先）
   - G：握手区+人工语义特征（cipher suite diversity/SNI entropy/cert anomaly）
   - B：会话行为统计（包长均值、间隔方差、持续长度等）
3. `fusion_common.py`：pcap分支读取原始文件前 `max_len-2` 字节（默认 782）并封装为 `[CLS] + bytes + [SEP]` 到 784 长度，同时图像分支读取 RGB 图进行融合训练。

## 关键差异（结论）
1. **表示形态不同**：论文是统一 28×28×1 灰度表示；`old` 是工程化的 28×28×3 多通道构造（含手工统计/语义特征）。
2. **清洗策略显式程度不同**：论文明确有 MAC/IP 随机化与重复/空会话清理；`old` 仅显式跳过空 payload/异常包，未见同等“随机化去偏置”步骤。
3. **时序分支输入构造不同**：论文描述的是统一后的 session 字节序列直接给 BiLSTM-Att；`old` 在字节序列前后加入特殊 token（CLS/SEP）并使用 PAD=256。
4. **切分粒度相近但实现细节不同**：两者都以会话为核心；论文指明使用 SplitCap，`old` 使用 dpkt 自行重组 session payload。
5. **任务侧重点不同**：论文强调模型侧“时空特征融合”；`old` 在数据阶段已经加入较多人工先验特征，属于“预处理增强 + 融合模型”路线。
