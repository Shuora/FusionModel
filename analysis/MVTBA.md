# MVTBA 论文结构

## 1) 数据预处理（从 pcap → session → 双模态输入）

论文把预处理明确拆成 4 步：**流量切分、清洗、长度统一、格式转换**，最终得到两路输入：

* 路 A：**784 bytes 序列**（给 BiLSTM-Att）
* 路 B：**28×28×1 灰度图**（给 MViT）


### 1.1 Traffic Segmentation：pcap → session（不是 flow）

* 按 **五元组**（src ip, src port, dst ip, dst port, protocol）对每个 pcap 做切分。
* **切分粒度选 session 而不是 flow**（论文认为 session 信息更充分）。
* 工具：**SplitCap**，把每个 pcap 切成多个 session。


> 复现要点（建议你实现时固定住）：
>
> * 输出目录结构建议：`dataset/<class_name>/<session_id>.bin`（只存 payload bytes 或完整包字节序列都行，但要保证后面取“前 784 bytes”的逻辑一致）。
> * session 内包的拼接顺序：按时间戳升序拼接（SplitCap一般会按会话内顺序导出，但你要核对）。

### 1.2 Data Cleaning：去偏置字段 + 去重/去空

* 将可能让模型“投机取巧”的字段随机化，例如 **MAC 地址、IP 地址**。
* **删除重复或空 session**。


> 复现要点：
>
> * “随机化”要可复现：固定随机种子；或者更简单，直接把 MAC/IP 字段置零（但要确保不会破坏你截取的 784 bytes 区域——如果你截的是“整个包字节流”，MAC/IP一般在链路层/网络层头部，确实会落在前 784 bytes 里）。
> * 去重：可用 session 字节串的 hash（如 SHA1）做去重。

### 1.3 Length Unification：统一为 784 bytes（截断 / 0 填充）

* **session 长度固定为 784 bytes**：

  * 超过 784：截断
  * 不足 784：尾部补 0
* 这一路直接作为 BiLSTM-Att 的输入序列。


### 1.4 Format Conversion：784 bytes → 28×28×1 灰度图

* 每个 byte ∈ [0,255]，可直接视作灰度像素值。
* 把 784 reshape 成 **28×28×1** 灰度图，作为 MViT 输入。


> 复现要点：
>
> * 不要做额外归一化/标准化，除非你在实现里也同步做（论文描述是“byte 值对应像素值”，没提额外缩放）。
> * 图像张量 shape：一般用 `1×28×28`（channel-first）。

---

## 2) 模型结构（MVTBA = MViT + BiLSTM-Att + 加权融合 + 分类器）

总体框架：预处理后同时走“空间特征”和“时间特征”两条支路，最后按权重融合。

### 2.1 空间分支：New MViT（输入 1×28×28，输出 128-d）

论文把 MobileViT 适配到小输入（28×28），重设计为 **4 层**结构，并最终输出 **128 维空间特征**：

关键配置点：

* 初始层：3×3 Conv，stride=2（先做下采样/抽特征）
* MobileViT block：

  * 卷积核大小 **n=3**
  * patch 空间尺寸 **h=w=2**（降低 Transformer self-attention 计算量）
* 最终：1×1 Conv 调整通道到 128，再 global pooling + flatten 得到 **128-d**。

MobileViT block 的核心是 “Conv(局部) + Unfold-Transformer-Fold(全局) + concat shortcut + Conv 融合”，并明确说明 patch=2×2 时复杂度约为原来的 1/4：

### 2.2 时间分支：BiLSTM-Att（输入 784 bytes，输出 128-d）

BiLSTM-Att 的输入就是 **每个 session 的前 784 bytes**：

结构与超参：

* **Embedding**：每个 byte 做 embedding 到 **64 维**，序列变成 `784×64` 
* **BiLSTM**：hidden size **64**（双向后合起来 128）
* **Attention**：对时序位置加权，得到最终 **128-d 时间特征** 

### 2.3 特征融合：加权线性融合（w 控制空间占比）

融合公式：
**F = w·f_s + (1-w)·f_t**，其中 f_s 空间特征，f_t 时间特征。

论文在 MTA 上试了 w∈{0,0.2,0.4,0.6,0.8,1}，并选择 **w=0.6** 最优：

> 复现建议：
>
> * 训练时固定 w=0.6 先对齐论文主结果；如果你换数据集（比如 CICAndMal2017），再重新 sweep w。

### 2.4 分类器：FC + Softmax

* 分类器：全连接层把 fused feature 映射到 `num_classes`，再 softmax 得到类别概率。
  实现上直接用 `CrossEntropyLoss` 即可（内部等价于 log-softmax + NLL）。

---

## 3) 训练配置（论文给出的“可复现训练超参”）

训练实现环境与关键超参在 4.3 里写得很直接：

* 框架：PyTorch
* **epochs=30**
* **batch size=64**
* 优化器：**Adam**
* **learning rate=0.001**


---

## 4) 任务设置与“训练的是几分类”

论文设计了 3 组实验（你复现时要先确定自己要跑哪一种）：

### Exp I：异常识别（normal vs abnormal）

* 把 ISCX VPN-nonVPN（正常加密流量）与 MTA、MFCP（恶意加密流量）混合，构造“9 normal + 9 abnormal”的异常识别数据集。

> 这里通常是 **二分类（正常/异常）** 的设定；“9+9”描述的是组成来源的类别数，不一定意味着 18 分类。复现时你要按论文实验意图做二分类更合理。

### Exp II：细粒度家族识别（多分类）

* **MTA：7 类**（7 个 malware family）
* **MFCP：6 类**（6 个 malware family）

### Exp III：小样本场景（多分类）

* USTC 恶意子集：保持测试集不变，训练集分别限制为 4000/3000/2000 样本做对比。

---

## 5) 最小可复现实现清单（你照这个做基本就能对齐论文）

### 5.1 数据落盘格式

对每个 session 保存：

* `bytes`: `uint8[784]`（截断/补零后的结果）
* `image`: `uint8[1,28,28]`（reshape）
* `label`: int

建议你在 Dataset 里在线生成 `image = bytes.reshape(1,28,28)`，这样不占双份存储。

### 5.2 DataLoader 输出

每个 batch 返回：

* `x_bytes`: `[B, 784]`（long/int64，用于 embedding 索引）
* `x_img`: `[B, 1, 28, 28]`（float 或 uint8 转 float）
* `y`: `[B]`

### 5.3 训练循环（关键点）

* `fs = MViT(x_img)` → `[B,128]`
* `ft = BiLSTM_Att(x_bytes)` → `[B,128]`
* `F = w*fs + (1-w)*ft`（w=0.6）
* `logits = FC(F)` → `[B,num_classes]`
* `loss = CrossEntropyLoss(logits, y)`
* Adam(lr=1e-3), epoch=30, bs=64 

