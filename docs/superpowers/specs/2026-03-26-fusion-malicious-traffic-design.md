# 基于特征融合的恶意流量识别系统设计

## 1. 项目目标

构建一个面向毕业设计的恶意流量识别项目，围绕“基于特征融合的恶意流量识别”这一主题，实现可复现实验链路，覆盖：

- PCAP 流量切分与样本构建
- 图像分支与时序分支的双模态深度融合模型
- 二分类与多分类训练任务
- GPU 训练、实时日志、评估报告、曲线图和混淆矩阵输出
- 按日期分层存储的实验结果目录
- 可直接部署环境的 `requirements.txt`

项目目标优先追求较高实验指标，二分类和多分类都尽量冲高。

## 2. 数据与任务定义

### 2.1 数据来源

项目使用仓库中的以下数据：

- `SourceData/ISCX-VPN-NonVPN-2016`
- `SourceData/MTA`
- `SourceData/MFCP`
- `SourceData/USTC-TFC2016`
- `SourceData/CICAndMal2017`

其中，本项目当前正式实验任务按如下方式组织：

- 二分类：
  - benign：`ISCX NonVPN + VPN`
  - malicious：`MTA + MFCP`
- 多分类：
  - `MTA` 单独训练 7 类
  - `MFCP` 单独训练 6 类
  - `USTC` 单独训练 10 类

`CICAndMal2017` 当前不作为主任务数据源，后续可保留为扩展实验候选。

### 2.2 样本粒度

一个 `session` 对应一个样本。模型训练和评估都以 session 为最小输入单位。

### 2.3 划分策略

主实验采用宽松随机划分：

- 对 session 样本做 `stratified random split`
- 不做 `source-grouped split`
- 允许同一来源 PCAP 或同一流量来源生成的相似会话跨 `train/val/test`

该策略有助于提升实验指标，但存在结果虚高和泛化解释性偏弱的风险。系统将按用户要求实现这一策略，并在文档中明确其含义。

默认划分比例为：

- `train = 0.7`
- `val = 0.1`
- `test = 0.2`

## 3. 数据预处理设计

### 3.1 流量切分

流量切分工具固定为仓库内的 `Tools/SplitCap.exe`。

切分流程：

1. 读取原始 PCAP 文件
2. 使用 SplitCap 按五元组切分为 session 级 PCAP
3. 将不同数据集的切分结果分别缓存，保留来源标记

### 3.2 清洗规则

对 session 级样本进行如下清洗：

- 删除空 session
- 删除重复 session
- 随机化 IP / MAC 地址，避免模型直接记忆地址字段
- 保留端口、协议、方向、长度等可反映流量行为的结构信息

### 3.3 定长表示

每个 session 提取统一的前 `784 bytes`：

- 长度不足：右侧补 `0`
- 长度超出：直接截断

统一后的 `784 bytes` 作为两个分支共享的基础输入。

## 4. 双模态输入表示

### 4.1 图像分支输入

图像分支采用真正的多特征 RGB 三通道，而不是单通道复制。

构造方式：

- 将 `784 bytes` 重排为 `28x28`
- 构造三个通道：
  - `R`：原始 byte 强度
  - `G`：相邻 byte 差分强度
  - `B`：局部熵或局部统计强度

最终得到 `28x28x3` 图像，再 resize 到更适合预训练视觉骨干的输入尺寸，默认采用 `112x112`，保留配置项支持切换到 `128x128`。

### 4.2 时序分支输入

时序分支主线采用 `ET-BERT`。

输入构造流程：

1. 从 session 定长字节序列生成 token 序列
2. 构建 ET-BERT 所需输入 `input_ids`、`attention_mask`
3. 对样本做缓存，避免每轮训练重复编码

ET-BERT 的 tokenizer 和权重路径将以可配置方式接入。

## 5. 模型架构设计

### 5.1 总体路线

本项目采用“受限激进版”路线：

- 保持较强的双分支骨干
- 使用深层双向融合，而非简单拼接
- 同时通过混合精度、小 batch、梯度累积等方式适配 `RTX 4060 Laptop 8GB`

### 5.2 图像骨干

图像分支使用预训练 `MobileViT`，优先选择 `MobileViT-S` 级别配置。

该分支负责从 RGB 字节图像中提取：

- 局部纹理模式
- 字节结构变化
- 统计分布差异

### 5.3 时序骨干

时序分支使用 `ET-BERT` 提取 session 的上下文语义表示。

该分支负责建模：

- token 级序列依赖
- 报文结构残留模式
- 加密流量中的上下文行为痕迹

### 5.4 融合模块

融合模块采用深层双向 cross-attention：

- 图像 patch 表示关注时序 token 表示
- 时序 token 表示反向关注图像 patch 表示
- 融合层深度默认 2 到 3 层

融合后追加：

- `gated fusion`
- `residual MLP`

得到最终联合表示后送入分类头。

### 5.5 分类头

分类头采用统一结构，但不同任务单独训练与保存权重：

- 二分类任务单独训练
- `MTA` 多分类单独训练
- `MFCP` 多分类单独训练
- `USTC` 多分类单独训练

## 6. 训练策略设计

### 6.1 训练阶段

采用三阶段训练策略：

#### Stage 1：Branch Warmup

- 先训练各自分支的投影层和分类头
- 让两个分支适应当前流量数据分布

#### Stage 2：Deep Fusion Training

- 打开 cross-attention 融合模块
- `MobileViT` 全量参与训练
- `ET-BERT` 进入深度微调模式
- 默认优先训练后部层并保留配置项支持更激进设置

#### Stage 3：Aggressive Fine-Tuning

- 进入更激进的微调阶段
- 在显存允许的情况下开启 ET-BERT 更大范围微调
- 若出现显存压力，则自动回退到更稳定的部分解冻策略

### 6.2 优化策略

训练默认采用：

- `AdamW`
- `cosine scheduler`
- `warmup`
- `AMP` 混合精度
- `gradient accumulation`
- 可选 `gradient checkpointing`

多分类任务默认支持：

- `class weight`
- `focal loss` 配置开关

### 6.3 GPU 约束

系统默认要求使用 GPU：

- 启动时检测 `cuda`
- 若未检测到可用 GPU，则直接报错退出
- 不允许静默回退到 CPU

## 7. 日志、评估与输出

### 7.1 控制台输出

训练过程中必须提供：

- 实时进度条
- 当前 epoch 平均 `loss`
- 当前 epoch 平均 `acc`
- 验证集主要指标输出

日志风格约束：

- 复杂说明性语句可以使用中文
- 指标名称、epoch、文件名和关键信息字段使用英文

### 7.2 指标输出

二分类输出：

- `acc`
- `precision`
- `recall`
- `f1`
- `roc_auc`

多分类输出：

- `acc`
- `macro_f1`
- `weighted_f1`

同时统一生成：

- `classification_report.txt`
- `confusion_matrix.png`

`classification_report.txt` 的格式应与 `sklearn classification_report` 类似，满足论文表格和截图需求。

### 7.3 曲线与权重

每次训练至少输出：

- `loss_curve.png`
- `acc_curve.png`
- `metrics.csv`
- `best_acc.pt`
- `best_f1.pt`
- `train.log`
- `config.yaml`

### 7.4 目录结构

训练结果按日期分一级目录，再按具体实验任务建二级目录。

示例：

```text
runs/
  2026-03-26/
    binary_iscx_mta_mfcp/
    mta_7cls/
    mfcp_6cls/
    ustc_10cls/
```

每个二级目录中保存对应任务的日志、模型权重、曲线图、混淆矩阵和分类报告。

## 8. 工程结构设计

建议工程结构如下：

```text
src/
  data/
  models/
  trainers/
  utils/
scripts/
configs/
requirements.txt
README.md
```

职责划分：

- `src/data/`：切分、清洗、样本构建、缓存
- `src/models/`：MobileViT、ET-BERT、fusion、heads
- `src/trainers/`：训练与验证循环
- `src/utils/`：日志、绘图、指标、配置、随机种子
- `scripts/`：实验入口脚本
- `configs/`：不同任务配置文件

## 9. 环境要求

目标环境：

- GPU：`RTX 4060 Laptop 8GB`
- 内存：`10GB RAM`
- CPU：`i7-13700`
- `CUDA 12.6`
- `Python 3.9`

项目需提供完整 `requirements.txt`，确保可在该环境中安装并复现实验。

## 10. 非目标与边界

当前版本不将以下内容设为必做项：

- 灰度与 RGB 表示的消融实验
- 严格 source-grouped split 主实验
- 多数据集联合多分类
- CICAndMal2017 主实验接入

这些能力可以保留为后续扩展，但不进入当前第一版实现范围。
