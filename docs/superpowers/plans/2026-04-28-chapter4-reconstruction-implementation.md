# 第四章实验重构实施计划 (Chapter 4 Reconstruction Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按照毕业论文重构要求，补充完整的消融实验、SOTA 对比实验、不平衡韧性测试，并自动化生成符合学术审美的实验图表。

**Architecture:** 采用模块化消融配置，在 `src/fusion_common.py` 中注入实验分支；在 `experiments/baselines/` 独立实现基准模型；通过 `tools/measure_efficiency.py` 统一评估。

**Tech Stack:** PyTorch, MobileViT, CharBERT, XGBoost, Matplotlib/Seaborn (学术配色), fvcore (复杂度评估).

---

### Task 1: 基础表征与融合机制消融支持

**Files:**
- Modify: `src/ssl_tls_rgb_image.py` (支持灰度图生成)
- Modify: `src/fusion_common.py` (支持 Concat 融合、灰度图加载、扁平序列加载)

- [ ] **Step 1: 修改 `src/ssl_tls_rgb_image.py` 增加 `--mode` 参数**
    - 支持 `rgb` (默认) 和 `gray` (单通道)。
    - 灰度图实现：将二进制数据重塑为 28x28 矩阵。

- [ ] **Step 2: 修改 `src/fusion_common.py` 中的 `FusionDataset`**
    - 增加 `use_temporal_sidecar` 标志，若为 False 则回退到 `.bin` 扁平字节读取。
    - 增加 `image_mode` 处理，支持加载单通道灰度图并根据模型需求决定是否 repeat 到 3 通道。

- [ ] **Step 3: 修改 `src/fusion_common.py` 中的 `AttentionFusionModel`**
    - 增加 `fusion_mode="concat"` 分支。
    - 实现：`torch.cat([img_feats, pcap_feats], dim=1)` 后接线性层。

- [ ] **Step 4: 提交变更**
    - `git add src/ssl_tls_rgb_image.py src/fusion_common.py`
    - `git commit -m "feat: add support for grayscale images and concat fusion ablation"`

---

### Task 2: SOTA 基准模型实现 (Baselines)

**Files:**
- Create: `experiments/baselines/deeppacket.py` (1D-CNN)
- Create: `experiments/baselines/lstm_baseline.py` (Bi-LSTM)
- Create: `experiments/baselines/vit_baseline.py` (Pure ViT / MobileViT-only)

- [ ] **Step 1: 实现 DeepPacket (1D-CNN)**
    - 参考原论文：2 层卷积 + 2 层全连接。
    - 集成到现有训练 pipeline。

- [ ] **Step 2: 实现 LSTM 基准**
    - 使用 Embedding + 双向 LSTM。

- [ ] **Step 3: 实现 ViT/MobileViT-only 基准**
    - 仅保留图像分支，验证单一模态性能。

- [ ] **Step 4: 运行基准测试并保存结果到 `outputs/baselines/`**

---

### Task 3: 自动化绘图系统与工程评估

**Files:**
- Create: `figures/code/fig4_*.py` (独立脚本生成图 4.3, 4.7, 4.9, 4.10, 4.11)
- Create: `tools/measure_efficiency.py` ( Params/FLOPs/Latency)

- [ ] **Step 1: 实现 `tools/measure_efficiency.py`**
    - 使用 `fvcore.nn.FlopCountAnalysis` 计算所有模型 FLOPs。
    - 在测试集上循环 100 次计算推理延迟。

- [ ] **Step 2: 实现 `figures/code/fig4_*.py` 并解决中文乱码**
    - 拆分原脚本为独立文件，引入 `SimHei` 等配置解决乱码。
    - 自动化读取 `outputs/` 下的 `metrics.json` 生成对比柱状图和折线图。

- [ ] **Step 3: 实现跨模态注意力热力图导出**
    - 修改评估代码，将权重保存为 `.npy` 并通过脚本绘图。

---

### Task 4: 不平衡梯度压力测试执行

**Files:**
- Modify: `src/run_all_modes.py` (支持循环运行不同比例的数据集)

- [ ] **Step 1: 自动化运行 2:1 -> 15:1 压力测试**
    - 记录各比例下的 Macro-F1。

- [ ] **Step 2: 提取少数类召回率增益**
    - 对比 Stacking 开启前后的类级别 Recall。

---

### Task 5: 最终报告整合

- [ ] **Step 1: 将所有图表整理到 `docs/figures/chapter4/`**
- [ ] **Step 2: 更新 `findings.md` 记录最终实验结论**
