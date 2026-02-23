# FusionModel

基于 TLS 侧信道信息的恶意流量家族分类工程骨架。

本项目围绕“数据处理 -> 表示构建 -> 融合训练 -> 集成增强 -> 评估报告 -> 消融验证”构建可复现实验链路，重点关注以下约束：

- 仅使用 TLS 侧信道特征，不依赖 payload 解密。
- 通过 `capture_id` 分组切分，降低数据泄漏风险。
- 支持 Full-TLS 与 Leakage-reduced 双轨配置。

---

## 1. 项目能力概览

- 数据处理：TLS record header 过滤、会话切分、capture 级分组切分。
- 表示构建：TLS-RGB（`28x28x3`）与 TLS token 序列编码。
- 主模型：Image 分支 + TLS 分支 + 双向 cross-attn + gating 融合。
- 训练流程：三阶段训练脚本，落盘 `config.yaml/train.log/metrics.csv/checkpoints`。
- 集成增强：stacking 元特征与 GBDT 元学习器。
- 评估报告：生成评估 JSON、混淆矩阵、指标曲线与 Markdown 报告。
- 消融实验：输出 `ablation_summary.csv`。
- 可选增强：MoE 路由器与蒸馏损失（smoke 级实现）。

> 注意：当前仓库实现以“最小可运行 / smoke 验证”优先，部分模块使用占位数据与简化逻辑，适合先打通流程后再替换为真实数据解析与训练策略。

---

## 2. 目录结构

```text
configs/
  ablation.yaml
  dataset_tls_full.yaml
  dataset_tls_leakage_reduced.yaml
  train_fusion.yaml
  train_stacking.yaml

src/
  common/
    config.py
    io_utils.py
    logging_utils.py
  pipeline/
    build_dataset.py
    leakage_control.py
    pcap_reader.py
    rgb_encoder.py
    sessionize.py
    split_strategy.py
    tls_filter.py
    token_encoder.py
    token_schema.py
  fusion/
    datasets.py
    distill.py
    evaluate.py
    moe_router.py
    report.py
    run_ablation.py
    stacking.py
    train_stagewise.py
    models/
      fusion_cross_attn.py
      heads.py
      image_branch.py
      tls_bert_branch.py

tests/
  config/
  pipeline/
  fusion/
  integration/

doc/
  plans/
  planning-with-files/
```

---

## 3. 环境准备

### 3.1 激活环境

```bash
conda activate FusionModel
```

### 3.2 依赖建议

项目当前实现与测试依赖以下 Python 包（按实际环境补齐）：

- `numpy`
- `pandas`
- `pyyaml`
- `torch`
- `scikit-learn`
- `matplotlib`
- `pytest`

---

## 4. 数据与输出目录约定

- `SourceData/`：原始 `pcap` 数据（不纳入版本控制）。
- `dataset/`：数据构建输出目录。
- `outputs/`：训练、评估、报告、消融结果输出目录。
- `doc/`：方案文档、执行计划、过程记录。

---

## 5. 运行命令（正式链路）

> 以下命令与当前代码入口一一对应。

### 5.1 数据构建

```bash
python -m src.pipeline.build_dataset --config configs/dataset_tls_full.yaml
python -m src.pipeline.build_dataset --config configs/dataset_tls_leakage_reduced.yaml
```

默认会在 `dataset/<dataset_name>/` 下生成：

- `image_data/*.npy`
- `pcap_data/*.json`

### 5.2 主模型训练

```bash
python -m src.fusion.train_stagewise --config configs/train_fusion.yaml
```

默认 `run_name: fusion_baseline`，产物位于：

- `outputs/runs/fusion_baseline/config.yaml`
- `outputs/runs/fusion_baseline/train.log`
- `outputs/runs/fusion_baseline/metrics.csv`
- `outputs/runs/fusion_baseline/checkpoints/best.pt`

### 5.3 集成训练（stacking）

```bash
python -m src.fusion.stacking --config configs/train_stacking.yaml
```

默认输出：

- `outputs/runs/stacking_baseline/stacking/meta_features.csv`
- `outputs/runs/stacking_baseline/stacking/meta_summary.yaml`

### 5.4 评估与报告

```bash
python -m src.fusion.evaluate --run-dir outputs/runs/fusion_baseline
python -m src.fusion.report --run-dir outputs/runs/fusion_baseline
```

输出：

- `outputs/runs/fusion_baseline/evaluation.json`
- `outputs/runs/fusion_baseline/figures/confusion_matrix_smoke.png`
- `outputs/runs/fusion_baseline/figures/metrics_curve_smoke.png`
- `outputs/runs/fusion_baseline/report.md`

### 5.5 消融实验

```bash
python -m src.fusion.run_ablation --config configs/ablation.yaml
```

默认输出：

- `outputs/runs/ablation_baseline/ablation/ablation_summary.csv`

---

## 6. 可选能力（MoE + Distill）

当前提供模块级实现：

- `src/fusion/moe_router.py`
- `src/fusion/distill.py`

可通过测试进行 smoke 验证：

```bash
python -m pytest tests/fusion/test_moe_distill_smoke.py -v
```

> 说明：真实收益（如 F1 提升、推理耗时下降）需在真实数据与训练配置上进一步评估。

---

## 7. 测试与验证

### 7.1 全量测试

```bash
python -m pytest tests -q
```

### 7.2 推荐最小验证链路

```bash
python -m src.pipeline.build_dataset --config configs/dataset_tls_full.yaml
python -m src.fusion.train_stagewise --config configs/train_fusion.yaml
python -m src.fusion.evaluate --run-dir outputs/runs/fusion_baseline
python -m src.fusion.report --run-dir outputs/runs/fusion_baseline
```

---

## 8. 关键配置说明

- `configs/dataset_tls_full.yaml`：Full-TLS 数据构建配置。
- `configs/dataset_tls_leakage_reduced.yaml`：泄漏控制配置（如 SNI 脱敏、证书指纹移除）。
- `configs/train_fusion.yaml`：三阶段融合训练配置。
- `configs/train_stacking.yaml`：stacking 元学习器配置。
- `configs/ablation.yaml`：消融实验组合配置。

---

## 9. 开发与提交规范

- Python 代码遵循 PEP 8，4 空格缩进。
- 命名规范：
  - 函数/变量/文件：`snake_case`
  - 类：`PascalCase`
  - 常量：`UPPER_SNAKE_CASE`
- 提交信息建议：`feat:`、`fix:`、`docs:`、`chore:`。
- 避免提交原始数据、缓存文件、大体积产物。

---

## 10. Roadmap（建议）

- 将 `build_dataset.py` 从占位样本生成替换为真实 `pcap` 解析与字段抽取。
- 将 `train_stagewise.py` 从 smoke 训练替换为真实 dataloader + 优化器 + checkpoint 策略。
- 将 `evaluate.py` 与 `report.py` 接入真实预测结果，完善 per-class 分析与错误案例聚类。
- 为 MoE 与蒸馏补充真实实验脚本与收益对比报告。
