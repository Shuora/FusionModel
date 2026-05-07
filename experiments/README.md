# 实验与基线 (Experiments & Baselines)

本目录用于存放主模型（V4 MobileViT + CharBERT）以外的所有对比工作。

## 目录指南

- **`baselines/`**: 存放通用基线模型（如 ResNet, LSTM, Transformer）。
- **`reproduction/`**: 存放针对特定论文（如 ATVITSC）的完整复现。
- **`comparisons/`**: 存放用于生成对比图表和汇总指标的分析脚本。

## 运行约定

1. **数据**: 请统一读取根目录下的 `ProcessedData/`。
2. **输出**: 实验产物请输出至 `outputs/experiments/<experiment_name>/`。
3. **指标**: 确保每个实验最后能产出符合项目规范的 `metrics.json`。

## 训练命令示例 (Training Commands)

使用 `experiments/baselines/train_baseline.py` 可以统一训练所有对比模型：

### 1. SOTA 对比基准 (SOTA Baselines)

- **DeepPacket (1D-CNN)**:
  ```bash
  python experiments/baselines/train_baseline.py --model_type deeppacket --task_name ustc_multiclass
  ```
- **Bi-LSTM**:
  ```bash
  python experiments/baselines/train_baseline.py --model_type lstm --task_name ustc_multiclass
  ```
- **ViT (Google pre-trained)**:
  ```bash
  python experiments/baselines/train_baseline.py --model_type vit --task_name ustc_multiclass
  ```
- **2D-CNN**:
  ```bash
  python experiments/baselines/train_baseline.py --model_type cnn2d --task_name ustc_multiclass
  ```

### 2. 消融实验单分支 (Ablation Study - Single Branch)

这些模型使用与融合模型分支完全一致的架构和分类头。

- **MobileViT 单分支 (Space Branch)**:
  ```bash
  python experiments/baselines/train_baseline.py --model_type mobilevit_ablation --task_name mfcp_multiclass
  ```
- **CharBERT 单分支 (Time Branch)**:
  ```bash
  python experiments/baselines/train_baseline.py --model_type charbert_ablation --task_name mfcp_multiclass
  ```

### 3. 注意事项
- 默认读取 `ProcessedData/` 下的对应任务目录。
- 可以使用 `--device cuda:0` 指定 GPU。
- 可以通过 `--epochs`, `--batch_size`, `--lr` 覆盖默认训练超参数。
