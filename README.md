# FusionModel（CharBERT + 原生 MobileViT）

本仓库用于构建加密流量分类流程，包含：
- MVTBA 风格预处理（session 切分、清洗、784 字节统一）
- RGB 图像分支（`R` 采用论文映射，`G/B` 采用自定义特征通道）
- CharBERT 时序分支
- 注意力融合
- XGBoost Stacking 集成学习

`old/` 目录仅做归档，当前有效实现全部在 `src/`。

## 仓库结构

- `src/preprocess_splitcap_rgb.py`：预处理主入口
- `src/splitcap_runner.py`：SplitCap 调用封装
- `src/session_postprocess.py`：会话提取与后处理
- `src/rgb_builder.py`：RGB 三通道构建
- `src/fusion_dataset.py`：数据集加载与字节 token 化
- `src/models_charbert_mobilevit.py`：原生 MobileViT + CharBERT + 注意力融合模型
- `src/train_common.py`：通用训练逻辑（日志、指标、曲线、checkpoint）
- `src/train_attention_fusion.py`：基础注意力融合训练
- `src/train_attention_stacking.py`：注意力融合 + XGBoost Stacking

数据源目录（仓库已包含）：
- `SourceData/USTC-TFC2016`
- `SourceData/CICAndMal2017`

仓库内置 SplitCap：
- `tools/SplitCap_2-1/SplitCap.exe`

## 环境要求

建议 Python 3.9+。

安装依赖：

```powershell
pip install -r requirements.txt
```

## 1）数据预处理

以下命令均在仓库根目录（`C:\Repositories\Traffic\FusionModel`）执行。

### 1.1 USTC（按文件名 10 分类）

```powershell
python src/preprocess_splitcap_rgb.py --input_root SourceData/USTC-TFC2016 --output_root dataset/USTC_10 --label_mode file_stem --splitcap_exe tools/SplitCap_2-1/SplitCap.exe --splitcap_mode external --train_ratio 0.8 --temporal_formats bin
```

### 1.2 CIC（4 分类：Adware/Ransomware/Scareware/SMSMalware）

```powershell
python src/preprocess_splitcap_rgb.py --input_root SourceData/CICAndMal2017 --output_root dataset/CIC_4 --label_mode group --splitcap_exe tools/SplitCap_2-1/SplitCap.exe --splitcap_mode external --train_ratio 0.8 --temporal_formats bin
```

### 1.3 CIC（42 家族分类）

```powershell
python src/preprocess_splitcap_rgb.py --input_root SourceData/CICAndMal2017 --output_root dataset/CIC_42 --label_mode family --splitcap_exe tools/SplitCap_2-1/SplitCap.exe --splitcap_mode external --train_ratio 0.8 --temporal_formats bin
```

如需额外导出时序文件（分析/复现实验用）：

```powershell
python src/preprocess_splitcap_rgb.py --input_root SourceData/USTC-TFC2016 --output_root dataset/USTC_10 --label_mode file_stem --splitcap_exe tools/SplitCap_2-1/SplitCap.exe --splitcap_mode external --train_ratio 0.8 --temporal_formats bin,npy,pt
```

## 2）模型训练

### 2.1 基础注意力融合

```powershell
python src/train_attention_fusion.py --dataset_root dataset/USTC_10 --batch_size 32 --epochs 30 --lr 1e-3 --device auto --output_dir outputs/train
```

### 2.2 注意力融合 + XGBoost Stacking（主流程）

先跑 USTC：

```powershell
python src/train_attention_stacking.py --dataset_root dataset/USTC_10 --batch_size 32 --epochs 30 --lr 1e-3 --early_stop_metric val_f1 --device auto --output_dir outputs/train
```

再跑 CIC 4 分类：

```powershell
python src/train_attention_stacking.py --dataset_root dataset/CIC_4 --batch_size 32 --epochs 30 --lr 3e-4 --class_balance weighted_sampler_loss --loss_type focal --early_stop_metric val_f1 --device auto --output_dir outputs/train
```

最后跑 CIC 42 分类：

```powershell
python src/train_attention_stacking.py --dataset_root dataset/CIC_42 --batch_size 16 --epochs 30 --lr 3e-4 --class_balance weighted_sampler_loss --loss_type focal --early_stop_metric val_f1 --device auto --output_dir outputs/train
```

## 3）消融实验（R-only）

```powershell
python src/train_attention_stacking.py --dataset_root dataset/CIC_42 --ablation r_only --output_dir outputs/train
```

将该结果与默认 RGB（`--ablation none`）对比，用于验证 G/B 通道是否有效。

## 输出文件

### 预处理输出
- 数据文件：`<output_root>/image_data/...` 和 `<output_root>/pcap_data/...`
- 可选时序文件：`<output_root>/temporal_data/...`
- 统计与日志：`outputs/preprocess/<run_id>/`
  - `preprocess.log`
  - `preprocess_summary.json`
  - `label_distribution.csv`

### 训练输出
目录：`outputs/train/<run_id>/`
- `logs/train.log`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics/epoch_metrics.csv`
- `metrics/final_metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/stacking_classification_report.txt`（stacking 模式）
- `reports/stacking_confusion_matrix.png`（stacking 模式）
- `figures/loss_curve.png`
- `figures/acc_curve.png`
- `figures/f1_curve.png`
- `stacking/meta_features_train.npy`（stacking 模式）
- `stacking/meta_features_test.npy`（stacking 模式）
- `stacking/meta_model_xgboost.json`（stacking 模式）

## 说明

- 默认 SplitCap 路径：`tools/SplitCap_2-1/SplitCap.exe`
- 训练阶段从 `pcap_data` 读取 `.bin`，内部补 `CLS/SEP` 后送入 CharBERT
- 当前流程以准确率与泛化为优先：按源 pcap 切分、宏平均 F1 早停、CIC-42 类别不平衡处理

## 常见问题（运行不了）

- `ModuleNotFoundError: xxx`
  - 说明依赖未安装完整，先执行：
  - `pip install -r requirements.txt`

- Markdown 点击运行报参数缺失
  - 本文档命令已改为单行，可直接点击运行。
  - 必须整行执行，不能只运行第一行 `python ...`。

- 训练时报数据目录缺失或为空
  - 先确认预处理完整跑完（不要中途中断）。
  - 训练前应存在：
    - `dataset/<name>/image_data/Train`
    - `dataset/<name>/image_data/Test`
    - `dataset/<name>/pcap_data/Train`
    - `dataset/<name>/pcap_data/Test`

- USTC 每类只有 1 个源 pcap
  - 代码已对该场景自动启用“单文件类别的 session 级切分”，避免测试集为空。



 python src/train_attention_stacking.py --dataset_root dataset/USTC_10 --train_profile auto --output_dir outputs/train