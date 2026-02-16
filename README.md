# FusionModel: MobileViT + CharBERT + Attention（USTC / CIC5）

本项目在 `FusionModel` 仓库内实现了完整的“预处理 + 训练 + 评估”链路，目标是：

- 保持 USTC 训练效果
- 提升 CIC 五大类（`Adware, Benign, Ransomware, SMSMalware, Scareware`）效果
- 采用 `MobileViT + CharBERT + Attention` 及 `Attention + Stacking`

说明：

- 不改动原仓库 `Data-Processing` 与 `CharBERT-MobileViT`
- 预处理坚持 RGB（不使用灰度）
- USTC/CIC 分开训练与验证，不做交替喂数

## 1. 目录结构

```text
FusionModel/
├─ SourceData/
│  ├─ USTC-TFC2016/
│  └─ CICAndMal2017/
├─ configs/
│  ├─ dataset_profiles.yaml
│  └─ train_profiles.yaml
├─ src/
│  ├─ pipeline/
│  │  ├─ dataset_builder.py
│  │  ├─ pcap_session.py
│  │  └─ feature_rgb.py
│  └─ fusion/
│     ├─ fusion_common.py
│     ├─ train_fusion_attention.py
│     ├─ train_fusion_attention_stacking.py
│     └─ run_attention_suite.py
├─ dataset/
└─ outputs/
```

## 2. 环境准备（PowerShell）

```powershell
Set-Location C:\Repositories\Traffic\FusionModel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 数据预处理（PowerShell）

### 3.1 USTC

```powershell
python src/pipeline/dataset_builder.py --profile ustc
```

说明：USTC 每类通常只有 1 个原始 pcap，本项目会自动启用“session 级回退切分”，保证 `Train/Test` 都有样本。

输出到：

- `dataset/USTC-TFC2016/pcap_data/{Train,Test}/<class>/*.bin`
- `dataset/USTC-TFC2016/image_data/{Train,Test}/<class>/*.png`

### 3.2 CIC5（payload 轨）

```powershell
python src/pipeline/dataset_builder.py --profile cic5_payload
```

### 3.3 CIC5（full_packet 轨）

```powershell
python src/pipeline/dataset_builder.py --profile cic5_fullpacket
```

### 3.4 覆盖重建

默认遇到已有样本会跳过；如果你要重建：

```powershell
python src/pipeline/dataset_builder.py --profile cic5_payload --overwrite
```

## 4. 模型训练（PowerShell）

### 4.1 CIC5：Attention（单跑）

```powershell
python src/fusion/train_fusion_attention.py --dataset_name CIC5_payload --preset cic_balanced --batch_size 64 --num_workers 4 --prefetch_factor 2
```

### 4.2 CIC5：Attention + Stacking（单跑）

```powershell
python src/fusion/train_fusion_attention_stacking.py --dataset_name CIC5_payload --preset cic_balanced --batch_size 64 --num_workers 4 --prefetch_factor 2
```

### 4.3 USTC：Attention（单跑）

```powershell
python src/fusion/train_fusion_attention.py --dataset_name USTC-TFC2016 --preset none --batch_size 64 --num_workers 4 --prefetch_factor 2
```

### 4.4 USTC：Attention + Stacking（单跑）

```powershell
python src/fusion/train_fusion_attention_stacking.py --dataset_name USTC-TFC2016 --preset none --batch_size 64 --num_workers 4 --prefetch_factor 2
```

说明：4.1-4.4 这些单跑命令现在也会默认自动归档本次产物到 `outputs/archive/<tag>_<timestamp>/`。  
可选参数：`--no_archive`（关闭归档）、`--archive_tag <name>`、`--archive_dir <path>`、`--archive_move`（移动而非复制）。

### 4.5 CIC5（full_packet 轨）

```powershell
# Attention（单跑）
python src/fusion/train_fusion_attention.py --dataset_name CIC5_fullpacket --preset cic_balanced --batch_size 64 --num_workers 4 --prefetch_factor 2

# Attention + Stacking（单跑）
python src/fusion/train_fusion_attention_stacking.py --dataset_name CIC5_fullpacket --preset cic_balanced --batch_size 64 --num_workers 4 --prefetch_factor 2

# 批跑（沿用 cic5_balanced 配置，覆盖 dataset_name）
python src/fusion/run_attention_suite.py --profile cic5_balanced --dataset_name CIC5_fullpacket --mode all
```

### 4.6 使用训练配置批跑（推荐）

按 `configs/train_profiles.yaml`：

运行 CIC5：

```powershell
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode all
```

运行 USTC：

```powershell
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode all
```

说明：以上两个 profile 已内置 `batch_size=64, num_workers=4, prefetch_factor=2`，适配 4060 Laptop 8GB + 16GB 内存的稳定优先场景。
说明：`run_attention_suite.py` 现在默认会在全部流程结束后自动归档本次输出到 `outputs/archive/<profile>_<mode>_<timestamp>/`。

常用归档参数：

```powershell
# 指定归档目录名（便于实验管理）
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode all --archive_tag ustc_try_01

# 归档时移动文件（默认是复制）
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode all --archive_move

# 关闭自动归档
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode all --no_archive
```

## 5. 结果产物与日志

训练输出默认写到 `outputs/`：

- `outputs/logs/*.log`：完整日志（中文高密度）
- `outputs/metrics_curve_*.png`：epoch 指标曲线
- `outputs/confusion_matrix_*.png`：混淆矩阵
- `outputs/report_*.md`：acc / macro-f1 / 分类报告
- `outputs/fusion_model_*.pth`：模型权重
- `outputs/meta_model_*.pkl`：stacking 元模型

预处理报告写到 `dataset/<dataset_name>/reports/preprocess_summary_*.json`。

## 6. 常见问题

- 训练时如果出现 `CharBERT 加载失败，已禁止静默降级`，说明当前环境无法导入 `src/CharBERT/src`。先检查该目录是否完整，再重跑。
- CIC 当前没有 `Benign` 文件夹时，`cic5_*` profile 会只处理现有类别；后续加入 `Benign` 后可直接重跑。
- 如果 PowerShell 禁止脚本执行，可先运行：
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
- Windows 路径建议避免中文和空格，减少第三方库路径问题。
