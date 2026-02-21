# FusionModel: MobileViT + CharBERT + Attention（USTC / CIC5 / MFCP）

本项目在 `FusionModel` 仓库内实现了完整的“预处理 + 训练 + 评估”链路，目标是：

- 保持 USTC 训练效果
- 提升 CIC 五大类（`Adware, Benign, Ransomware, SMSMalware, Scareware`）效果
- 新增 MFCP 家族分类实验链路（预处理 + 训练 + 验证）
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
│  └─ MFCP/
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

### 3.5 MFCP（payload）

```powershell
# 首次构建（会写入预处理索引）
python src/pipeline/dataset_builder.py --profile mfcp_payload

# 强制重建索引（源数据变更后可用）
python src/pipeline/dataset_builder.py --profile mfcp_payload --rebuild_index_cache
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

说明：4.1-4.4 这些单跑命令现在也会默认自动归档本次产物到 `outputs/archive/<timestamp>_<dataset>_<method>/`。  
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

运行 MFCP：

```powershell
# Attention
python src/fusion/run_attention_suite.py --profile mfcp_baseline --mode attention

# Attention + Stacking
python src/fusion/run_attention_suite.py --profile mfcp_baseline --mode attention_stacking
```

说明：以上两个 profile 已内置 `batch_size=64, num_workers=4, prefetch_factor=2`，适配 4060 Laptop 8GB + 16GB 内存的稳定优先场景。
说明：`run_attention_suite.py` 现在默认会在全部流程结束后自动归档本次输出到 `outputs/archive/<timestamp>_<dataset>_<method>/`。

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

## 7. USTC strict split check (commands)

下面这组命令用于验证 USTC 高分是否受切分口径影响。  
不会改动原数据集 `dataset/USTC-TFC2016`，只会在 `dataset/` 下新建目录。

```powershell
Set-Location C:\Repositories\Traffic\FusionModel
```

### 7.1 Build strict datasets (new folders only)

```powershell
# A) no fallback: single-pcap class -> Train only
python src/pipeline/dataset_builder.py --profile ustc_strict_nofallback --dataset_root dataset

# B) time split fallback: same pcap, first 80% sessions -> Train, last 20% -> Test
python src/pipeline/dataset_builder.py --profile ustc_strict_time80 --dataset_root dataset
```

### 7.2 Audit split leakage

```powershell
python src/pipeline/split_audit.py --dataset_dir dataset/USTC-TFC2016 --output outputs/split_audit_USTC-TFC2016.json
python src/pipeline/split_audit.py --dataset_dir dataset/USTC-TFC2016-strict-nofallback --output outputs/split_audit_USTC-TFC2016-strict-nofallback.json
python src/pipeline/split_audit.py --dataset_dir dataset/USTC-TFC2016-strict-time80 --output outputs/split_audit_USTC-TFC2016-strict-time80.json
```

### 7.3 Fair model comparison (same model/hparams, different split)

```powershell
# baseline split
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode attention --archive_tag ustc_baseline_splitcheck

# strict time split
python src/fusion/run_attention_suite.py --profile ustc_strict_time80_eval --mode attention --archive_tag ustc_strict_time80_splitcheck
```

### 7.4 Optional: stacking comparison

```powershell
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode attention_stacking --archive_tag ustc_baseline_splitcheck_stack
python src/fusion/run_attention_suite.py --profile ustc_strict_time80_eval --mode attention_stacking --archive_tag ustc_strict_time80_splitcheck_stack
```

### 7.5 Note

- `ustc_strict_nofallback` 默认是 `train_only`，Test 为空，主要用于反证切分回退依赖，不用于最终 ACC/F1 对比。  
- 主对比口径：`USTC-TFC2016` vs `USTC-TFC2016-strict-time80`。

## 8. CIC4 full_packet l1024 (CIC-only, keep USTC unchanged)

本方案只改 CIC，不改 USTC。

```powershell
Set-Location C:\Repositories\Traffic\FusionModel
```

### 8.1 Build dataset (new folder)

```powershell
python src/pipeline/dataset_builder.py --profile cic4_fullpacket_l1024_hraw --dataset_root dataset
```

输出目录：`dataset/CIC4_fullpacket_l1024_hraw`

### 8.2 Train attention (new train profile)

```powershell
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention --archive_tag cic4_fp_l1024_attn
```

### 8.3 Optional: attention + stacking

```powershell
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention_stacking --archive_tag cic4_fp_l1024_stack
```

### 8.4 Compare with previous CIC baseline

```powershell
# old CIC payload baseline
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode attention --archive_tag cic5_payload_baseline

# new CIC full_packet l1024
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention --archive_tag cic4_fp_l1024_compare
```

## 9. CIC only runbook (copy and run)

```powershell
Set-Location C:\Repositories\Traffic\FusionModel
```

```powershell
# 1) build / resume CIC4 full_packet dataset (safe to rerun; existing files will be skipped)
python src/pipeline/dataset_builder.py --profile cic4_fullpacket_l1024_hraw --dataset_root dataset
```

```powershell
# 2) train attention (primary)
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention --archive_tag cic4_fp_l1024_attn
```

```powershell
# 3) optional: attention + stacking
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention_stacking --archive_tag cic4_fp_l1024_stack
```

```powershell
# 4) optional: compare with old CIC payload baseline
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode attention --archive_tag cic5_payload_baseline
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention --archive_tag cic4_fp_l1024_compare
```

## 10. MFCP 快速命令

```powershell
Set-Location C:\Repositories\Traffic\FusionModel
```

```powershell
# 1) 预处理（输出到 dataset/mfcp）
python src/pipeline/dataset_builder.py --profile mfcp_payload
```

```powershell
# 2) 训练与验证（attention）
python src/fusion/train_fusion_attention.py --dataset_name mfcp --preset none --epochs 24 --batch_size 64 --output_tag_prefix mfcp
```

```powershell
# 3) 训练与验证（attention + stacking）
python src/fusion/train_fusion_attention_stacking.py --dataset_name mfcp --preset none --epochs 24 --batch_size 64 --output_tag_prefix mfcp
```

```powershell
# 4) 连续运行两次，验证索引命中（第二次应出现“融合索引缓存命中”）
python src/fusion/train_fusion_attention.py --dataset_name mfcp --preset none --epochs 1 --batch_size 16 --output_tag_prefix mfcp --no_archive
python src/fusion/train_fusion_attention.py --dataset_name mfcp --preset none --epochs 1 --batch_size 16 --output_tag_prefix mfcp --no_archive
```
