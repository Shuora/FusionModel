# FusionModel: MobileViT + CharBERT + Attention

本仓库实现了“预处理 + 训练 + 评估”一体化流程，面向加密恶意流量分类任务。

- 融合模型：`MobileViT + CharBERT + Attention`
- 扩展模型：`Attention + Stacking`
- 数据集：`USTC-TFC2016`、`CICAndMal2017`（5 大类）、`MFCP`
- 约束：坚持 RGB 三通道，不使用灰度图；USTC 与 CIC 分开训练/验证

## 1. 目录结构

```text
FusionModel/
├─ SourceData/
│  ├─ USTC-TFC2016/
│  ├─ CICAndMal2017/
│  └─ MFCP/
├─ configs/
│  ├─ dataset_profiles.yaml
│  └─ train_profiles.yaml
├─ src/
│  ├─ pipeline/
│  │  ├─ dataset_builder.py
│  │  ├─ pcap_session.py
│  │  ├─ feature_rgb.py
│  │  └─ split_audit.py
│  └─ fusion/
│     ├─ fusion_common.py
│     ├─ train_fusion_attention.py
│     ├─ train_fusion_attention_stacking.py
│     └─ run_attention_suite.py
├─ dataset/
└─ outputs/
```

## 2. Ubuntu 环境准备

```bash
cd /home/shuora/Repositories/Traffic/FusionModel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 数据源组织

将原始数据放到 `SourceData/` 下：

- USTC：`SourceData/USTC-TFC2016/*.pcap`
- CIC：`SourceData/CICAndMal2017/<Major>/<Subclass>/*.pcap`
- MFCP：`SourceData/MFCP/<Family>/**/*.pcap`

说明：CIC 的主类按 `Adware/Benign/Ransomware/SMSMalware/Scareware` 组织。

## 4. 预处理（生成 RGB 图像 + 字节序列）

通用命令：

```bash
python src/pipeline/dataset_builder.py --profile <profile_name>
```

可用 profile：

- `ustc`
- `ustc_strict_nofallback`
- `ustc_strict_time80`
- `cic5_payload`
- `cic5_fullpacket`
- `cic4_fullpacket_l1024_hraw`
- `mfcp_payload`

常用示例：

```bash
# USTC
python src/pipeline/dataset_builder.py --profile ustc

# CIC5 payload
python src/pipeline/dataset_builder.py --profile cic5_payload

# CIC5 full_packet
python src/pipeline/dataset_builder.py --profile cic5_fullpacket

# MFCP
python src/pipeline/dataset_builder.py --profile mfcp_payload
```

重建相关参数：

```bash
# 覆盖重建 image/bin 文件
python src/pipeline/dataset_builder.py --profile cic5_payload --overwrite

# 强制重建 pcap 索引缓存
python src/pipeline/dataset_builder.py --profile mfcp_payload --rebuild_index_cache
```

预处理输出目录：

- `dataset/<name>/pcap_data/{Train,Test}/<class>/*.bin`
- `dataset/<name>/image_data/{Train,Test}/<class>/*.png`
- `dataset/<name>/reports/preprocess_summary_*.json`

## 5. 训练（推荐用 profile 批跑）

### 5.1 推荐：`run_attention_suite.py`

按 `configs/train_profiles.yaml` 运行：

```bash
# CIC5: attention + stacking
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode all

# USTC: attention + stacking
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode all

# MFCP: 仅 attention
python src/fusion/run_attention_suite.py --profile mfcp_baseline --mode attention

# MFCP: 仅 attention + stacking
python src/fusion/run_attention_suite.py --profile mfcp_baseline --mode attention_stacking
```

可选归档参数：

```bash
# 指定归档目录名
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode all --archive_tag ustc_try_01

# 归档时移动文件（默认复制）
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode all --archive_move

# 关闭自动归档
python src/fusion/run_attention_suite.py --profile cic5_balanced --mode all --no_archive
```

### 5.2 单脚本训练（手动调参）

```bash
# Attention
python src/fusion/train_fusion_attention.py --dataset_name CIC5_payload --preset cic_balanced --batch_size 64 --num_workers 4 --prefetch_factor 2

# Attention + Stacking
python src/fusion/train_fusion_attention_stacking.py --dataset_name CIC5_payload --preset cic_balanced --batch_size 64 --num_workers 4 --prefetch_factor 2
```

常见 `dataset_name`：

- `USTC-TFC2016`
- `CIC5_payload`
- `CIC5_fullpacket`
- `CIC4_fullpacket_l1024_hraw`
- `mfcp`

## 6. 训练产物与日志

默认输出到 `outputs/`：

- `outputs/logs/*.log`：完整训练日志
- `outputs/metrics_curve_*.png`：epoch 指标曲线
- `outputs/confusion_matrix_*.png`：混淆矩阵
- `outputs/report_*.md`：acc / macro-f1 / 分类报告
- `outputs/fusion_model_*.pth`：融合模型权重
- `outputs/meta_model_*.pkl`：stacking 元模型
- `outputs/archive/<timestamp>_<dataset>_<method>/`：自动归档目录

## 7. 常见问题

- `CharBERT 加载失败，已禁止静默降级`：检查 `src/CharBERT/src` 是否完整可导入。
- CIC 缺少 `Benign` 时，`cic5_*` profile 只处理当前存在类别；后续补齐后重跑即可。
- `num_workers` 说明：默认是 `4`。在 Windows + CUDA 环境中可能自动降级为 `0`（用于规避 WinError 1455）；Ubuntu 下不会触发这条降级逻辑。

## 8. 附录 A：USTC strict split 检查

```bash
cd /home/shuora/Repositories/Traffic/FusionModel

# 构建 strict 数据集
python src/pipeline/dataset_builder.py --profile ustc_strict_nofallback --dataset_root dataset
python src/pipeline/dataset_builder.py --profile ustc_strict_time80 --dataset_root dataset

# 检查切分泄漏
python src/pipeline/split_audit.py --dataset_dir dataset/USTC-TFC2016 --output outputs/split_audit_USTC-TFC2016.json
python src/pipeline/split_audit.py --dataset_dir dataset/USTC-TFC2016-strict-nofallback --output outputs/split_audit_USTC-TFC2016-strict-nofallback.json
python src/pipeline/split_audit.py --dataset_dir dataset/USTC-TFC2016-strict-time80 --output outputs/split_audit_USTC-TFC2016-strict-time80.json

# 公平对比（同模型同超参，不同切分）
python src/fusion/run_attention_suite.py --profile ustc_baseline --mode attention --archive_tag ustc_baseline_splitcheck
python src/fusion/run_attention_suite.py --profile ustc_strict_time80_eval --mode attention --archive_tag ustc_strict_time80_splitcheck
```

备注：`ustc_strict_nofallback` 默认可能产生 Train-only 类，主要用于反证切分策略依赖，不建议作为最终 ACC/F1 主结果。

## 9. 附录 B：CIC4 full_packet l1024（仅改 CIC）

```bash
cd /home/shuora/Repositories/Traffic/FusionModel

# 预处理
python src/pipeline/dataset_builder.py --profile cic4_fullpacket_l1024_hraw --dataset_root dataset

# 训练 attention
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention --archive_tag cic4_fp_l1024_attn

# 可选：attention + stacking
python src/fusion/run_attention_suite.py --profile cic4_fullpacket_l1024_balanced --mode attention_stacking --archive_tag cic4_fp_l1024_stack
```

## 10. 附录 C：MFCP 快速命令

```bash
cd /home/shuora/Repositories/Traffic/FusionModel

# 1) 预处理
python src/pipeline/dataset_builder.py --profile mfcp_payload

# 2) Attention
python src/fusion/train_fusion_attention.py --dataset_name mfcp --preset none --epochs 24 --batch_size 64 --output_tag_prefix mfcp

# 3) Attention + Stacking
python src/fusion/train_fusion_attention_stacking.py --dataset_name mfcp --preset none --epochs 24 --batch_size 64 --output_tag_prefix mfcp
```
