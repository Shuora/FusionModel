# FusionModel

FusionModel 是一个面向加密流量分类实验的多模态融合仓库，当前 V4 版本聚焦于 `MobileViT + CharBERT` 的 attention 融合方案。项目把网络会话的二进制载荷转换为两种模态：

- 图像模态：把 `.bin` 会话字节映射为 `28x28` RGB 图像
- 序列模态：把同一个 `.bin` 会话作为字节序列送入 CharBERT 编码器

随后模型使用 cross-attention 融合图像特征与字节序列特征，完成二分类或多分类的加密流量识别任务。

## 当前版本包含的能力

- attention 融合训练
- attention + stacking 集成实验
- 原始 pcap/pcapng 到会话级 `.bin` 的切分
- 会话级 `.bin` 到 RGB 图像的批量生成
- 基础测试用例与命令行入口验证

## 支持的任务

当前仓库只支持以下 `task_name`：

- `binary_benign_vs_malicious`
- `ustc_multiclass`
- `mta_multiclass`
- `mfcp_multiclass`

其中：

- `binary_benign_vs_malicious` 会把 `ISCX-VPN-NonVPN-2016` 视为 `benign`，其余恶意数据集视为 `malicious`
- 其余三个任务分别对应单数据集多分类

## 仓库结构

```text
FusionModel/
├── src/
│   ├── run_all_modes.py
│   ├── train_fusion_attention.py
│   ├── train_fusion_attention_stacking.py
│   ├── split_data.py
│   ├── ssl_tls_rgb_image.py
│   ├── fusion_common.py
│   └── CharBERT/
├── tests/
├── tools/
├── docs/
├── SourceData/          # 原始 pcap 数据，需自行准备，默认被 git 忽略
├── ProcessedData/       # 处理后的任务数据，需自行生成，默认被 git 忽略
└── outputs/ 或 src/outputs/
```

说明：

- 请从仓库根目录执行所有命令。
- `SourceData/`、`ProcessedData/`、`outputs/` 默认不会被 Git 跟踪。
- 训练脚本默认输出目录是 [src/outputs](/home/shuora/Traffic/FusionModel/src/outputs)，不是仓库根下的 `outputs/`。

## 数据处理与训练流程总览

```text
原始 pcap/pcapng
  -> SourceData/<dataset>
  -> split_data.py
  -> ProcessedData/<task>/pcap_data/{Train,Test}/<label>/*.bin
  -> ssl_tls_rgb_image.py
  -> ProcessedData/<task>/image_data/{Train,Test}/<label>/*.png
  -> train_fusion_attention.py / run_all_modes.py
  -> 模型权重、日志、混淆矩阵、训练曲线、Markdown 报告
```

## 环境准备

### 1. 激活 conda 环境

```bash
conda activate FusionModel
```

### 2. 安装依赖

基础依赖：

```bash
python3 -m pip install -r requirements.txt
```

数据切分额外依赖：

```bash
python3 -m pip install dpkt
```

如果你需要单独检查 CharBERT 子目录依赖，也可以执行：

```bash
python3 -m pip install -r src/CharBERT/requirements.txt
```

### 3. 建议检查 Python 与 PyTorch

```bash
python3 --version
python3 -c "import torch; print(torch.__version__)"
```

## 数据目录约定

### 原始数据目录

`split_data.py` 默认从仓库根下的 `SourceData/` 读取原始数据。目录需要符合任务约定：

```text
SourceData/
├── ISCX-VPN-NonVPN-2016/
├── USTC-TFC2016/
├── MTA/
└── MFCP/
```

具体约定：

- `binary_benign_vs_malicious`
  - 读取 `ISCX-VPN-NonVPN-2016`、`USTC-TFC2016`、`MTA`、`MFCP`
  - 其中 `ISCX-VPN-NonVPN-2016` 映射为 `benign`
  - 其余映射为 `malicious`
- `ustc_multiclass`
  - 读取 `SourceData/USTC-TFC2016/*.pcap`
- `mta_multiclass`
  - 读取 `SourceData/MTA/<family>/*.pcap`
- `mfcp_multiclass`
  - 读取 `SourceData/MFCP/<family>/*.pcap`

### 处理后数据目录

训练前必须生成如下结构：

```text
ProcessedData/<task_name>/
├── pcap_data/
│   ├── Train/<label>/*.bin
│   └── Test/<label>/*.bin
├── image_data/
│   ├── Train/<label>/*.png
│   └── Test/<label>/*.png
└── metadata/manifest.json
```

如果 `pcap_data` 或 `image_data` 任一缺失，训练脚本会直接报错。

## 从零开始完整命令手册

下面直接给出四个任务各自的完整实验命令。所有命令都从仓库根目录执行，并先激活：

```bash
conda activate FusionModel
```

通用说明：

- `--source_root` 统一使用 `/home/shuora/Traffic/FusionModel/SourceData`
- `--dataset_root` 统一传 `ProcessedData` 的父目录 `/home/shuora/Traffic/FusionModel/ProcessedData`
- `split_data.py` 的 `--processed_root` 指向具体任务目录 `ProcessedData/<task_name>`
- 默认训练参数与当前仓库文档保持一致：`batch_size=32`、`num_workers=4`、`prefetch_factor=2`

### 实验 1：二分类 `binary_benign_vs_malicious`

原始数据目录：

```text
SourceData/ISCX-VPN-NonVPN-2016/
SourceData/USTC-TFC2016/
SourceData/MTA/
SourceData/MFCP/
```

完整命令：

```bash
python3 src/split_data.py   --task_name binary_benign_vs_malicious   --source_root /home/shuora/Traffic/FusionModel/SourceData   --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/binary_benign_vs_malicious   --train_ratio 0.8   --seed 42

python3 src/ssl_tls_rgb_image.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/binary_benign_vs_malicious

python3 src/train_fusion_attention.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name binary_benign_vs_malicious   --batch_size 32   --image_size 28   --max_pcap_length 784   --epochs 32   --lr 1e-3   --patience 8   --device auto   --seed 42   --num_workers 4   --pin_memory   --persistent_workers   --prefetch_factor 2   --output_dir /home/shuora/Traffic/FusionModel/src/outputs   --attention_dim 256

python3 src/train_fusion_attention_stacking.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name binary_benign_vs_malicious   --meta_methods xgboost

python3 src/run_all_modes.py   --mode all   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name binary_benign_vs_malicious
```

### 实验 2：USTC 多分类 `ustc_multiclass`

原始数据目录：

```text
SourceData/USTC-TFC2016/
```

完整命令：

```bash
python3 src/split_data.py   --task_name ustc_multiclass   --source_root /home/shuora/Traffic/FusionModel/SourceData   --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/ustc_multiclass   --train_ratio 0.8   --seed 42

python3 src/ssl_tls_rgb_image.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/ustc_multiclass

python3 src/train_fusion_attention.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name ustc_multiclass   --batch_size 32   --image_size 28   --max_pcap_length 784   --epochs 32   --lr 1e-3   --patience 8   --device auto   --seed 42   --num_workers 4   --pin_memory   --persistent_workers   --prefetch_factor 2   --output_dir /home/shuora/Traffic/FusionModel/src/outputs   --attention_dim 256

python3 src/train_fusion_attention_stacking.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name ustc_multiclass   --meta_methods xgboost

python3 src/run_all_modes.py   --mode all   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name ustc_multiclass
```

### 实验 3：MTA 多分类 `mta_multiclass`

原始数据目录：

```text
SourceData/MTA/
```

完整命令：

```bash
python3 src/split_data.py   --task_name mta_multiclass   --source_root /home/shuora/Traffic/FusionModel/SourceData   --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mta_multiclass   --train_ratio 0.8   --seed 42

python3 src/ssl_tls_rgb_image.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/mta_multiclass

python3 src/train_fusion_attention.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name mta_multiclass   --batch_size 32   --image_size 28   --max_pcap_length 784   --epochs 32   --lr 1e-3   --patience 8   --device auto   --seed 42   --num_workers 4   --pin_memory   --persistent_workers   --prefetch_factor 2   --output_dir /home/shuora/Traffic/FusionModel/src/outputs   --attention_dim 256

python3 src/train_fusion_attention_stacking.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name mta_multiclass   --meta_methods xgboost

python3 src/run_all_modes.py   --mode all   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name mta_multiclass
```

### 实验 4：MFCP 多分类 `mfcp_multiclass`

原始数据目录：

```text
SourceData/MFCP/
```

完整命令：

```bash
python3 src/split_data.py   --task_name mfcp_multiclass   --source_root /home/shuora/Traffic/FusionModel/SourceData   --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass   --train_ratio 0.8   --seed 42

python3 src/ssl_tls_rgb_image.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass

python3 src/train_fusion_attention.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name mfcp_multiclass   --batch_size 32   --image_size 28   --max_pcap_length 784   --epochs 32   --lr 1e-3   --patience 8   --device auto   --seed 42   --num_workers 4   --pin_memory   --persistent_workers   --prefetch_factor 2   --output_dir /home/shuora/Traffic/FusionModel/src/outputs   --attention_dim 256

python3 src/train_fusion_attention_stacking.py   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name mfcp_multiclass   --meta_methods xgboost

python3 src/run_all_modes.py   --mode all   --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData   --task_name mfcp_multiclass
```

## 常用训练参数说明

最常用参数如下：

- `--dataset_root`
  - 对于任务训练，应该传 `ProcessedData` 的父目录
  - 例如：`/home/shuora/Traffic/FusionModel/ProcessedData`
- `--task_name`
  - 必填，指定任务名
- `--batch_size`
  - 默认 `32`
- `--image_size`
  - 默认 `28`
- `--max_pcap_length`
  - 默认 `784`
- `--epochs`
  - 默认 `32`
- `--lr`
  - 默认 `1e-3`
- `--patience`
  - 默认 `8`
- `--num_workers`
  - 默认 `4`
- `--prefetch_factor`
  - 默认 `2`
- `--device`
  - 默认 `auto`
- `--output_dir`
  - 默认 [src/outputs](/home/shuora/Traffic/FusionModel/src/outputs)
- `--attention_dim`
  - 默认 `256`
- `--no_amp`
  - 禁用 CUDA AMP 混合精度
- `--no_index_cache`
  - 禁用数据索引缓存
- `--rebuild_index_cache`
  - 强制重建索引缓存
- `--class_balance`
  - 可选：`none`、`weighted_loss`、`weighted_sampler`、`weighted_sampler_loss`
- `--loss_type`
  - 可选：`ce`、`focal`
- `--preset`
  - 可选：`none`、`cic_balanced`

一个更偏稳定训练的示例：

```bash
python3 src/train_fusion_attention.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --task_name binary_benign_vs_malicious \
  --epochs 40 \
  --lr 3e-4 \
  --patience 10 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --grad_clip_norm 1.0
```

## 输出结果说明

训练完成后，默认会在 [src/outputs](/home/shuora/Traffic/FusionModel/src/outputs) 下生成：

- `logs/*.log`
- `fusion_model_attention_dim256.pth`
- `metrics_curve_*.png`
- `confusion_matrix_*.png`
- `report_*.md`

attention 模式还会额外生成 attention 诊断结果。

典型查看命令：

```bash
ls -lah src/outputs
ls -lah src/outputs/logs
tail -n 50 src/outputs/logs/*.log
```

## 测试命令

仓库当前使用 `unittest` 风格测试。可以在仓库根目录执行：

```bash
python3 -m unittest discover -s tests -v
```

如果只想验证关键入口：

```bash
python3 -m unittest tests.test_run_all_modes -v
python3 -m unittest tests.test_attention_entrypoints -v
python3 -m unittest tests.test_split_data_tasks -v
```

## 命令行帮助

以下命令可以直接查看当前版本支持的参数：

```bash
python3 src/split_data.py --help
python3 src/ssl_tls_rgb_image.py --help
python3 src/train_fusion_attention.py --help
python3 src/train_fusion_attention_stacking.py --help
python3 src/run_all_modes.py --help
```

## 辅助脚本说明

### `tools/sort_outputs_by_mode.py`

用途是按文件名里的 mode token 整理输出文件，但当前脚本包含硬编码历史路径，不适合作为通用跨平台命令直接写入实验流程。

### `tools/split_concat_log.py`

用途是把大的拼接日志拆回各自日志文件，同样包含硬编码路径，更适合作为历史维护脚本，而不是标准训练流程的一部分。

## 常见问题与注意事项

### 1. 为什么 README 里的命令都用 `python3`

因为当前环境中 `python` 命令不存在，直接使用 `python` 会报：

```text
command not found: python
```

### 2. 为什么训练脚本找不到数据

最常见原因有两个：

- 你传入的 `--dataset_root` 不是 `ProcessedData` 的父目录
- `ProcessedData/<task_name>` 下缺少 `pcap_data/Train`、`pcap_data/Test`、`image_data/Train` 或 `image_data/Test`

### 3. 为什么 `split_data.py` 报缺少 `dpkt`

因为根目录 `requirements.txt` 里没有包含 `dpkt`，需要手动安装：

```bash
python3 -m pip install dpkt
```

### 4. 为什么输出目录和我想的不一样

当前 V4 代码里，训练默认输出目录是：

```text
src/outputs
```

如果你希望改到别的位置，请显式传：

```bash
--output_dir /your/path
```

### 5. 这些数据目录为什么没有出现在 Git 里

因为以下目录默认被 `.gitignore` 忽略：

- `SourceData/`
- `ProcessedData/`
- `outputs/`
- `dataset/`

这意味着：

- 原始数据需要你自己准备
- 处理后数据需要你自己生成
- 训练结果默认不会被 Git 跟踪

## 推荐执行顺序

如果你是第一次接手这个仓库，建议严格按下面顺序执行：

```bash
conda activate FusionModel
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install dpkt
python3 src/split_data.py --task_name ustc_multiclass --source_root /home/shuora/Traffic/FusionModel/SourceData --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/ustc_multiclass
python3 src/ssl_tls_rgb_image.py --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/ustc_multiclass
python3 src/run_all_modes.py --mode all --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData --task_name ustc_multiclass
python3 -m unittest discover -s tests -v
```

这套命令覆盖了：

- 环境准备
- 原始数据切分
- 图像生成
- attention 与 stacking 训练
- 基础测试验证
