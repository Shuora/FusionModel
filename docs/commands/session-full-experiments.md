# Session Full 实验命令（MobileViT + ET-BERT Adapter）

本文档只记录当前代码可执行口径：

- 预处理：`PCAP -> Session PCAP -> RGB + ET-BERT(input_ids/attention_mask/token_type_ids)`
- 图像主干：`transformers.MobileViTForImageClassification.mobilevit`（支持本地 checkpoint 复用）
- 文本主干：`ETBertBackbone` 兼容适配器（支持 vocab/config/checkpoint 接入）

说明：文本侧仍不是原始 UER ET-BERT 预训练模型的完整实现，而是兼容其 vocab/config/checkpoint 形态的工程化 adapter。

## 0. 环境与路径

假设 Conda 环境已创建（创建步骤见 `README.md`），此处只做激活与依赖补齐：

```bash
cd /home/shuora/Traffic/FusionModel
conda activate FusionModel
pip install -r requirements.txt
```

## 1. 预处理（session_full）

全量预处理：

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

仅处理指定数据集：

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --datasets MFCP USTC-TFC2016 \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

调试保留 session pcap：

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --keep-sessions \
  --preview-per-family 20
```

中断续跑：

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20 \
  --resume
```

## 2. 阶段1（二分类，ISCX vs 恶意）

生成 manifest：

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv
```

完整执行（train -> evaluate -> report）：

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv \
  --execute \
  --run-root runs \
  --run-id stage1-binary \
  --stage fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

约束说明：

- 必须存在 `ISCX`（或 `ISCX-VPN-NonVPN-2016` 别名）、`MFCP`、`MTA` 三个数据集的 `session_full/manifest/session_manifest.*`。
- 标签规则为 `ISCX=0(normal)`，`MFCP/MTA=1(malicious)`。

## 3. 阶段2（三任务多分类）

生成任务文件（仅 3 个基础任务，不包含 USTC 限样任务）：

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json
```

完整执行（每任务 train -> evaluate -> report）：

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json \
  --execute \
  --processed-root outputs/processed \
  --policy session_full \
  --run-root runs \
  --stage fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

默认任务：

- `MTA` 7 类
- `MFCP` 6 类
- `USTC-TFC2016` 10 类

执行行为说明：

- `stage2_tasks.json` 只记录上述 3 个基础任务。
- USTC `4000/3000/2000` 限样任务只在 `--execute` 时额外触发，不会写入 `stage2_tasks.json`。
- 可通过 `--skip-ustc-limited` 关闭这些额外限样任务。
