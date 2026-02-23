# Session Full 实验命令

本文档对应 `session_full` 论文口径流程：

- 预处理：`PCAP -> Session PCAP -> RGB+时序特征`
- 默认行为：特征提取后自动清理 `tmp_sessions`，同时保留抽检 `preview_png`
- 协议输出：
  - 阶段1混合二分类清单（ISCX=normal, 其余=malicious）
  - 阶段2多分类任务清单（MTA-7 / MFCP-6 / USTC-10）

## 1. 预处理（session_full）

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

只处理指定数据集（示例）：

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

如需保留 session pcap（调试）：

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --keep-sessions \
  --preview-per-family 20
```

## 2. 阶段1二分类清单生成（严格检查缺失数据集）

> 需要 `outputs/processed` 下同时存在：`ISCX`、`MFCP`、`MTA`、`USTC-TFC2016` 的 `session_full/manifest/session_manifest.*`。

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv
```

## 3. 阶段2多分类任务清单生成

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json
```

