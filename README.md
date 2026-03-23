# FusionModel

面向流量分类的多模态实验工程，当前主线为 `session_full`：`PCAP -> Session PCAP -> RGB + ET-BERT 输入三元组`，并支持阶段1二分类与阶段2多分类协议执行。

## 当前架构（与代码一致）

- 图像分支：`src/models/mobilevit_backbone.py`
  - 使用 `transformers.MobileViTForImageClassification` 的 `mobilevit` 主干抽特征。
  - 默认尝试复用本地 checkpoint：`/tmp/Shuora-MobileViT/malicious_traffic_mobilevit_model.pth`。
  - 当前行为：仅在本地 checkpoint 存在时尝试加载；文件缺失或 `torch.load` 返回非 `dict` 时会跳过加载；其他参数不兼容情况可能在加载阶段抛错。
- 文本分支：`src/models/etbert_backbone.py`
  - ET-BERT 风格适配器，支持 `vocab/config/checkpoint` 接入与 checkpoint 映射加载报告。
  - 支持按 `num_layers` 对 encoder 层数截断（不超过配置层数）。
  - 兼容多种外部 key 风格并输出 `last_checkpoint_report/checkpoint_report` 诊断信息。
- 融合头：`src/models/fusion_model.py`
  - `MobileViTETBertFusionClassifier` 当前使用 `2-layer bidirectional fusion encoder`，不再使用旧的 gate fusion。
  - 图像侧从 `MobileViT` 多尺度 hidden states 提取并投影 image tokens，以适配当前 `28x28` RGB 输入。
  - 文本侧输出 `tokens + mask + pooled`，与图像 token 一起进入双向 cross-attention 融合。
  - 模型输出当前为 `logits_fuse/logits_img/logits_tls`，供 train/evaluate/stacking/moe 全链路复用。

说明：当前 ET-BERT 侧是“ET-BERT 风格兼容适配器”，并非原始 UER ET-BERT 预训练模型的完整等价实现。

## 环境准备

```bash
cd /home/shuora/Traffic/FusionModel
conda env create -f environment.yml
conda activate FusionModel
pip install -r requirements.txt
```

`environment.yml` 提供基础依赖；`requirements.txt` 补充了当前 MobileViT 所需的 `transformers` 等包。

## 数据目录约定

- `SourceData/CICAndMal2017`
- `SourceData/MFCP`
- `SourceData/USTC-TFC2016`
- `SourceData/ISCX` 或 `SourceData/ISCX-VPN-NonVPN-2016`（阶段1需要）
- `SourceData/MTA`（阶段1需要；默认阶段2任务也包含 MTA）

## 运行命令

预处理、阶段1二分类、阶段2多分类等详细且权威的运行命令，请查看 `docs/commands/session-full-experiments.md`。
