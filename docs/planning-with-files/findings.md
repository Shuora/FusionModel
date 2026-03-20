# Findings

## 文档对齐结论（2026-03-18）

- `MobileViTBackbone` 当前为真实 `transformers.MobileViTForImageClassification` 主干，并在默认路径存在时复用本地 checkpoint：`/tmp/Shuora-MobileViT/malicious_traffic_mobilevit_model.pth`。
- ET-BERT 侧已具备工程化接入基础设施：
  - `vocab` 文件加载（`src/data/etbert_tokenizer.py`）
  - `config` / `config_path` 注入（`src/models/etbert_backbone.py`）
  - `num_layers` 截断
  - checkpoint 加载与映射诊断报告（`last_checkpoint_report` / `checkpoint_report`）
- 能力边界已在文档明确：
  - 当前 ET-BERT 侧为 ET-BERT 风格兼容 adapter，不是原始 UER ET-BERT 预训练模型的完整实现。
- 训练/评估/协议管线状态：
  - `train/evaluate/report + stage1/stage2 + stacking/moe` 相关测试链路可通过。
  - 用户指定回归命令已在本次文档更新后复验通过：`46 passed`。

## 运行时支持结论（2026-03-19）

- `src.train` 现已支持显式设备选择：
  - `--device auto`
  - `--device cpu`
  - `--device cuda`
- `src.train` 现已支持 `--num-workers`，并将解析后的 `device` / `num_workers` 写入 `config.yaml`。
- `src.evaluate` 现已支持：
  - CLI `--device`
  - 若未显式传入，则优先复用训练时保存的 `device_requested`
  - 当请求 `cuda` 但当前环境不可用时，自动回退到 `cpu`
- 当前实现仍有一条重要边界：
  - 数据在训练/评估前会整体载入内存，因此在 `8GB RAM` 机器上，内存往往比显存更早成为瓶颈。

## Stage1 论文协议核对结论（2026-03-20）

- 论文原文依据：`docs/paper/MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification.pdf` 第 10-12 页。
- 论文 Exp. I 使用的是：
  - `ISCX VPN-nonVPN`
  - `MTA`
  - `MFCP`
  - 不包含 `USTC`
- 论文 Table 1 为 ISCX 的 9 个 normal traffic group，并给出每组固定的 train/test 配额。
- 论文 Table 2 为 MTA 的 7 个家族，并给出每家族固定的 train/test 配额。
- 论文 Table 3 为 MFCP 的 6 个家族，并给出每家族固定的 train/test 配额。
- 论文明确写到 `MFCP` 做过 `trimmed some of the traffic`，因此论文协议不是“原始数据集全量样本直接混合”。
- 当前仓库 `stage1_binary` 的实现问题：
  - 仅通过 ISCX 文件名前缀白名单与 MTA/MFCP 家族白名单近似论文子集
  - 没有严格按论文表 1-3 的每组 train/test 配额构造 manifest
  - 保留了“匹配不到论文子集时 fallback 到未过滤数据”的行为，不符合严格复现要求
- 本轮实现边界：
  - 严格复现论文的类别/家族集合与每组 train/test 数
  - 使用仓库现有 `session_full` session 样本按稳定排序裁样
  - 不承诺与论文作者原始逐 session 列表一一对应
