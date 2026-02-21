## 为什么

当前仓库仅覆盖 `USTC-TFC2016` 与 `CICAndMal2017`，而 `SourceData` 已新增 `MFCP` 数据集，导致现有预处理与训练流程无法直接复用。现在补齐 MFCP 支持可以在不影响既有两套数据流程的前提下，统一实验入口并持续产出可对比的训练与评估结果。

## 变更内容

- 新增 `MFCP` 的数据预处理流程与配置，生成与现有数据集一致的目录结构：`dataset/<name>/pcap_data/{Train,Test}/<class>/*.bin` 与 `dataset/<name>/image_data/{Train,Test}/<class>/*.png`。
- 复用既有数据集的索引缓存逻辑：MFCP 首次读取执行全量扫描并落盘索引，后续读取优先复用索引；当数据变化导致索引失效时自动重建。
- 对 MFCP `pcap` 读取新增尾部截断容错：当文件前序包可读但尾部存在不完整记录时，保留已解析会话继续产出，并记录可审计告警日志。
- 明确 `MFCP` 图像预处理保持 RGB 三通道，不引入灰度图分支。
- 扩展训练与验证配置/入口，使 `MFCP` 可以独立完成训练与验证，并生成与现有流程一致的日志和评估产物（日志、曲线、混淆矩阵、报告）。
- 保证 `USTC-TFC2016` 与 `CICAndMal2017` 现有行为不变，不改变其数据组织、训练流程和默认结果输出约定。
- 更新 `README`，补充 `MFCP` 预处理、训练与验证命令。

## 功能 (Capabilities)

### 新增功能
- `mfcp-data-pipeline`: 为 MFCP 增加端到端预处理能力，产出与现有数据集一致的数据目录与文件格式，复用索引缓存逻辑，并满足 RGB 三通道约束。
- `mfcp-training-workflow`: 为 MFCP 增加训练、验证与评估产物输出能力，复用现有日志/指标输出结构，并在文档中提供可执行命令。

### 修改功能
- （无）当前仓库未发现可复用的既有 capability specs 需要增量修改。

## 影响

- 代码目录：`src/pipeline/`、`src/fusion/`
- 配置目录：`configs/`
- 文档：`README.md`
- 输出约定：`dataset/` 与 `outputs/` 下新增 MFCP 对应产物与索引缓存文件，不改变 USTC/CIC 既有产物结构
