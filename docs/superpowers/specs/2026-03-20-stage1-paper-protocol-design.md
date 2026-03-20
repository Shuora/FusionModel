# Stage1 Paper Protocol Design

## Goal

将 `src.experiments.stage1_binary` 从“近似论文白名单筛选”改为“按论文表 1-3 严格构造 stage1 binary manifest”的协议实现。

## Background

当前仓库的 `stage1_binary` 基于 `session_full` 预处理结果，使用：

- ISCX 文件名前缀白名单
- MTA / MFCP 家族白名单

来近似论文子集。这种做法可以大致对齐论文使用的数据集名称，但不能严格对齐论文给出的：

- ISCX 9 个 normal traffic group
- MTA 7 个家族及其 train/test 数
- MFCP 6 个家族及其 train/test 数

论文原文依据：

- Table 1: ISCX VPN-nonVPN dataset
- Table 2: MTA dataset
- Table 3: MFCP dataset
- Exp. I: `Mix ISCX VPN-nonVPN, MTA, and MCFP`

## Scope

- 修改 `src/experiments/stage1_binary.py`
- 修改 `tests/pipeline/test_stage1_binary_protocol.py`
- 视需要微调 `tests/pipeline/test_protocol_execution.py`
- 更新 `docs/commands/session-full-experiments.md`
- 更新 `docs/planning-with-files/findings.md`
- 更新 `docs/planning-with-files/progress.md`

不修改：

- `session_full` 预处理主链路
- `stage2_multiclass`
- `train/evaluate/report`

## Paper Protocol To Reproduce

### Stage1 datasets

- `ISCX VPN-nonVPN`
- `MTA`
- `MFCP`

不包含 `USTC`。

### ISCX groups

按论文 Table 1，保留 9 个 normal traffic group：

- Chat / VPN / `facebook chat` / train 927 / test 232
- File transfer / VPN / `ftps, sftp, skype file` / train 805 / test 201
- Streaming / VPN / `hangouts audio` / train 2538 / test 634
- VoIP / VPN / `voipbuster` / train 1294 / test 324
- Email / nonVPN / `email` / train 2798 / test 699
- Streaming / nonVPN / `hangouts audio` / train 1384 / test 346
- Chat / nonVPN / `skype chat` / train 3542 / test 886
- P2P / nonVPN / `Torrent` / train 836 / test 209
- VoIP / nonVPN / `voipbuster` / train 1420 / test 355

### MTA families

按论文 Table 2，保留 7 个家族及配额：

- Dridex / 492 train / 123 test
- Emotet / 3368 train / 842 test
- Hancitor / 13452 train / 3363 test
- IcedID / 1454 train / 364 test
- Qakbot / 3350 train / 838 test
- Trickbot / 1794 train / 448 test
- Ursnif / 506 train / 127 test

### MFCP families

按论文 Table 3，保留 6 个家族及配额：

- Artemis / 6000 train / 1500 test
- Cobalt / 1501 train / 375 test
- Dridex / 6000 train / 1500 test
- PUA / 5614 train / 1403 test
- TrickBot / 6000 train / 1500 test
- Ursnif / 6000 train / 1500 test

## Reproduction Boundary

仓库当前 `session_full/manifest/session_manifest.csv` 中不保存论文作者原始的逐样本 train/test 名单，也不保存 MFCP 的 `CTU Num` 字段。因此本轮“严格复现”的定义是：

- 严格复现论文的类别/家族集合
- 严格复现论文的每组 train/test 配额
- 使用仓库当前可见的 session 样本，通过稳定、可重复的排序规则裁样

不承诺：

- 与论文作者原始 session 列表逐条一致
- 复原 MFCP 原文中 `trimmed some of the traffic` 的精确裁剪细节

## Implementation Approach

### Manifest construction

在 `stage1_binary.py` 中引入论文表驱动协议定义：

- ISCX：按 traffic group 配置匹配 `capture_id`
- MTA：按 family 配置匹配
- MFCP：按 family 配置匹配

每个 group/family 独立裁样：

1. 先筛出候选样本
2. 按稳定键排序：推荐 `(capture_id, session_id)`
3. 从候选中精确选出论文要求的 `train` 数量
4. 再精确选出论文要求的 `test` 数量
5. 合并为最终 manifest

### Split handling

优先复用现有 `session_manifest` 内的 `split` 字段：

- `split == train` 的候选只用于 train 配额
- `split == test` 的候选只用于 test 配额

若某组 train/test 任一侧样本不足，直接报错，并在错误信息中包含：

- dataset
- group/family
- required_train / available_train
- required_test / available_test

### Labels

保留现有标签映射：

- `ISCX -> label_binary=0 / normal`
- `MTA/MFCP -> label_binary=1 / malicious`

## Error Handling

- 缺任一必需数据集：报错
- 某一组没有匹配到候选样本：报错
- 某一组候选数少于论文配额：报错
- 不再保留“paper subset matched zero rows 时 fallback to unfiltered dataset”的行为

## Testing Strategy

### Unit-like protocol tests

在 `tests/pipeline/test_stage1_binary_protocol.py` 中新增/更新：

- `torrent` 必须被保留
- `PUA` 必须被保留
- ISCX 9 组按表 1 精确裁到目标数
- MTA 7 家族按表 2 精确裁到目标数
- MFCP 6 家族按表 3 精确裁到目标数
- 样本不足时报错
- 同一输入多次构造 manifest 结果稳定

### Execution tests

保留 `stage1_binary --execute` 的行为测试，确保：

- 仍会保存 manifest
- 仍会把 manifest 透传给 train

## Documentation Notes

需要明确说明：

- `stage1_binary` 现在按论文 Table 1-3 构造
- 这是“类别与数量严格复现”
- 不是“作者原始逐 session 名单逐条还原”
