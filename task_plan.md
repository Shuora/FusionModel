## Task
修复 `src/split_data.py` 在读取尾部残缺但主体可用的 `.pcap` 文件时直接报错并跳过整文件的问题。

## Plan
1. 在隔离 worktree 中检查当前 `split_data.py` 的 pcap 读取路径与现有测试覆盖。
2. 先新增一个尾部截断 `pcap` 的回归测试，确认当前行为失败。
3. 仅对 `.pcap` 的尾部不完整 packet header/data 增加容错，保留前面已经读出的包。
4. 运行相关 `unittest` 验证修复，并同步更新本次排查发现与进度记录。

## Constraints
- 不运行 `mvn test`。
- 只放宽 `.pcap` 尾部截断场景，不吞掉其他真实格式错误。
- 先写失败测试，再写生产代码。

---

## Task (2026-03-31 MFCP 样本一致性排查)
排查 `mfcp_multiclass` 处理结果中 `Cobalt` 家族缺失的根因，确认是否与论文口径不一致及其成因。

## Plan
1. 对齐三种口径：论文统计、`SourceData/MFCP` 原始文件统计、`ProcessedData/mfcp_multiclass` 处理后统计。
2. 检查 `src/split_data.py` 在截断 pcap 上的异常处理逻辑，并结合历史提交确认行为变化。
3. 用原始包级统计验证 `Cobalt.pcap` 是否实际包含可提取会话，避免误判为“无有效流量”。
4. 输出根因结论与可复现修复步骤（重建处理数据）。


## Task (2026-03-31 Early Stopping 严谨化)
将融合训练早停默认耐心轮次统一为 8，并审查/加固早停监控指标逻辑，避免模式与指标方向不一致或 NaN 指标导致误停。

## Plan
1. 统一 `EarlyStopping` 与 `train_fusion_model` 的默认 `patience=8`，消除默认值漂移。
2. 增加早停指标方向解析与校验（`auto` 自动推断、手动模式不一致时 fail-fast）。
3. 增加监控值有限性检查，遇到 NaN/Inf 时跳过该轮 early stop 与 ReduceLROnPlateau 更新。
4. 补充单元测试并同步 README、findings、progress。

## Task Status (2026-04-01 early stop 遇到 NaN 未停训)
- [x] 复核日志与训练循环，确认根因是 NaN 分支跳过 early-stop 更新。
- [x] 新增 NaN 场景回归测试。
- [x] 最小改动修复早停逻辑（非有限值按未改善处理并可触发停止）。
- [ ] 运行测试验证并回报结果。
- [x] 运行测试验证并回报结果。
- [x] 增加训练 batch 非有限 loss 保护与回归测试，并通过验证。

## Task (2026-04-01 训练记录审计)
检查 `outputs/` 下所有训练记录，识别异常训练、无效产物和可疑指标。

## Plan
1. 枚举全部 run 目录并按“完成/未完成”分类。
2. 对已完成 run 提取 `metrics.json` 与 `train.log` 的关键指标，识别崩坏点（NaN、单类塌缩、极端类不平衡影响）。
3. 对未完成 run 对齐启动参数与终止位置，区分人为中断与程序异常。
4. 输出问题清单与可执行建议，同步 findings/progress。

## Task (2026-04-02 全量训练日志审计)
审计 2026-04-01 ~ 2026-04-02 最新 8 个 attention / attention_stacking run，识别数值稳定性问题、任务级短板和可执行改进项。

## Plan
1. 汇总四个任务的 attention 与 attention_stacking `metrics.json` / `epoch_metrics.csv` / `train.log` 关键指标。
2. 对异常 run 做根因归类：数值爆炸、早停触发、类别不均衡、accuracy 与 macro_f1 偏离。
3. 对照训练入口与公共训练逻辑，确认是否存在可复现性或配置层面的系统性问题。
4. 输出按优先级排序的改进建议，并同步 findings/progress。

## Task (2026-04-03 多分类 stacking 全量优化)
针对 `mta_multiclass` 与 `mfcp_multiclass` 的弱表现，改造 attention stacking 流程，落地 OOF 元学习、元特征扩展、class-weighted XGBoost、多模型 soft-voting 与任务定向校正。

## Plan
1. 在 worktree 中先补失败测试，覆盖新逻辑核心函数（OOF、soft-voting、class gain、0/4 校正）。
2. 改造 `src/fusion_common.py`：
   - 增加元特征扩展（text/image/fusion 概率 + entropy + margin + agreement）；
   - 增加 OOF stacking 训练/评估路径；
   - 增加 class-weighted XGBoost；
   - 增加多 meta learner 的加权 soft-voting；
   - 增加 `mta` 类别阈值增益调优与 `mfcp` 0/4 二分类校正头。
3. 更新训练入口参数透传与 README 使用说明（涉及运行参数/流程变化）。
4. 运行目标测试并记录结果，同步更新 findings/progress。

## Task Status
- [x] 新增失败测试并确认红灯。
- [x] 实现 OOF stacking、元特征扩展、class-weighted meta learner、soft-voting。
- [x] 实现 `mta` 类增益调优与 `mfcp` 0/4 二分类校正头。
- [x] 更新 README 与过程文档。
- [x] 最终验证并输出结果摘要。

## Task (2026-04-04 修复“梯度无效”误告警/误跳过)
修复 attention/attention_stacking 在 AMP 训练下频繁出现“梯度无效（NaN/Inf）”并跳过 batch 的问题，恢复与早期版本一致的稳定训练行为。

## Plan
1. 复核 `train.log` 与训练循环，确认“梯度无效”主要出现在 AMP 分支且从首批次即可触发。
2. 先新增回归测试：模拟 AMP scaler overflow，验证不应计入 `invalid_grad_batches`（避免把可恢复 overflow 误判成梯度损坏）。
3. 最小改动 `src/fusion_common.py`：AMP 路径交由 `GradScaler.step/update` 处理 overflow，不再走“梯度无效”硬跳过分支；保留非 AMP 的有限值梯度保护。
4. 运行相关测试并同步更新 findings/progress；若行为文档受影响则同步 README。

## Task Status (2026-04-04 梯度无效修复)
- [x] 完成日志与代码链路复核，确认 root cause 在 AMP 分支误判 overflow。
- [x] 已新增失败测试并验证红灯。
- [x] 已完成最小代码修复（仅 AMP 分支）。
- [x] 已运行相关回归测试并通过。
- [x] 已同步更新 findings/progress 文档。

## Task (2026-04-04 mta/mfcp 定向增强)
仅针对 `mta_multiclass` 与 `mfcp_multiclass` 进一步增强 stacking 后处理，提升弱类稳定性且不影响其他任务。

## Plan
1. 审核现有 `fusion_common.py` 中 mta/mfcp 定向逻辑是否已覆盖（gain 与 0/4 校正）。
2. 在不改其他任务行为前提下增强：
   - `mta`：由固定类索引改为“按样本数最少类”自动选 gain 目标类；
   - `mfcp`：对 `0/4` 校正新增 OOF 驱动的 `alpha` 自适应。
3. 补充单测覆盖新增行为（`alpha=0` 不改变预测、自动调参不劣于基线）。
4. 运行目标测试并同步 README / findings / progress。

## Task Status (2026-04-04 mta/mfcp 定向增强)
- [x] 完成实现现状审计，确认已有 OOF/soft-voting/mta gain/mfcp pair correction。
- [x] 已完成 mta 最小类动态 gain 目标类改造（仅 mta 生效）。
- [x] 已完成 mfcp 二分类校正 `alpha` 自适应（仅 mfcp 生效）。
- [x] 运行目标测试并记录结果。
- [x] 同步更新 findings/progress/README。

## Task (2026-04-04 mfcp 后处理二次调优)
针对你反馈“效果仍不理想”，继续优化 `mfcp_multiclass` stacking 后处理：在 `0/4` pair 校正上引入 pair-f1 导向目标、概率温度校准与阈值搜索，优先压制 `0<->4` 混淆。

## Plan
1. 先在 `tests/test_stacking_improvements.py` 增加失败测试，覆盖 `pair_f1` 目标、概率校准和阈值搜索行为。
2. 在 `src/fusion_common.py` 新增 pair 专用打分/校准/阈值函数，并扩展 `tune_binary_correction_alpha_for_pair` 支持 `objective=\"pair_f1\"`。
3. 仅在 `mfcp` 分支接入新后处理链路（method 与 soft-voting 两条路径保持一致），记录新的 postprocess 参数。
4. 运行相关回归测试并同步 README、findings、progress。

## Task (2026-04-04 对齐论文 MTA/MFCP 训练分布)
将 `mta_multiclass` 与 `mfcp_multiclass` 的预处理样本类别与 Train/Test 数量对齐到论文 MVTBA Table 2/3，USTC 不调整。

## Plan
1. 在隔离 worktree 中以 TDD 先新增 `paper_mvtba` 分布模式失败测试（固定目标计数、缺类报错）。
2. 修改 `src/split_data.py`：增加 `--distribution_profile paper_mvtba`，仅对 `mta_multiclass`/`mfcp_multiclass` 启用固定抽样。
3. 运行 `tests.test_split_data_tasks` 验证通过。
4. 重建 `ProcessedData/mta_multiclass` 与 `ProcessedData/mfcp_multiclass`，并统计校验。
5. 同步更新 README、findings、progress。

## Task Status (2026-04-04 对齐论文 MTA/MFCP 训练分布)
- [x] 新增失败测试并确认红灯。
- [x] 实现 `paper_mvtba` 分布模式与 CLI 参数。
- [x] 相关测试转绿。
- [x] 重建 MTA/MFCP 处理数据并校验计数。
- [x] 同步更新 README。


## Task (2026-04-04 MTA stacking 指标提升修复)
针对你反馈的 `mta_multiclass` stacking 指标不理想，定位并修复导致泛化偏差与任务定向后处理失效的实现问题。

## Plan
1. 复核最新 MTA/USTC/binary 训练产物与分布差异，确认问题属于“数据难度 + 实现偏差”而非单点参数。
2. 先补失败测试，覆盖：
   - `MTA` 任务识别在包含 `IcedID` 时仍应触发；
   - stacking 元特征 loader 不应继承训练阶段 `WeightedRandomSampler/drop_last`。
3. 修改 `src/fusion_common.py`：
   - 引入 deterministic meta loader（顺序、全量、无 sampler、无 drop_last）；
   - 以 task hint + class signature 做 `mta/mfcp` 任务识别，修复 MTA 后处理漏触发；
   - 增加 OOF-test gap 诊断日志。
4. 回归测试并同步 README、findings、progress。

## Task Status (2026-04-04 MTA stacking 指标提升修复)
- [x] 完成根因定位与对比分析。
- [x] 新增失败测试并确认红灯。
- [x] 完成核心修复实现。
- [x] 回归测试通过并完成文档同步。

## Task (2026-04-05 预处理进度条与日志落盘)
将预处理阶段的逐条图片输出改为进度条展示，并补齐预处理日志落盘。

## Plan
1. 定位 `split_data.py` 与 `ssl_tls_rgb_image.py` 的输出/日志配置。
2. 修改图片生成流程：移除逐条 `Saved` 日志，使用进度条+processed/skipped 统计。
3. 为两个预处理脚本增加 `--log_file`，并默认写入任务目录 `metadata/*.log`。
4. 回归相关测试并同步 README。

## Task Status (2026-04-05 预处理进度条与日志落盘)
- [x] 完成代码改造。
- [x] 完成 README 同步。
- [x] 完成语法校验；单测受本地 numpy 环境污染影响未全绿（见 progress/findings）。

## Task (2026-04-06 参照 MTA 最新改动对齐 MFCP 训练/验证)
将 `mfcp_multiclass` 的 stacking 训练/验证后处理链路对齐到 MTA 近期“按任务语义而非硬编码索引”思路，修复新增 `Cobalt` 后 `0/4` 索引漂移导致的校正失效/误校正风险。

## Plan
1. 复核 `src/fusion_common.py` 中 MTA 最新改造点（任务识别鲁棒性、deterministic meta loader）并定位 MFCP 仍使用固定索引的路径。
2. 将 MFCP pair 校正从固定 `class_a=0,class_b=4` 改为按类名解析（`Artemis/Ursnif`），同时兼容 5 类与 6 类（含 `Cobalt`）分布。
3. 补充回归测试：覆盖 MFCP 含 `Cobalt` 的任务识别与 pair 索引解析。
4. 运行目标测试并同步 README / findings / progress。

## Task Status (2026-04-06 MFCP 训练/验证对齐)
- [x] 完成 MTA/MFCP 差异复核与根因定位。
- [x] 完成 MFCP pair 后处理按类名解析改造。
- [x] 补充 MFCP 含 `Cobalt` 回归测试。
- [x] 运行目标测试并记录结果。

## Task (2026-04-06 更新 MFCP 训练命令)
更新 `README.md` 中 `mfcp_multiclass` 的 stacking 训练命令，使其与当前 MTA 推荐稳定配置保持一致，避免示例命令与现有调优实践不一致。

## Plan
1. 定位 README 的 `mfcp_multiclass` 训练命令区块。
2. 仅修改 `attention_stacking` 命令参数，保持入口脚本与路径不变。
3. 同步更新 findings/progress。

## Task Status (2026-04-06 MFCP 训练命令更新)
- [x] README 命令已更新。
- [x] findings/progress 已同步。

## Task (2026-04-07 CharBERT 兼容式升级设计)
在不改变现有 attention/attention_stacking 训练入口的前提下，将当前“命名为 CharBERT 的轻量 byte Transformer”升级为更接近真正 CharBERT 思想的 char-aware byte encoder，并先产出设计 spec。

## Plan
1. 审核当前 `src/CharBERT/src` 与 `src/fusion_common.py` 的文本分支实现，确认与真正 CharBERT 的差距。
2. 在“兼容式升级”约束下形成多方案对比，确定推荐方案（char-aware + 分层门控融合）。
3. 编写设计文档到 `docs/superpowers/specs/2026-04-07-charaware-byte-charbert-design.md`，覆盖架构、数据流、参数兼容、checkpoint 兼容、验证与回滚。
4. 完成 spec 自检并同步 `findings.md`、`progress.md`。

## Task Status (2026-04-07 CharBERT 兼容式升级实现)
- [x] 先补测试：新增 CLI/kwargs 参数透传与 `charaware` 构建回归测试并确认红灯。
- [x] 完成 `src/CharBERT/src/model.py` 的 char-aware 升级（char encoder + layer-wise fusion）。
- [x] 完成 `src/fusion_common.py` 参数扩展与模型初始化透传，保持入口脚本不变。
- [x] 同步更新 `README.md` 参数文档与启用示例。
- [x] 运行目标测试并通过。

## Task (2026-04-07 二层代价敏感 Stacking 全量改造)
将 `attention_stacking` 从当前单层 stacking + soft-voting 主路径升级为二层 cost-sensitive stacking 主路径，覆盖四个任务并优先提升弱类召回；保留单层/soft-voting 作为对照与降级路径。

## Plan
1. 在 worktree 中先补失败测试：新增 two-level 参数、校准、Level-2 特征、阈值优化、two-level 编排与报告字段测试。
2. 改造 `src/fusion_common.py`：
   - 增加 two-level 参数接线与 `build_common_kwargs` 透传；
   - 增加多分类校准（temp/isotonic）与 ECE/Brier 统计；
   - 增加 Level-2 特征构造、per-class threshold 优化；
   - 增加 two-level blender 主链路与 `requested/effective level` 降级策略；
   - 在 `metrics.json` 增加 stacking 配置与 two-level 后处理字段。
3. 同步更新 `README.md`（四任务独立 attention_stacking 命令与新参数说明）。
4. 运行目标回归并同步 `findings.md`、`progress.md`。

## Task Status (2026-04-07 二层代价敏感 Stacking 全量改造)
- [x] 完成 TDD 红灯（新增 9 个 two-level 相关测试，确认失败）。
- [x] 完成核心实现（参数接线、校准、Level-2、阈值优化、主链路与落盘）。
- [x] 完成 README 同步（四任务命令+参数说明）。
- [x] 运行完整回归并同步 findings/progress。

## Task (2026-04-07 二层 Stacking 严格复核补齐)
对已实现 two-level 方案执行二次严格复核，补齐“函数存在但未接入主流程”与“调用签名不一致”类隐性缺口，并再次完成回归验证。

## Plan
1. 修复 two-level postprocess 参数接线，确保 `threshold_objective/objective_value` 被正确落盘。
2. 将 `single_layer_baseline` 正式接入 `method_results` 主流程。
3. 补充对应单测并执行全量目标回归。
4. 同步 `README.md`、`findings.md`、`progress.md`。

## Task Status (2026-04-07 二层 Stacking 严格复核补齐)
- [x] 参数接线与主流程接入已完成。
- [x] 单测补充并通过。
- [x] 全量目标回归通过。
- [x] 文档与过程记录已同步。

## Task (2026-04-07 预处理日志补充家族统计)
增强预处理日志可观测性：除文件处理数量外，输出生成家族数，以及每个家族的 Train/Test/Total 样本统计。

## Plan
1. 在 `split_data.py` 增加家族级统计汇总函数并接入日志输出。
2. 按 TDD 补充单元测试，覆盖统计结果与日志输出内容。
3. 运行 `tests.test_split_data_tasks` 验证通过。
4. 同步更新 README、findings、progress。

## Task Status (2026-04-07 预处理日志补充家族统计)
- [x] 代码实现完成：新增汇总日志与家族 Train/Test/Total 输出。
- [x] 测试已补齐并通过（15 tests, OK）。
- [x] README/findings/progress 已同步。

## Task (2026-04-07 MTA/MFCP 训练问题定位)
基于 `outputs/` 中 MTA 与 MFCP 的训练日志、`metrics.json`、分类报告与处理后分布，判断主要瓶颈属于模型能力、分支平衡、数据分布还是集成学习链路。

## Plan
1. 收集并对比 MTA/MFCP 最新 attention + attention_stacking 的关键指标（macro_f1、弱类 recall、OOF-test gap）。
2. 从 `train.log` 排查训练稳定性与集成链路告警（NaN/Inf、early stop、OOF gap warning）。
3. 从 `ProcessedData/*/metadata/manifest.json` 统计 Train/Test 类别分布不均衡程度。
4. 按证据给出 root cause 排序与下一步优化优先级。

## Task Status (2026-04-07 MTA/MFCP 训练问题定位)
- [x] 指标/日志/分类报告已完成交叉核对。
- [x] 类别分布统计完成（MTA max/min≈66:1，MFCP max/min≈34.6:1）。
- [x] 已形成结论：MTA 以分布不均衡为主，MFCP 以类间可分性不足为主；集成学习当前不是主故障源。

## Task (2026-04-07 MTA/MFCP 均衡下载链接清单)
从 MTA 与 MFCP 官方页面抓取可下载 pcap 链接，按家族尽量均衡生成 CSV 并落盘。

## Plan
1. 抓取并解析 `https://www.stratosphereips.org/datasets-malware` 与 `https://malware-traffic-analysis.net/` 的链接信息。
2. 从 MTA 现有整理文件提取 7 类家族 direct pcap.zip 链接，按每类固定数量采样。
3. 从 MFCP 数据集页提取家族候选 capture，并二次解析 capture 页面得到 direct `.pcap` 下载链接。
4. 输出 MTA、MFCP、combined 三份 CSV 到 `outputs/`。

## Task Status (2026-04-07 MTA/MFCP 均衡下载链接清单)
- [x] 官方页面已抓取并解析。
- [x] 已生成 MTA 每类 20 条（共 140 条）链接清单。
- [x] 已生成 MFCP 每类目标 2 条（共 12 条）链接清单，并尽量补齐稀缺家族。
- [x] CSV 已落盘到 `outputs/`。

## Task (2026-04-10 MTA 最新训练日志复盘)
复盘最新 MTA 训练日志，确认训练是否完成、是否出现数值异常或早停异常，并给出无提升的原因判断与下一步建议。

## Plan
1. 定位最新 MTA run（attention / attention_stacking / attention_stacking_v2）日志与产物。
2. 复核 train.log 结尾与关键指标走势，检查是否中断、是否存在 NaN/Inf 或早停异常。
3. 对比此前 MTA run 的已知基线，输出问题归因与改进建议。

## Task Status (2026-04-10 MTA 每类约1w整理)
- [x] 已回滚精确 1w/类版本。
- [x] 已按“每类约1w”完成重采样（上限 12000，小类不放大）。
- [x] 已完成 manifest 与目录计数校验。
- [x] 已保留可回滚快照目录。

## Task (2026-04-13 时序预处理增强)
将融合流水线的时序分支从“纯字节流裁剪”升级为“保留 packet 边界、方向、长度与 `delta_t` 的分层输入”，同时尽量保持现有训练入口与图像分支接口不变。

## Plan
1. 审计当前 `split_data.py`、`fusion_common.py` 与 `CharBERT` 文本分支的数据契约，确认 packet 级信息现在哪里被丢弃。
2. 先新增回归测试，覆盖：
   - 新的时序样本格式是否保留 packet 边界、方向、长度与时间间隔；
   - 数据集加载器是否能解析分层 packet 序列；
   - 旧字节流输入的兼容行为是否明确。
3. 最小改造预处理与加载链路：
   - `split_data.py` 输出新的时序样本表示；
   - `fusion_common.py` 的数据集读取与 CharBERT 输入适配分层 packet 序列；
   - 如有必要，补充轻量的 packet/session 聚合逻辑，但不扩大训练入口。
4. 运行目标测试并同步更新 `README.md`、`findings.md`、`progress.md`，明确新格式与验证方式。

## Task Status (2026-04-13 时序预处理增强)
- [x] 已审计当前时序输入契约，确认 packet 级信息在 `extract_sessions` / `load_pcap_data` 处丢失。
- [x] 已补红灯测试，覆盖 packet 元数据写入与 temporal token 生成。
- [x] 已实现 packet 级 sidecar、hierarchical byte 序列加载与 legacy fallback。
- [x] 已运行相关回归测试并通过。
- [x] 已同步 README / AGENTS / findings / progress。
