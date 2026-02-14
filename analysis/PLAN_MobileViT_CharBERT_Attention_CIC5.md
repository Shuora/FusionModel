# MobileViT + CharBERT + Attention（CIC5）改良计划

## 1. 目标与边界
- 在 `FusionModel` 仓库内落地一套独立流程，不修改原仓库：
  - `C:\Repositories\Traffic\Data-Processing`
  - `C:\Repositories\Traffic\CharBERT-MobileViT`
- 模型主线固定为：`MobileViT + CharBERT + Attention fusion`，保留 `Attention + Stacking`。
- USTC 与 CIC 分开训练与验证，不做交替喂数据。
- CIC 任务固定为 5 类：`Adware, Benign, Ransomware, SMSMalware, Scareware`。
- 验收目标：
  - USTC 准确率尽量保持（允许轻微波动，建议 <= 0.5pp）
  - CIC5 目标 `acc >= 98%`，并同时关注 `macro-F1`

## 2. 总体技术路线
- 迁移原 `CharBERT-MobileViT` 的融合训练主干到本仓库后再做改良。
- 预处理吸收 `analysis/MVTBA.md` 的 session + 784 思路，并结合原 RGB 三通道方案（坚持 RGB，不使用灰度）。
- 在 CIC 预处理阶段完成子家族到五大类的稳定映射，确保标签空间一致。
- 训练与评估全流程产出高密度日志、进度条、曲线图、混淆矩阵与分类报告。

## 3. 计划中的目录结构
- `src/pipeline/`
  - `dataset_builder.py`：统一预处理入口（USTC/CIC）
  - `pcap_session.py`：pcap -> session、去重、切分
  - `feature_rgb.py`：784 + 统计特征生成 `28x28x3`
- `src/fusion/`
  - `fusion_common.py`
  - `train_fusion_attention.py`
  - `train_fusion_attention_stacking.py`
  - `run_attention_suite.py`
- `configs/`
  - `dataset_profiles.yaml`
  - `train_profiles.yaml`
- `outputs/<timestamp>/`
  - `logs/`、`reports/`、`figures/`、`models/`、`metrics/`
- 文档与工程说明
  - `requirements.txt`
  - `README.md`
  - `AGENTS.md`

## 4. 预处理方案
### 4.1 输入与标签
- USTC：`SourceData/USTC-TFC2016/*.pcap`，由文件名映射类别。
- CIC：`SourceData/CICAndMal2017/<Major>/<Subclass>/*.pcap`。
- 标签策略：CIC 全部映射到 `<Major>` 五类，`Benign` 与四类恶意同级。

### 4.2 切分与防泄漏
- 默认按原始 `pcap` 文件切分（非 session 随机切分），再展开 session。
- 固定随机种子，保证可复现。
- 每类独立切分，保证 Train/Test 标签集合一致。

### 4.3 双轨字节源（做对照）
- `payload` 轨：TCP/UDP payload 优先（贴近原实现）。
- `full_packet` 轨：完整包字节，配合可复现头字段净化（IP/MAC/端口）。

### 4.4 session 与清洗
- 五元组聚合 session，按时间拼接字节。
- 去空会话、会话级 hash 去重。
- 固定长度 784 bytes（截断/补零）。

### 4.5 RGB 生成（非灰度）
- 输出固定 `28x28x3`。
- R：784 主序列（MVTBA 核心）。
- G：前缀/握手统计增强。
- B：会话行为统计增强。

## 5. 训练与评估方案
### 5.1 训练主线
- 主线 1：`attention`
- 主线 2：`attention_stacking`
- 保留原有参数风格：`--preset`、`--class_balance`、`--loss_type` 等。

### 5.2 训练输出
- 实时显示：`tqdm` 进度条 + `loss/acc`
- 每个 epoch 记录：`train/val loss`、`acc`、`macro-F1`
- 最终输出：classification report、混淆矩阵、指标变化曲线、模型权重
- 日志风格：中文高密度 + icon + 关键英文术语

## 6. CLI 与运行入口（计划）
### 6.1 预处理 CLI
- `python src/pipeline/dataset_builder.py --profile ustc`
- `python src/pipeline/dataset_builder.py --profile cic5_payload`
- `python src/pipeline/dataset_builder.py --profile cic5_fullpacket`

### 6.2 训练 CLI
- `python src/fusion/train_fusion_attention.py --dataset_name CIC5_payload --preset cic_balanced`
- `python src/fusion/train_fusion_attention_stacking.py --dataset_name CIC5_payload --preset cic_balanced`
- `python src/fusion/run_attention_suite.py --dataset_name USTC-TFC2016`

## 7. 新增文档与依赖计划（本次补充）
### 7.1 requirements.txt 编写
- 明确基础运行依赖（深度学习、数据处理、可视化、评估、日志、yaml 读取等）。
- 给出版本下限或建议版本区间，避免环境漂移导致复现失败。
- 按模块分组写注释（训练、预处理、可视化、工具）。

### 7.2 README.md 编写（重点补充 PowerShell 指令）
- 项目概述：目标、模型结构、数据集范围（USTC/CIC5）。
- 环境搭建：Windows + PowerShell 下的创建环境、安装依赖指令。
- 数据准备：目录结构示例与注意事项（Benign 放置规则）。
- 运行指令（PowerShell）：
  - 预处理命令示例
  - 训练命令示例
  - 评估与结果查看命令示例
- 结果说明：日志、曲线图、混淆矩阵、报告文件位置。
- 常见问题：路径、编码、权限、随机种子与复现说明。

### 7.3 AGENTS.md 项目介绍补充
- 增加“本项目简介”段落：任务目标、核心模型、工作流边界。
- 增加“仓库协作约束”段落：不改原仓库、先计划后实现、日志与产物要求。
- 增加“术语与目录约定”段落：CIC5 标签定义、关键目录用途。

## 8. 验收标准
- 预处理验收：
  - image/bin 一一匹配率 100%
  - Train/Test 标签一致
  - CIC 标签严格为五类（含 `Benign`）
- 训练验收：
  - USTC 与历史最优相比降幅 <= 0.5pp（目标）
  - CIC5 `acc >= 98%`，并检查 `macro-F1` 与混淆矩阵
- 对照结论：
  - `payload` 与 `full_packet` 双轨比较后，固定最优轨为默认

## 9. 风险与回退
- 风险：
  - Benign 分布变化导致阈值波动
  - full_packet 可能引入额外噪声
  - CIC 子类映射错误会直接拉低准确率
- 回退策略：
  - 优先回退到 `payload` + `attention` 主线
  - 保留各实验配置与日志，支持快速定位回退点

## 10. 当前状态声明
- 本文档为“计划修订版”，仅定义后续执行清单。
- 当前未实施代码改动（除计划文档本身外）。
- 待你确认后，再进入实现阶段。
