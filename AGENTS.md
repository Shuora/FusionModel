尽可能用中文回答我

## 本项目简介

- 项目名称：`FusionModel`
- 任务目标：对加密恶意流量做图像+序列融合分类，核心模型为 `MobileViT + CharBERT + Attention`
- 数据范围：`USTC-TFC2016` 与 `CICAndMal2017`（CIC 按五大类 `Adware/Benign/Ransomware/SMSMalware/Scareware`）
- 当前实现路径：
  - 预处理：`src/pipeline/`
  - 训练：`src/fusion/`
  - 配置：`configs/`
  - 输出：`dataset/` 与 `outputs/`

## 仓库协作约束

- 不在原仓库直接修改代码：
  - `C:\Repositories\Traffic\Data-Processing`
  - `C:\Repositories\Traffic\CharBERT-MobileViT`
- 任何改造先落地到本仓库，再进行调试与验证。
- 预处理坚持 RGB 三通道，不使用灰度图。
- USTC 与 CIC 分开训练/验证，不做交替喂数。
- 训练需保留完整日志与评估产物（曲线、混淆矩阵、acc、f1、报告）。

## 术语与目录约定

- `dataset/<name>/pcap_data/{Train,Test}/<class>/*.bin`：序列输入
- `dataset/<name>/image_data/{Train,Test}/<class>/*.png`：RGB 图像输入
- `outputs/logs/*.log`：训练日志
- `outputs/confusion_matrix_*.png`：混淆矩阵
- `outputs/metrics_curve_*.png`：epoch 曲线
- `outputs/report_*.md`：评估报告
