# 实验与基线管理目录设计 (experiments/)

本设计旨在项目中建立一个规范的、可扩展的实验管理体系，用于存放基线模型（Baselines）、论文复现（Reproduction）以及对比实验（Comparisons）。

## 目录结构

```text
experiments/
├── README.md           # 目录导览与运行规范
├── baselines/          # 经典模型与基线实现
│   └── .gitkeep
├── reproduction/       # 他人论文复现代码
│   └── .gitkeep
└── comparisons/        # 对比实验脚本、统计与可视化
    └── .gitkeep
```

## 各子目录规范

### 1. baselines/
- **定位**：存放行业公认的基线模型（如 CNN, LSTM, ResNet, Transformer 原型等）。
- **要求**：
  - 每个模型独立文件夹。
  - 必须包含简单的 `run.sh` 或 `README.md` 说明如何启动训练/评估。
  - 尽量复用项目 `src/` 中的数据加载逻辑以确保公平对比。

### 2. reproduction/
- **定位**：存放针对特定论文（如 `ATVITSC`, `MVTBA` 等）的完整复现代码。
- **要求**：
  - 文件夹以论文缩写或作者命名。
  - 包含对原论文环境要求的说明。

### 3. comparisons/
- **定位**：跨模型的横向对比。
- **内容**：
  - 汇总各实验 `metrics.json` 的脚本。
  - 绘图脚本（PR 曲线、ROC 曲线、混淆矩阵对比图）。
  - LaTeX 表格生成脚本。

## 集成要求

- **数据共享**：实验代码应优先读取 `ProcessedData/` 下的标准化数据。
- **日志记录**：所有实验输出应重定向至 `outputs/experiments/` 下的对应子目录。
- **文档同步**：新增重要基线或复现结果时，应在 `findings.md` 中记录关键结论。
