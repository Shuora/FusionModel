# 初始化实验目录 (experiments/) 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在项目根目录创建 `experiments/` 文件夹及其子结构，用于管理基线、复现与对比实验。

**Architecture:** 按照功能划分为 `baselines/`, `reproduction/`, `comparisons/` 三个子目录，并包含必要的 `.gitkeep` 与导览文件。

**Tech Stack:** Shell commands, Markdown.

---

### Task 1: 创建目录结构与占位文件

**Files:**
- Create: `experiments/baselines/.gitkeep`
- Create: `experiments/reproduction/.gitkeep`
- Create: `experiments/comparisons/.gitkeep`

- [ ] **Step 1: 执行目录创建命令**

Run: `mkdir -p experiments/baselines experiments/reproduction experiments/comparisons`

- [ ] **Step 2: 创建 .gitkeep 文件确保目录被 Git 追踪**

Run: `touch experiments/baselines/.gitkeep experiments/reproduction/.gitkeep experiments/comparisons/.gitkeep`

- [ ] **Step 3: 验证目录结构**

Run: `ls -R experiments/`
Expected:
```
experiments/:
baselines  comparisons  reproduction

experiments/baselines:
.gitkeep

experiments/comparisons:
.gitkeep

experiments/reproduction:
.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add experiments/
git commit -m "chore: initialize experiments directory structure"
```

---

### Task 2: 撰写 experiments/README.md

**Files:**
- Create: `experiments/README.md`

- [ ] **Step 1: 写入 README 内容**

```markdown
# 实验与基线 (Experiments & Baselines)

本目录用于存放主模型（V4 MobileViT + CharBERT）以外的所有对比工作。

## 目录指南

- **`baselines/`**: 存放通用基线模型（如 ResNet, LSTM, Transformer）。
- **`reproduction/`**: 存放针对特定论文（如 ATVITSC）的完整复现。
- **`comparisons/`**: 存放用于生成对比图表和汇总指标的分析脚本。

## 运行约定

1. **数据**: 请统一读取根目录下的 `ProcessedData/`。
2. **输出**: 实验产物请输出至 `outputs/experiments/<experiment_name>/`。
3. **指标**: 确保每个实验最后能产出符合项目规范的 `metrics.json`。
```

- [ ] **Step 2: 验证文件内容**

Run: `cat experiments/README.md`

- [ ] **Step 3: Commit**

```bash
git add experiments/README.md
git commit -m "docs: add README for experiments directory"
```

---

### Task 3: 同步更新项目文档 (AGENTS.md & README.md)

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: 在 AGENTS.md 中添加实验目录规范**

在 `Project Notes` 部分添加：
> - 实验管理：所有基线模型与论文复现代码存放在 `experiments/` 目录下，遵循其内部 README 规范。

- [ ] **Step 2: 在 README.md 项目结构部分添加新目录说明**

找到 `Project Structure` 树形图，插入：
`├── experiments/         # Baselines, reproduction, and comparison scripts`

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: sync project docs with new experiments directory"
```
