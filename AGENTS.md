# AGENTS.md

<<<<<<< HEAD
本文件约束在 `/home/shuora/Traffic/FusionModel` 内工作的 AI agent。

## Communication

- 默认使用中文沟通。
- 英文只用于 technical terms、code、command、path、identifier。

## Working Rules

- 开始复杂任务前，先检查并维护 `task_plan.md`、`findings.md`、`progress.md`。
- 对于复杂任务，优先拆分子任务并并行处理。
- 需要修改代码时，优先遵守项目既有 git/worktree 规范。
- 不要运行 `mvn test`。
- 如果在 WSL 中执行命令，使用原生 Linux 命令，不要改用 PowerShell 或 `pwsh`。

## Mandatory Documentation Sync

只要本仓库发生以下任一变化，AI 在同一任务中必须同步更新 `README.md` 和必要时更新本 `AGENTS.md`：

- 新增、删除或重命名训练脚本
- 新增、删除或重命名预处理脚本
- 新增、删除或重命名实验任务
- 修改训练命令、参数、默认值或输出目录
- 修改数据目录结构、数据流、预处理步骤或训练流程
- 修改依赖安装方式、运行方式或验证方式
- 任何会导致 README 中命令失效、过期或歧义的变更

## README Update Requirements

当 AI 修改项目后，如果变更影响运行方式，必须检查 `README.md` 中以下内容是否仍然准确：

- 项目结构
- 环境准备
- 数据目录要求
- 预处理步骤
- 四个实验任务的独立命令
- attention 训练命令
- attention_stacking 训练命令
- 输出目录
- 自检命令

禁止只更新代码而不更新对应命令文档。

## AGENTS Update Requirements

当 AI 修改了协作流程、开发约束、文档维护规则、目录约定、测试约束或 agent 行为要求时，必须同步更新 `AGENTS.md`。

如果 AI 发现 `README.md` 或 `AGENTS.md` 与仓库当前实现不一致，应优先修正文档，再结束任务。

## Command Writing Rules

- README 中涉及实验运行时，必须给出从仓库根目录可直接执行的命令。
- 涉及四个实验任务时，必须分别列出独立命令，不能只提供一个合并命令替代。
- 命令里的路径、脚本名、参数名必须与仓库当前实现一致。
- 如果默认参数容易误导，README 中必须显式写出推荐参数。

## Completion Checklist

任务完成前，AI 必须自行检查：

1. `README.md` 中的命令是否与当前脚本参数一致。
2. 四个实验是否都给出了各自独立命令。
3. `AGENTS.md` 是否反映了最新协作要求。
4. `task_plan.md`、`findings.md`、`progress.md` 是否已同步更新。

未完成以上检查，不应宣称任务完成。
=======
## Communication

- Prefer Chinese for communication.
- English may be used for technical terms, code, commands, and identifiers.

## Environment

- Before running project commands, activate the conda environment:

```bash
conda activate FusionModel
```

- When writing documentation or command examples for this repository, use `python3` in commands.
- Run commands from the repository root: `/home/shuora/Traffic/FusionModel`.

## Documentation Sync

- If you modify README, usage instructions, environment instructions, workflow documents, or other repository-facing guidance, you must also review and update `AGENTS.md` when needed so agent instructions stay consistent with the docs.
- If you modify project behavior, command flow, environment assumptions, output paths, or task conventions, sync those changes to both `README.md` and `AGENTS.md`.

## Project Notes

- This repository currently documents and runs the V4 `MobileViT + CharBERT` attention fusion workflow.
- Supported `task_name` values are:
  - `binary_benign_vs_malicious`
  - `ustc_multiclass`
  - `mta_multiclass`
  - `mfcp_multiclass`
- Standard data flow:
  - `SourceData/<dataset>`
  - `ProcessedData/<task>/pcap_data/{Train,Test}`
  - `ProcessedData/<task>/image_data/{Train,Test}`
  - training via `src/train_fusion_attention.py`, `src/train_fusion_attention_stacking.py`, or `src/run_all_modes.py`
- Default training runtime parameters should stay aligned across code and docs:
  - `batch_size=32`
  - `num_workers=4`
  - `prefetch_factor=2`

## Constraints

- Do not run `mvn test`.
- If historical helper scripts under `tools/` contain hardcoded paths, do not present them as standard cross-platform commands without updating them first.
>>>>>>> b23c1ae5f224cc4b17dd1dd42703f3536fdf21d9
