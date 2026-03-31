## Task
将训练日志和各种训练输出的默认目录从 `src/outputs` 改为仓库根目录下的 `outputs`。

## Plan
1. 在隔离 worktree 中检查默认输出目录与日志目录的实现位置，以及 README/AGENTS 的同步要求。
2. 先补一个默认路径回归测试，确认当前默认值仍指向 `src/outputs`。
3. 仅修改训练相关默认输出根目录与默认日志目录，使其统一落到仓库根目录 `outputs`。
4. 同步更新 `README.md` 中四个实验命令和输出目录说明，必要时修正 `AGENTS.md`。
5. 运行相关 `unittest` 验证改动，并同步更新 `findings.md` 与 `progress.md`。

## Constraints
- 不运行 `mvn test`。
- 不迁移已有历史输出产物，只修改默认路径与文档。
- 先写失败测试，再写生产代码。
