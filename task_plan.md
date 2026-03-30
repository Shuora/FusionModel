## Task
<<<<<<< Updated upstream
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
=======
补充项目级 README 和 AGENTS 文档，完整覆盖数据预处理、四个实验任务的独立训练命令，以及后续 AI 改动时的文档同步要求。

## Plan
1. 读取训练、预处理和任务配置入口，确认实际可执行命令与参数来源。
2. 按项目目录结构编写 README，覆盖环境、目录、数据流、预处理步骤、四个任务各自的 attention 与 attention_stacking 命令。
3. 编写 AGENTS.md，约束 AI 在修改代码、脚本、数据流程或命令时同步更新 README 与 AGENTS.md。
4. 做一次基础校验，确认文档中的脚本名、参数名、任务名与仓库实现一致。

## Constraints
- 不运行 `mvn test`。
- 本次以文档更新为主，不改训练逻辑。
- README 中四个任务必须分别给出独立命令，不能只给合并入口。
>>>>>>> Stashed changes
