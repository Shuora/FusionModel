## Findings

- `src/fusion_common.py` 里有两处默认路径原本仍指向 `src/outputs`：`setup_logging()` 默认日志目录、`add_common_args()` 的 `--output_dir` 默认值。
- `src/run_all_modes.py` 只是透传 `fusion_common.py` 生成的 `output_dir`，本次无需单独修改。
- `README.md` 的四个实验命令和“训练输出”章节原本写成 `src/outputs/<task_name>/...`，已同步改为根目录 `outputs/<task_name>/...`。
- `tests/test_fusion_output_artifacts.py` 原先不覆盖“默认路径在仓库根目录”的行为，本次新增了默认 `output_dir` 与默认日志目录的回归测试。
- `AGENTS.md` 原先存在未解决的 merge conflict 标记；本次已合并有效约束并补充默认输出目录说明。
- 本机测试环境存在 `PYTHONPATH=/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages` 污染；运行 conda 环境下的验证命令时，需要显式 `unset PYTHONPATH PYTHONHOME PYTHONUSERBASE` 并设置 `PYTHONNOUSERSITE=1`。
