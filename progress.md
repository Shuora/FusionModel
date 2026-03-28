# Progress

## SplitCap Resume

- [x] 读取并确认错误根因与触发样本
- [x] 审阅 `prepare_dataset.py` / `run_prepare_binary.sh` / 相关测试
- [x] 新增断点续跑与误标后缀处理测试
- [x] 实现脚本改造
- [x] 跑测试验证
- [x] 输出使用说明

## Preprocess Workers

- [x] 设计确认：并行后处理、后处理续跑、进度日志、细粒度路径过滤
- [x] 创建隔离 worktree 并验证相关测试基线
- [x] 写规格与实现计划文档
- [x] 补失败测试
- [x] 实现脚本改造
- [x] 跑测试验证

## Verification

- `env -u PYTHONPATH -u PYTHONUSERBASE TMPDIR=/tmp /home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q --capture=no tests/test_splitcap_cleaning_and_manifest.py`
