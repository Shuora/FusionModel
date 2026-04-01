## Findings

- `src/fusion_common.py` 里有两处默认路径原本仍指向 `src/outputs`：`setup_logging()` 默认日志目录、`add_common_args()` 的 `--output_dir` 默认值。
- `src/run_all_modes.py` 只是透传 `fusion_common.py` 生成的 `output_dir`，本次无需单独修改。
- `README.md` 的四个实验命令和“训练输出”章节原本写成 `src/outputs/<task_name>/...`，已同步改为根目录 `outputs/<task_name>/...`。
- `tests/test_fusion_output_artifacts.py` 原先不覆盖“默认路径在仓库根目录”的行为，本次新增了默认 `output_dir` 与默认日志目录的回归测试。
- `AGENTS.md` 原先存在未解决的 merge conflict 标记；本次已合并有效约束并补充默认输出目录说明。
- 本机测试环境存在 `PYTHONPATH=/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages` 污染；运行 conda 环境下的验证命令时，需要显式 `unset PYTHONPATH PYTHONHOME PYTHONUSERBASE` 并设置 `PYTHONNOUSERSITE=1`。


- 2026-03-31 MFCP: SourceData has 6 families/7 pcaps; processed mfcp currently misses Cobalt due old pre-fix truncation handling and should be regenerated with current split_data.

- 2026-03-31 EarlyStopping: CLI 默认 `--patience=8`，但 `EarlyStopping.__init__` 与 `train_fusion_model` 内部默认仍是 7，存在默认值不一致。
- 2026-03-31 EarlyStopping: 旧实现允许 `early_stop_metric` 与手动 `early_stop_mode` 方向冲突（如 `val_f1 + min`），会造成静默错误早停；现改为显式报错。
- 2026-03-31 EarlyStopping: 旧实现在监控值为 NaN/Inf 时会累加 early-stop 计数并喂给 ReduceLROnPlateau；现改为跳过该轮并记录 warning。
- 2026-04-01 NaN早停: `src/fusion_common.py` 在 `monitor_is_finite=False` 时原先仅 warning 并跳过 early-stop 更新；这会让计数器停滞，导致 `patience` 失效并持续训练至 `num_epochs`。
- 2026-04-01 NaN训练防护: 训练循环此前对 batch-level 非有限 loss 缺少保护；现新增 finite-check，NaN/Inf batch 直接跳过，避免无效梯度污染参数。
