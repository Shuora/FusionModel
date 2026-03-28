# Findings

## SplitCap Resume

- 根因确认：`vpn_hangouts_audio2.pcap` 文件头为 `0A0D0D0A`，实际是 `pcapng`，SplitCap 仅接受经典 pcap。
- 兼容性修复：新增按文件头魔数识别 `pcapng`，即使扩展名是 `.pcap` 也会先用 `editcap -F pcap` 转码。
- 续跑能力：为每个 raw capture 的输出目录写入 `.splitcap.done`，重跑时可跳过已完成样本。
- 容错行为：单个文件 SplitCap 失败不再让整个任务中断，会输出告警并继续后续文件。
- 脚本入口：`run_prepare_binary.sh` 支持 `SPLITCAP_LAUNCHER`、`EDITCAP_BIN`，并默认在 `USE_SPLITCAP=1` 时开启 resume（`RESUME_SPLITCAP=0` 可关闭）。

## Preprocess Workers

- 新增 `--include-path`，支持按数据集根目录或更细子目录筛选原始 capture。
- 后处理续跑按产物存在判断：`sessions_clean/<sample_id>.pcap` 和 `cache/<sample_id>.npz`。
- 父进程保留样本顺序和去重判定，worker 只负责清洗、tokenize 和写缓存，避免并行导致去重结果漂移。
- 新增 `--num-workers` 和 `--progress-every`，默认值偏保守，优先兼顾吞吐和内存占用。
- 二分类与多分类入口脚本都支持 `INCLUDE_PATHS`、`NUM_WORKERS`、`PROGRESS_EVERY`。
