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

## Preprocess Planning Logs

- `Ctrl+C` 栈明确落在 `prepare_cached_rows(...) -> read_session_bytes(...)`，说明长时间无输出发生在父进程串行去重阶段，而不是 worker 池。
- 全量二分类会在 `sessions_raw` 上先做一次完整的 payload 扫描和 fingerprint 判重；这一阶段在进入 `[cache] planned` 前原本没有任何 heartbeat。
- 最小且安全的修复是给规划阶段增加周期性 `[plan]` 日志，保留现有去重顺序和判定逻辑，不把去重下放到 worker。

## Preprocess Planning Performance

- 当前性能瓶颈来自 `prepare_cached_rows(...)` 在父进程串行调用 `read_session_bytes(...)`。
- payload 提取与 fingerprint 计算可以并行，因为每个 session 的检查彼此独立；真正需要顺序的是“首次出现保留、后续重复丢弃”的决策。
- 使用 `ProcessPoolExecutor.map(...)` 按输入顺序消费 worker 结果，可以在并行读取的同时保持去重结果与串行实现一致。

- 规划阶段现在也会复用 `--num-workers`，将 `read_session_bytes(...)` + fingerprint 计算分发给多个进程。
- 父进程仍按 manifest 顺序消费 `executor.map(...)` 结果，因此重复样本的“首个保留”语义保持不变。

## Session Byte Extraction

- 根因确认：`read_session_bytes(...)` 之前只拼接 `packet.getlayer(Raw)` 的 `load`，会把“传输层 payload 存在，但 Scapy 未建成 `Raw` 层”的 session 误判为空。
- 这类误判可以用 `TCP/UDP + Padding` 稳定复现：`Raw` 为 `None`，但 `bytes(packet[TCP].payload)` 或 `bytes(packet[UDP].payload)` 仍有有效字节。
- 修复后，session 字节提取逻辑改为优先读取 TCP/UDP payload bytes，必要时回退到 IP/IPv6 更底层 payload，最后再回退 `Raw`；真正的 empty session 定义为“最终拼接出的有效字节串长度为 0”。
- 这次修改保持了下游 `normalize_session_bytes(..., size=784)`、图像构造、tokenize、去重与缓存流程不变，因此语义变化只集中在“哪些 session 被视为非空并进入后续处理”。
