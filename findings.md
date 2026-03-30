## Findings

<<<<<<< Updated upstream
- `src/split_data.py` 当前的 `iter_packets()` 直接把 `dpkt.pcap.Reader` 的异常向上抛出，导致单个尾部不完整的 `.pcap` 会被整文件跳过。
- 对用户提供的 `Cobalt.pcap` 手工解析后确认：文件主体可连续读出 `1471709` 个 packet，最后在偏移 `149958654` 处只剩 `2` 字节，不足一个 `16` 字节 packet header。
- 当前需求不是忽略所有坏包，而是把“尾部残缺 EOF”视为可恢复情况，保留已成功读取的数据包。
- 最小修复边界是：`.pcap` 改用顺序字节解析并在 EOF 尾部残缺时记录 warning 后停止；`.pcapng` 继续依赖 `dpkt.pcapng.Reader`，不改变现有行为。
=======
- 仓库当前训练主入口是 `src/train_fusion_attention.py` 和 `src/train_fusion_attention_stacking.py`，二者都要求显式传入 `--task_name`。
- 当前支持的四个实验任务名固定为 `binary_benign_vs_malicious`、`ustc_multiclass`、`mta_multiclass`、`mfcp_multiclass`。
- 数据预处理分为两步：`src/split_data.py` 先把原始抓包转成 `ProcessedData/<task>/pcap_data/...`，`src/ssl_tls_rgb_image.py` 再把 `.bin` 转成 `image_data/...`。
- `src/run_all_modes.py` 提供合并运行入口，但用户明确要求 README 中不要把四个实验写成一个合并命令。
>>>>>>> Stashed changes
