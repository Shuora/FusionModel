## Findings

- `src/split_data.py` 当前的 `iter_packets()` 直接把 `dpkt.pcap.Reader` 的异常向上抛出，导致单个尾部不完整的 `.pcap` 会被整文件跳过。
- 对用户提供的 `Cobalt.pcap` 手工解析后确认：文件主体可连续读出 `1471709` 个 packet，最后在偏移 `149958654` 处只剩 `2` 字节，不足一个 `16` 字节 packet header。
- 当前需求不是忽略所有坏包，而是把“尾部残缺 EOF”视为可恢复情况，保留已成功读取的数据包。
- 最小修复边界是：`.pcap` 改用顺序字节解析并在 EOF 尾部残缺时记录 warning 后停止；`.pcapng` 继续依赖 `dpkt.pcapng.Reader`，不改变现有行为。
- 为避免格式兼容面缩窄，`.pcap` 顺序解析需要同时支持 microsecond 与 nanosecond 两类常见 magic bytes。
