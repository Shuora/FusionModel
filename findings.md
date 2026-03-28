# Findings

- 根因确认：`vpn_hangouts_audio2.pcap` 文件头为 `0A0D0D0A`，实际是 `pcapng`，SplitCap 仅接受经典 pcap。
- 现状限制：`prepare_splitcap_input` 仅按后缀 `.pcapng` 转码，无法处理误标后缀文件。
- 现状限制：`collect_session_paths` 每次都全量调用 SplitCap，缺少 checkpoint 与跳过逻辑。
