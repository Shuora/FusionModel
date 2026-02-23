import socket
from pathlib import Path

import dpkt

from src.data.preprocess import preprocess_source


def _tls_record(content_type: int, payload: bytes, version: int = 0x0303) -> bytes:
    length = len(payload).to_bytes(2, "big")
    return bytes([content_type]) + version.to_bytes(2, "big") + length + payload


def _make_eth_tcp(src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes) -> bytes:
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, seq=1, flags=dpkt.tcp.TH_ACK, data=payload)
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src_ip),
        dst=socket.inet_aton(dst_ip),
        p=dpkt.ip.IP_PROTO_TCP,
        ttl=64,
        data=tcp,
    )
    ip.len = 20 + len(tcp)
    eth = dpkt.ethernet.Ethernet(
        src=b"\xaa\xaa\xaa\xaa\xaa\xaa",
        dst=b"\xbb\xbb\xbb\xbb\xbb\xbb",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def _write_demo_pcap(path: Path) -> None:
    tls_pkt1 = _make_eth_tcp(
        "10.0.0.1",
        "10.0.0.2",
        12345,
        443,
        _tls_record(22, bytes([1]) + b"\x00" * 8),
    )
    tls_pkt2 = _make_eth_tcp(
        "10.0.0.2",
        "10.0.0.1",
        443,
        12345,
        _tls_record(23, b"abcd"),
    )
    http_pkt = _make_eth_tcp(
        "10.0.0.3",
        "10.0.0.4",
        23456,
        80,
        b"GET / HTTP/1.1\r\nHost: test\r\n\r\n",
    )
    with path.open("wb") as f:
        writer = dpkt.pcap.Writer(f)
        writer.writepkt(tls_pkt1, ts=1.0)
        writer.writepkt(tls_pkt2, ts=2.0)
        writer.writepkt(http_pkt, ts=3.0)
        writer.close()


def test_session_full_cleanup_and_preview_png(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "sample.pcap")
    output_root = tmp_path / "outputs" / "processed"

    summary = preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="session_full",
        filter_mode="session_full",
        cleanup_sessions=True,
        preview_per_family=5,
        show_progress=False,
    )

    assert summary["tmp_session_pcaps_removed"] >= 1
    assert not (output_root / "DemoSet" / "session_full" / "tmp_sessions").exists()
    preview_dir = output_root / "DemoSet" / "session_full" / "debug" / "preview_png"
    assert preview_dir.exists()
    assert any(preview_dir.rglob("*.png"))

