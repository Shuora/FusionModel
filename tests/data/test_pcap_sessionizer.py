import socket
from pathlib import Path

import dpkt

from src.data.pcap_sessionizer import classify_pcap_sessions, read_tcp_sessions


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


def _make_eth_udp(src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes) -> bytes:
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = 8 + len(payload)
    ip = dpkt.ip.IP(
        src=socket.inet_aton(src_ip),
        dst=socket.inet_aton(dst_ip),
        p=dpkt.ip.IP_PROTO_UDP,
        ttl=64,
        data=udp,
    )
    ip.len = 20 + len(udp)
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
    udp_pkt = _make_eth_udp("10.0.0.5", "10.0.0.6", 9999, 53, b"\x12\x34")

    with path.open("wb") as f:
        writer = dpkt.pcap.Writer(f)
        writer.writepkt(tls_pkt1, ts=1.0)
        writer.writepkt(tls_pkt2, ts=2.0)
        writer.writepkt(http_pkt, ts=3.0)
        writer.writepkt(udp_pkt, ts=4.0)
        writer.close()


def test_read_tcp_sessions_aggregates_bidirectional_and_ignores_udp(tmp_path: Path):
    pcap_path = tmp_path / "demo.pcap"
    _write_demo_pcap(pcap_path)

    sessions = read_tcp_sessions(pcap_path)

    assert len(sessions) == 2
    sizes = sorted(len(s["payload_chunks"]) for s in sessions)
    assert sizes == [1, 2]
    assert all(s["protocol"] == "TCP" for s in sessions)


def test_classify_pcap_sessions_returns_tls_and_non_tls(tmp_path: Path):
    pcap_path = tmp_path / "demo.pcap"
    _write_demo_pcap(pcap_path)

    accepted, dropped = classify_pcap_sessions(pcap_path, mode="strict")

    assert len(accepted) == 1
    assert len(dropped) == 1
    assert dropped[0]["drop_reason"] == "cleartext_signature"
