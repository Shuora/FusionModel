import socket
from pathlib import Path

import dpkt

from src.data.session_splitcap import cleanup_session_pcaps, split_pcap_to_session_pcaps


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


def _write_two_session_pcap(path: Path) -> None:
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


def _write_two_session_pcapng(path: Path) -> None:
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
        writer = dpkt.pcapng.Writer(f)
        writer.writepkt(tls_pkt1, ts=1.0)
        writer.writepkt(tls_pkt2, ts=2.0)
        writer.writepkt(http_pkt, ts=3.0)
        writer.close()


def test_split_pcap_to_sessions_and_cleanup(tmp_path: Path):
    pcap_path = tmp_path / "input.pcap"
    tmp_dir = tmp_path / "tmp_sessions"
    _write_two_session_pcap(pcap_path)

    session_files = split_pcap_to_session_pcaps(pcap_path, tmp_dir)
    assert len(session_files) == 2
    assert all(p.exists() for p in session_files)
    assert all(p.suffix == ".pcap" for p in session_files)

    removed = cleanup_session_pcaps(session_files)
    assert removed == 2
    assert not any(p.exists() for p in session_files)


def test_split_pcapng_to_sessions(tmp_path: Path):
    pcapng_path = tmp_path / "input.pcap"
    tmp_dir = tmp_path / "tmp_sessions"
    _write_two_session_pcapng(pcapng_path)

    session_files = split_pcap_to_session_pcaps(pcapng_path, tmp_dir)
    assert len(session_files) == 2
    assert all(p.exists() for p in session_files)


def test_split_pcap_tolerates_trailing_truncated_record(tmp_path: Path):
    pcap_path = tmp_path / "truncated_tail.pcap"
    tmp_dir = tmp_path / "tmp_sessions"
    _write_two_session_pcap(pcap_path)

    # Simulate a real-world damaged capture with trailing incomplete bytes.
    with pcap_path.open("ab") as f:
        f.write(b"\x00\x01")

    session_files = split_pcap_to_session_pcaps(pcap_path, tmp_dir)
    assert len(session_files) == 2
    assert all(p.exists() for p in session_files)
