import socket
from pathlib import Path

import dpkt
import numpy as np

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


def test_preprocess_source_writes_expected_outputs(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    pcap_path = pcap_dir / "sample.pcap"
    _write_demo_pcap(pcap_path)

    output_root = tmp_path / "outputs" / "processed"
    logs = []
    summary = preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="strict",
        show_progress=False,
        log_fn=logs.append,
    )

    assert summary["total_pcaps"] == 1
    assert summary["accepted_sessions"] == 1
    assert summary["dropped_sessions"] == 1
    assert any("✅SUCCESS" in line for line in logs)
    assert any("🧱 Data" in line for line in logs)

    split_dir = output_root / "manifest"
    assert (split_dir / "split_manifest.csv").exists()
    assert (split_dir / "leakage_report.json").exists()

    manifest_dir = output_root / "DemoSet" / "strict" / "manifest"
    assert (manifest_dir / "session_manifest.csv").exists()
    assert (manifest_dir / "tls_sessions.csv").exists()
    assert (manifest_dir / "non_tls_dropped.csv").exists()

    rgb_path = output_root / "DemoSet" / "strict" / "rgb" / "rgb_shard_00000.npz"
    seq_path = output_root / "DemoSet" / "strict" / "seq" / "seq_shard_00000.npz"
    assert rgb_path.exists()
    assert seq_path.exists()

    rgb_npz = np.load(rgb_path, allow_pickle=False)
    seq_npz = np.load(seq_path, allow_pickle=False)
    assert rgb_npz["rgb"].shape == (1, 3, 28, 28)
    assert seq_npz["token_ids"].shape == (1, 256)
