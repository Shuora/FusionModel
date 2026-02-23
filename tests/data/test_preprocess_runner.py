import socket
from pathlib import Path

import dpkt

from src.data.preprocess_runner import run_preprocess_policies


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
    with path.open("wb") as f:
        writer = dpkt.pcap.Writer(f)
        writer.writepkt(tls_pkt1, ts=1.0)
        writer.writepkt(tls_pkt2, ts=2.0)
        writer.close()


def test_run_preprocess_policies_outputs_strict_and_full(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "sample.pcap")
    output_root = tmp_path / "outputs" / "processed"

    results = run_preprocess_policies(
        source_root=source_root,
        output_root=output_root,
        policies=["strict", "full"],
        seed=42,
        show_progress=False,
    )

    assert set(results.keys()) == {"strict", "full"}
    assert (output_root / "DemoSet" / "strict" / "rgb" / "rgb_shard_00000.npz").exists()
    assert (output_root / "DemoSet" / "full" / "rgb" / "rgb_shard_00000.npz").exists()


def test_run_preprocess_policies_with_dataset_filter(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir_a = source_root / "DataA" / "FamilyA"
    pcap_dir_b = source_root / "DataB" / "FamilyB"
    pcap_dir_a.mkdir(parents=True)
    pcap_dir_b.mkdir(parents=True)
    _write_demo_pcap(pcap_dir_a / "a.pcap")
    _write_demo_pcap(pcap_dir_b / "b.pcap")
    output_root = tmp_path / "outputs" / "processed"

    results = run_preprocess_policies(
        source_root=source_root,
        output_root=output_root,
        policies=["strict"],
        datasets=["DataB"],
        seed=42,
        show_progress=False,
    )

    assert set(results.keys()) == {"strict"}
    assert results["strict"]["total_pcaps"] == 1
    assert (output_root / "DataB" / "strict" / "rgb" / "rgb_shard_00000.npz").exists()
    assert not (output_root / "DataA" / "strict" / "rgb" / "rgb_shard_00000.npz").exists()


def test_run_preprocess_policies_supports_session_full(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "sample.pcap")
    output_root = tmp_path / "outputs" / "processed"

    results = run_preprocess_policies(
        source_root=source_root,
        output_root=output_root,
        policies=["session_full"],
        seed=42,
        show_progress=False,
    )

    assert set(results.keys()) == {"session_full"}


def test_run_preprocess_policies_passes_cleanup_and_preview_flags(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "sample.pcap")
    output_root = tmp_path / "outputs" / "processed"

    results = run_preprocess_policies(
        source_root=source_root,
        output_root=output_root,
        policies=["session_full"],
        seed=42,
        cleanup_sessions=True,
        preview_per_family=5,
        show_progress=False,
    )

    assert results["session_full"]["tmp_session_pcaps_removed"] >= 1
    preview_dir = output_root / "DemoSet" / "session_full" / "debug" / "preview_png"
    assert preview_dir.exists()
    assert any(preview_dir.rglob("*.png"))
    assert not (output_root / "DemoSet" / "session_full" / "tmp_sessions").exists()
