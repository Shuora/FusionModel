import socket
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

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

    rgb_files = sorted((output_root / "DemoSet" / "strict" / "rgb").glob("rgb_shard_*.npz"))
    seq_files = sorted((output_root / "DemoSet" / "strict" / "etbert").glob("etbert_shard_*.npz"))
    assert len(rgb_files) == 1
    assert len(seq_files) == 1

    rgb_npz = np.load(rgb_files[0], allow_pickle=False)
    seq_npz = np.load(seq_files[0], allow_pickle=False)
    assert rgb_npz["rgb"].shape == (1, 3, 28, 28)
    assert seq_npz["input_ids"].shape == (1, 256)


def test_preprocess_source_session_ids_are_unique_across_captures(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "a.pcap")
    _write_demo_pcap(pcap_dir / "b.pcap")

    output_root = tmp_path / "outputs" / "processed"
    summary = preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="strict",
        show_progress=False,
    )
    assert summary["accepted_sessions"] == 2

    manifest_csv = output_root / "DemoSet" / "strict" / "manifest" / "session_manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)
    assert len(manifest_df) == 2
    assert manifest_df["session_id"].nunique() == 2


def test_preprocess_source_split_mapping_uses_unique_capture_identity(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    f1_dir = source_root / "DemoSet" / "F1"
    f2_dir = source_root / "DemoSet" / "F2"
    f1_dir.mkdir(parents=True)
    f2_dir.mkdir(parents=True)

    # Intentionally reuse capture file name across families.
    _write_demo_pcap(f1_dir / "a.pcap")
    _write_demo_pcap(f1_dir / "b.pcap")
    _write_demo_pcap(f2_dir / "a.pcap")
    _write_demo_pcap(f2_dir / "c.pcap")

    output_root = tmp_path / "outputs" / "processed"
    preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="strict",
        seed=4,
        show_progress=False,
    )

    split_df = pd.read_csv(output_root / "manifest" / "split_manifest.csv")
    session_df = pd.read_csv(output_root / "DemoSet" / "strict" / "manifest" / "session_manifest.csv")

    expected = {
        (str(r["dataset"]), str(r["family"]), str(r["capture_id"])): str(r["split"])
        for _, r in split_df.iterrows()
    }
    observed = (
        session_df.groupby(["dataset", "family", "capture_id"])["split"]
        .agg(lambda x: x.iloc[0])
        .to_dict()
    )
    for key, split in expected.items():
        assert observed[key] == split


def test_preprocess_source_normalizes_flat_family_names_in_manifest(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    source_root.mkdir(parents=True)
    (source_root / "MFCP").mkdir(parents=True, exist_ok=True)
    _write_demo_pcap(source_root / "MFCP" / "PUA-1.pcap")

    output_root = tmp_path / "outputs" / "processed"
    preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="strict",
        show_progress=False,
    )

    manifest_csv = output_root / "MFCP" / "strict" / "manifest" / "session_manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)
    assert set(manifest_df["family"]) == {"PUA"}


def test_preprocess_source_session_full_splits_single_capture_family_by_session(tmp_path: Path):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "USTC-TFC2016"
    pcap_dir.mkdir(parents=True)
    pcap_path = pcap_dir / "Virut.pcap"
    _write_demo_pcap(pcap_path)

    output_root = tmp_path / "outputs" / "processed"
    preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="session_full",
        filter_mode="session_full",
        seed=7,
        show_progress=False,
    )

    manifest_csv = output_root / "USTC-TFC2016" / "session_full" / "manifest" / "session_manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)
    assert set(manifest_df["capture_id"]) == {"Virut.pcap"}
    assert set(manifest_df["split"]) == {"train", "test"}

    split_manifest_csv = output_root / "manifest" / "split_manifest.csv"
    split_df = pd.read_csv(split_manifest_csv)
    assert set(split_df["capture_id"]) == {"Virut.pcap"}
    assert "session_id" in set(split_df.columns)
    assert set(split_df["split"]) == {"train", "test"}
