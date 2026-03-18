import socket
from pathlib import Path

import dpkt
import pandas as pd
import pytest

import src.data.preprocess as preprocess_mod
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


def _write_demo_pcap(path: Path, sport: int) -> None:
    tls_pkt1 = _make_eth_tcp(
        "10.0.0.1",
        "10.0.0.2",
        sport,
        443,
        _tls_record(22, bytes([1]) + b"\x00" * 8),
    )
    tls_pkt2 = _make_eth_tcp(
        "10.0.0.2",
        "10.0.0.1",
        443,
        sport,
        _tls_record(23, b"abcd"),
    )
    with path.open("wb") as f:
        writer = dpkt.pcap.Writer(f)
        writer.writepkt(tls_pkt1, ts=1.0)
        writer.writepkt(tls_pkt2, ts=2.0)
        writer.close()


def test_preprocess_source_resume_after_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "a.pcap", sport=12345)
    _write_demo_pcap(pcap_dir / "b.pcap", sport=22345)
    output_root = tmp_path / "outputs" / "processed"

    real_classify = preprocess_mod.classify_pcap_sessions
    calls = {"n": 0}

    def flaky_classify(pcap_path, mode: str = "strict"):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_classify(pcap_path, mode=mode)

    monkeypatch.setattr(preprocess_mod, "classify_pcap_sessions", flaky_classify)
    with pytest.raises(RuntimeError, match="simulated crash"):
        preprocess_source(
            source_root=source_root,
            output_root=output_root,
            policy="session_full",
            filter_mode="session_full",
            show_progress=False,
            resume=True,
        )

    checkpoint_dir = output_root / "DemoSet" / "session_full" / "checkpoints" / "preprocess"
    assert len(list(checkpoint_dir.glob("*.done.json"))) == 1

    monkeypatch.setattr(preprocess_mod, "classify_pcap_sessions", real_classify)
    summary = preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="session_full",
        filter_mode="session_full",
        show_progress=False,
        resume=True,
    )

    assert summary["total_pcaps"] == 2
    assert summary["accepted_sessions"] == 2
    assert summary["dropped_sessions"] == 0

    manifest_csv = output_root / "DemoSet" / "session_full" / "manifest" / "session_manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)
    assert len(manifest_df) == 2
    assert set(manifest_df["capture_id"]) == {"a.pcap", "b.pcap"}

    rgb_files = sorted((output_root / "DemoSet" / "session_full" / "rgb").glob("rgb_shard_*.npz"))
    seq_files = sorted((output_root / "DemoSet" / "session_full" / "etbert").glob("etbert_shard_*.npz"))
    assert len(rgb_files) == 2
    assert len(seq_files) == 2

    summary_again = preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="session_full",
        filter_mode="session_full",
        show_progress=False,
        resume=True,
    )
    assert summary_again["accepted_sessions"] == 2
    assert summary_again["dropped_sessions"] == 0
    manifest_df_again = pd.read_csv(manifest_csv)
    assert len(manifest_df_again) == 2


def test_preprocess_source_resume_survives_capture_index_shift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / "SourceData"
    pcap_dir = source_root / "DemoSet" / "FamilyA"
    pcap_dir.mkdir(parents=True)
    _write_demo_pcap(pcap_dir / "a.pcap", sport=12345)
    _write_demo_pcap(pcap_dir / "b.pcap", sport=22345)
    output_root = tmp_path / "outputs" / "processed"

    real_classify = preprocess_mod.classify_pcap_sessions
    calls = {"n": 0}

    def flaky_classify(pcap_path, mode: str = "strict"):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_classify(pcap_path, mode=mode)

    monkeypatch.setattr(preprocess_mod, "classify_pcap_sessions", flaky_classify)
    with pytest.raises(RuntimeError, match="simulated crash"):
        preprocess_source(
            source_root=source_root,
            output_root=output_root,
            policy="session_full",
            filter_mode="session_full",
            show_progress=False,
            resume=True,
        )

    _write_demo_pcap(pcap_dir / "0.pcap", sport=32345)
    monkeypatch.setattr(preprocess_mod, "classify_pcap_sessions", real_classify)

    summary = preprocess_source(
        source_root=source_root,
        output_root=output_root,
        policy="session_full",
        filter_mode="session_full",
        show_progress=False,
        resume=True,
    )

    assert summary["total_pcaps"] == 3
    assert summary["accepted_sessions"] == 3
    assert summary["dropped_sessions"] == 0

    manifest_csv = output_root / "DemoSet" / "session_full" / "manifest" / "session_manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)
    assert len(manifest_df) == 3
    assert list(sorted(manifest_df["capture_id"].tolist())) == ["0.pcap", "a.pcap", "b.pcap"]

    rgb_files = sorted((output_root / "DemoSet" / "session_full" / "rgb").glob("rgb_shard_*.npz"))
    seq_files = sorted((output_root / "DemoSet" / "session_full" / "etbert").glob("etbert_shard_*.npz"))
    assert len(rgb_files) == 3
    assert len(seq_files) == 3
