from pathlib import Path

from src.data.build_dataset import (
    build_output_paths,
    make_manifest_row,
    split_tls_and_non_tls,
)


def _tls_record(content_type: int, payload: bytes, version: int = 0x0303) -> bytes:
    length = len(payload).to_bytes(2, "big")
    return bytes([content_type]) + version.to_bytes(2, "big") + length + payload


def test_build_output_paths_follow_plan_layout():
    paths = build_output_paths("outputs/processed", "USTC-TFC2016", "strict")

    assert paths["rgb_shard"] == Path(
        "outputs/processed/USTC-TFC2016/strict/rgb/rgb_shard_00000.npz"
    )
    assert paths["seq_shard"] == Path(
        "outputs/processed/USTC-TFC2016/strict/seq/seq_shard_00000.npz"
    )
    assert paths["manifest"] == Path(
        "outputs/processed/USTC-TFC2016/strict/manifest/session_manifest.parquet"
    )
    assert paths["tls_manifest"] == Path(
        "outputs/processed/USTC-TFC2016/strict/manifest/tls_sessions.parquet"
    )
    assert paths["non_tls_manifest"] == Path(
        "outputs/processed/USTC-TFC2016/strict/manifest/non_tls_dropped.parquet"
    )


def test_make_manifest_row_contains_required_fields():
    row = make_manifest_row(
        session_id="s1",
        dataset="USTC-TFC2016",
        family="Virut",
        capture_id="Virut.pcap",
        split="train",
        policy="strict",
    )
    assert row["session_id"] == "s1"
    assert row["dataset"] == "USTC-TFC2016"
    assert row["family"] == "Virut"
    assert row["capture_id"] == "Virut.pcap"
    assert row["split"] == "train"
    assert row["policy"] == "strict"


def test_split_tls_and_non_tls_outputs_drop_reason():
    tls_session = {
        "session_id": "tls-1",
        "protocol": "TCP",
        "payload_chunks": [
            _tls_record(22, bytes([1]) + b"\x00" * 5),
            _tls_record(23, b"abcd"),
        ],
    }
    non_tls_session = {
        "session_id": "non-tls-1",
        "protocol": "TCP",
        "payload_chunks": [b"GET / HTTP/1.1\r\nHost: test\r\n\r\n"],
    }

    accepted, dropped = split_tls_and_non_tls(
        [tls_session, non_tls_session], mode="strict"
    )

    assert [s["session_id"] for s in accepted] == ["tls-1"]
    assert dropped[0]["session_id"] == "non-tls-1"
    assert dropped[0]["drop_reason"] == "cleartext_signature"
