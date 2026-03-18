import json
from pathlib import Path

from src.data.dataset_inventory import (
    detect_capture_leakage,
    detect_session_leakage,
    scan_source_pcaps,
    split_by_capture,
    split_by_session,
    write_split_artifacts,
)


def test_scan_source_pcaps_extracts_dataset_and_family(tmp_path: Path):
    (tmp_path / "USTC-TFC2016").mkdir(parents=True)
    (tmp_path / "USTC-TFC2016" / "Virut.pcap").write_bytes(b"pcap")
    (tmp_path / "MFCP" / "Artemis").mkdir(parents=True)
    (tmp_path / "MFCP" / "Artemis" / "Artemis.pcap").write_bytes(b"pcap")
    (tmp_path / "MFCP" / "Artemis" / "Artemis.pcap:Zone.Identifier").write_text("x")

    records = scan_source_pcaps(tmp_path)

    assert len(records) == 2
    assert records[0]["dataset"] == "MFCP"
    assert records[0]["family"] == "Artemis"
    assert records[1]["dataset"] == "USTC-TFC2016"
    assert records[1]["family"] == "Virut"


def test_scan_source_pcaps_supports_pcapng_and_normalizes_flat_family_names(tmp_path: Path):
    (tmp_path / "ISCX-VPN-NonVPN-2016" / "NonVPN-PCAPs-01").mkdir(parents=True)
    (tmp_path / "ISCX-VPN-NonVPN-2016" / "NonVPN-PCAPs-01" / "facebook_audio1a.pcapng").write_bytes(b"pcapng")
    (tmp_path / "MFCP").mkdir(parents=True, exist_ok=True)
    (tmp_path / "MFCP" / "PUA-1.pcap").write_bytes(b"pcap")
    (tmp_path / "MTA").mkdir(parents=True, exist_ok=True)
    (tmp_path / "MTA" / "IcedID_2.pcap").write_bytes(b"pcap")

    records = scan_source_pcaps(tmp_path)

    observed = {(r["dataset"], r["family"], r["capture_id"]) for r in records}
    assert ("ISCX-VPN-NonVPN-2016", "NonVPN-PCAPs-01", "facebook_audio1a.pcapng") in observed
    assert ("MFCP", "PUA", "PUA-1.pcap") in observed
    assert ("MTA", "IcedID", "IcedID_2.pcap") in observed


def test_split_by_capture_ensures_one_capture_in_one_split_only():
    records = [
        {"dataset": "D1", "family": "F1", "capture_id": f"cap_{i}.pcap", "pcap_path": f"/x/{i}.pcap"}
        for i in range(6)
    ]
    split_rows = split_by_capture(records, seed=7)

    capture_to_splits = {}
    for row in split_rows:
        capture_to_splits.setdefault(row["capture_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in capture_to_splits.values())
    assert {r["split"] for r in split_rows}.issubset({"train", "val", "test"})


def test_split_by_capture_ensures_dataset_level_val_and_test_when_possible():
    records = [
        {"dataset": "D1", "family": "F1", "capture_id": f"cap_{i}.pcap", "pcap_path": f"/x/{i}.pcap"}
        for i in range(5)
    ]
    split_rows = split_by_capture(records, seed=11)
    splits = [r["split"] for r in split_rows]
    assert sorted(splits) == ["test", "train", "train", "train", "train"]


def test_split_by_capture_two_captures_dataset_has_train_and_test():
    records = [
        {"dataset": "D1", "family": "F1", "capture_id": "cap_a.pcap", "pcap_path": "/x/a.pcap"},
        {"dataset": "D1", "family": "F1", "capture_id": "cap_b.pcap", "pcap_path": "/x/b.pcap"},
    ]
    split_rows = split_by_capture(records, seed=13)
    splits = sorted(r["split"] for r in split_rows)
    assert splits == ["test", "train"]


def test_split_by_capture_no_val_split_in_paper_style_default():
    records = [
        {"dataset": "D1", "family": "F1", "capture_id": f"cap_{i}.pcap", "pcap_path": f"/x/{i}.pcap"}
        for i in range(10)
    ]
    split_rows = split_by_capture(records, seed=1)
    assert "val" not in {r["split"] for r in split_rows}


def test_split_by_session_creates_train_and_test_within_single_capture_family():
    records = [
        {
            "dataset": "USTC-TFC2016",
            "family": "Virut",
            "capture_id": "Virut.pcap",
            "session_id": f"sid_{i}",
        }
        for i in range(5)
    ]

    split_rows = split_by_session(records, seed=11)

    assert sorted(r["split"] for r in split_rows) == ["test", "train", "train", "train", "train"]


def test_detect_capture_leakage_flags_conflicts():
    rows = [
        {"dataset": "D1", "capture_id": "c1.pcap", "split": "train"},
        {"dataset": "D1", "capture_id": "c1.pcap", "split": "test"},
    ]
    report = detect_capture_leakage(rows)
    assert report["has_leakage"] is True
    assert report["leaked_capture_count"] == 1


def test_detect_session_leakage_flags_conflicts():
    rows = [
        {"dataset": "D1", "session_id": "s1", "split": "train"},
        {"dataset": "D1", "session_id": "s1", "split": "test"},
    ]
    report = detect_session_leakage(rows)
    assert report["has_leakage"] is True
    assert report["leaked_session_count"] == 1


def test_write_split_artifacts_creates_manifest_and_report(tmp_path: Path):
    split_rows = [
        {
            "dataset": "D1",
            "family": "F1",
            "capture_id": "c1.pcap",
            "pcap_path": "/tmp/c1.pcap",
            "split": "train",
        }
    ]
    leakage = detect_capture_leakage(split_rows)
    paths = write_split_artifacts(split_rows, leakage, tmp_path)

    assert paths["split_manifest"].exists()
    assert paths["leakage_report"].exists()
    payload = json.loads(paths["leakage_report"].read_text())
    assert payload["has_leakage"] is False
