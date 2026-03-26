from pathlib import Path
from scapy.all import Ether, IP, TCP, Raw, rdpcap, wrpcap

from fusion_malicious.data.manifest import build_manifest_dataframe
from fusion_malicious.data.cleaning import (
    anonymize_session_pcap,
    fingerprint_session_bytes,
    should_keep_session,
)
from fusion_malicious.data.splitcap import build_splitcap_command


def test_build_splitcap_command_uses_repo_tool_path(tmp_path: Path) -> None:
    command = build_splitcap_command(
        splitcap_exe=Path("Tools/SplitCap.exe"),
        input_pcap=tmp_path / "input.pcap",
        output_dir=tmp_path / "sessions",
    )
    assert command[0].endswith("SplitCap.exe")
    assert "-r" in command
    assert "-o" in command
    assert "-s" in command
    s_index = command.index("-s")
    assert command[s_index + 1] == "session"


def test_anonymize_session_pcap_rewrites_endpoints_but_keeps_payload(tmp_path: Path) -> None:
    input_path = tmp_path / "input.pcap"
    output_path = tmp_path / "output.pcap"
    packet = (
        Ether(src="00:11:22:33:44:55", dst="aa:bb:cc:dd:ee:ff")
        / IP(src="1.1.1.1", dst="2.2.2.2")
        / TCP()
        / Raw(load=b"payload")
    )
    wrpcap(str(input_path), [packet])
    anonymize_session_pcap(input_path, output_path, seed=17)
    anonymized = rdpcap(str(output_path))[0]
    assert anonymized[IP].src != "1.1.1.1"
    assert anonymized[IP].dst != "2.2.2.2"
    assert bytes(anonymized[Raw].load) == b"payload"


def test_fingerprint_session_bytes_is_stable() -> None:
    assert fingerprint_session_bytes(b"abc") == fingerprint_session_bytes(b"abc")


def test_should_keep_session_rejects_empty_and_duplicate_payloads() -> None:
    seen = set()
    assert should_keep_session(b"", seen) is False
    assert should_keep_session(b"payload", seen) is True
    assert should_keep_session(b"payload", seen) is False


def test_build_manifest_dataframe_infers_dataset_and_family(tmp_path: Path) -> None:
    session_path = tmp_path / "SourceData" / "MTA" / "Trickbot" / "flow1.pcap"
    session_path.parent.mkdir(parents=True)
    session_path.write_bytes(b"pcap")
    frame = build_manifest_dataframe([session_path], task_name="binary")
    assert list(frame.columns)[:5] == [
        "sample_id",
        "dataset",
        "family",
        "source_path",
        "task_name",
    ]
    assert frame.loc[0, "dataset"] == "MTA"
    assert frame.loc[0, "family"] == "Trickbot"


def test_sample_id_stable_between_roots(tmp_path: Path) -> None:
    def make_session(root: Path) -> Path:
        session = root / "SourceData" / "MTA" / "Trickbot" / "flow1.pcap"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_bytes(b"pcap")
        return session

    session_a = make_session(tmp_path / "repo_a")
    session_b = make_session(tmp_path / "repo_b")
    sample_a = build_manifest_dataframe([session_a], task_name="binary").loc[0, "sample_id"]
    sample_b = build_manifest_dataframe([session_b], task_name="binary").loc[0, "sample_id"]
    assert sample_a == sample_b


def test_anonymize_session_creates_output_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "input.pcap"
    output_path = tmp_path / "nested" / "dir" / "output.pcap"
    packet = (
        Ether(src="00:11:22:33:44:55", dst="aa:bb:cc:dd:ee:ff")
        / IP(src="1.1.1.1", dst="2.2.2.2")
        / TCP()
        / Raw(load=b"payload")
    )
    wrpcap(str(input_path), [packet])
    anonymize_session_pcap(input_path, output_path, seed=17)
    assert output_path.exists()


def test_anonymize_invalidates_tcp_checksum(tmp_path: Path) -> None:
    input_path = tmp_path / "input.pcap"
    output_path = tmp_path / "output.pcap"
    packet = (
        Ether(src="00:11:22:33:44:55", dst="aa:bb:cc:dd:ee:ff")
        / IP(src="1.1.1.1", dst="2.2.2.2")
        / TCP()
        / Raw(load=b"payload")
    )
    wrpcap(str(input_path), [packet])
    original_tcp = rdpcap(str(input_path))[0][TCP].chksum
    anonymize_session_pcap(input_path, output_path, seed=17)
    anonymized = rdpcap(str(output_path))[0]
    assert anonymized[TCP].chksum != original_tcp
