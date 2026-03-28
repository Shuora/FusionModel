from pathlib import Path
from scapy.all import Ether, IP, TCP, Raw, rdpcap, wrpcap

from fusion_malicious.data.manifest import build_manifest_dataframe
from fusion_malicious.data.cleaning import (
    anonymize_session_pcap,
    fingerprint_session_bytes,
    should_keep_session,
)
from fusion_malicious.data.splitcap import build_splitcap_command
from scripts.prepare_dataset import (
    collect_session_paths,
    discover_capture_files,
    prepare_splitcap_input,
)


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


def test_build_manifest_dataframe_for_multiclass_task(tmp_path: Path) -> None:
    paths = []
    for family in ("Dridex", "Trickbot"):
        session_path = tmp_path / "SourceData" / "MTA" / family / f"{family}.pcap"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_bytes(b"pcap")
        paths.append(session_path)

    frame = build_manifest_dataframe(paths, task_name="mta")
    assert len(frame) == 2
    expected_label_map = {"Dridex": 0, "Trickbot": 1}
    for row in frame.itertuples():
        assert row.label_name in expected_label_map
        assert row.label_id == expected_label_map[row.label_name]


def test_discover_capture_files_filters_by_task(tmp_path: Path) -> None:
    mta_path = tmp_path / "SourceData" / "MTA" / "Dridex" / "a.pcap"
    benign_path = tmp_path / "SourceData" / "ISCX-VPN-NonVPN-2016" / "VPN-PCAPs-02" / "b.pcapng"
    other_path = tmp_path / "SourceData" / "USTC-TFC2016" / "c.pcap"
    for path in (mta_path, benign_path, other_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    binary_files = discover_capture_files(tmp_path / "SourceData", "binary")
    assert mta_path in binary_files
    assert benign_path in binary_files
    assert other_path not in binary_files

    ustc_files = discover_capture_files(tmp_path / "SourceData", "ustc")
    assert ustc_files == [other_path]


def test_prepare_splitcap_input_converts_pcapng(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "sample.pcapng"
    source.write_bytes(b"pcapng")
    commands = []

    def fake_run(command, check):
        commands.append(command)
        Path(command[-1]).write_bytes(b"pcap")

    monkeypatch.setattr("scripts.prepare_dataset.which", lambda name: "/usr/bin/editcap")
    monkeypatch.setattr("scripts.prepare_dataset.subprocess.run", fake_run)
    converted = prepare_splitcap_input(
        source,
        working_dir=tmp_path / "work",
        editcap_path="editcap",
    )
    assert converted.suffix == ".pcap"
    assert converted.exists()
    assert commands[0][:3] == ["/usr/bin/editcap", "-F", "pcap"]


def test_prepare_splitcap_input_converts_pcapng_magic_with_pcap_suffix(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "misnamed.pcap"
    source.write_bytes(b"\x0a\x0d\x0d\x0a" + b"payload")
    commands = []

    def fake_run(command, check):
        commands.append(command)
        Path(command[-1]).write_bytes(b"pcap")

    monkeypatch.setattr("scripts.prepare_dataset.which", lambda name: "/usr/bin/editcap")
    monkeypatch.setattr("scripts.prepare_dataset.subprocess.run", fake_run)
    converted = prepare_splitcap_input(
        source,
        working_dir=tmp_path / "work",
        editcap_path="editcap",
    )
    assert converted.suffix == ".pcap"
    assert converted.exists()
    assert converted.name != source.name
    assert commands[0][:3] == ["/usr/bin/editcap", "-F", "pcap"]


def test_collect_session_paths_reuses_checkpointed_splitcap_output(tmp_path: Path) -> None:
    raw_path = tmp_path / "captures" / "sample.pcap"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"pcap")
    output_root = tmp_path / "dataset"
    session_dir = output_root / "binary" / "sessions_raw" / "sample"
    session_dir.mkdir(parents=True, exist_ok=True)
    expected_session = session_dir / "TCP" / "flow.pcap"
    expected_session.parent.mkdir(parents=True, exist_ok=True)
    expected_session.write_bytes(b"x")
    (session_dir / ".splitcap.done").write_text("ok")

    sessions = collect_session_paths(
        [raw_path],
        task="binary",
        output_root=output_root,
        splitcap_exe=Path("Tools/SplitCap.exe"),
        splitcap_launcher=None,
        editcap_path="editcap",
        skip_splitcap=False,
        resume_splitcap=True,
    )
    assert sessions == [expected_session]


def test_build_splitcap_command_supports_launcher(tmp_path: Path) -> None:
    command = build_splitcap_command(
        splitcap_exe=Path("Tools/SplitCap.exe"),
        input_pcap=tmp_path / "input.pcap",
        output_dir=tmp_path / "sessions",
        launcher=["mono"],
    )
    assert command[0] == "mono"
    assert command[1].endswith("SplitCap.exe")
