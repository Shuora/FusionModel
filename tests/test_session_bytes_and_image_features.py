from pathlib import Path

import numpy as np
import pytest
from scapy.all import Ether, IP, TCP, UDP, Padding, Raw, wrpcap

from fusion_malicious.data.image_features import bytes_to_rgb_image
from fusion_malicious.data.session_bytes import normalize_session_bytes, read_session_bytes


def test_normalize_session_bytes_pads_and_truncates() -> None:
    assert normalize_session_bytes(b"\x01\x02", size=4).tolist() == [1, 2, 0, 0]
    assert normalize_session_bytes(bytes(range(6)), size=4).tolist() == [0, 1, 2, 3]


def test_read_session_bytes_extracts_packet_payload(tmp_path: Path) -> None:
    pcap_path = tmp_path / "one_session.pcap"
    packet = Ether() / IP() / TCP() / Raw(load=b"\x11\x22\x33")
    wrpcap(str(pcap_path), [packet])
    raw_bytes = read_session_bytes(pcap_path)
    assert raw_bytes[:3] == b"\x11\x22\x33"


def test_read_session_bytes_uses_transport_payload_when_raw_layer_missing(tmp_path: Path) -> None:
    pcap_path = tmp_path / "padding_session.pcap"
    packets = [
        Ether() / IP() / TCP() / Padding(load=b"\xaa\xbb"),
        Ether() / IP() / UDP() / Padding(load=b"\xcc"),
    ]
    wrpcap(str(pcap_path), packets)

    raw_bytes = read_session_bytes(pcap_path)

    assert raw_bytes == b"\xaa\xbb\xcc"


def test_read_session_bytes_returns_empty_for_header_only_session(tmp_path: Path) -> None:
    pcap_path = tmp_path / "empty_session.pcap"
    packets = [
        Ether() / IP() / TCP(),
        Ether() / IP() / UDP(),
    ]
    wrpcap(str(pcap_path), packets)

    raw_bytes = read_session_bytes(pcap_path)

    assert raw_bytes == b""


def test_bytes_to_rgb_image_returns_28x28x3_uint8() -> None:
    image = bytes_to_rgb_image(bytes(range(255)) * 4)
    assert image.shape == (28, 28, 3)
    assert image.dtype == np.uint8
    assert image[..., 0].max() > 0
    assert image[..., 1].max() > 0
    assert image[..., 2].max() > 0


def test_bytes_to_rgb_image_diff_handles_negative_changes() -> None:
    raw = bytes([0x0A, 0x05]) + b"\x00" * (784 - 2)
    diff_channel = bytes_to_rgb_image(raw)[..., 1].flatten()
    assert diff_channel[1] == 5


def test_bytes_to_rgb_image_requires_784_bytes() -> None:
    with pytest.raises(ValueError, match="size=784"):
        bytes_to_rgb_image(bytes(100), size=100)
