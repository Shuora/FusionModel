from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, TCP, Raw, wrpcap

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


def test_bytes_to_rgb_image_returns_28x28x3_uint8() -> None:
    image = bytes_to_rgb_image(bytes(range(255)) * 4)
    assert image.shape == (28, 28, 3)
    assert image.dtype == np.uint8
    assert image[..., 0].max() > 0
    assert image[..., 1].max() > 0
