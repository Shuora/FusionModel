from pathlib import Path
from typing import Union

import numpy as np
from scapy.all import IP, IPv6, Raw, TCP, UDP
from scapy.utils import PcapReader


def _packet_payload_bytes(packet) -> bytes:
    for layer_type in (TCP, UDP):
        if layer_type in packet:
            return bytes(packet[layer_type].payload)

    for layer_type in (IP, IPv6):
        if layer_type in packet:
            return bytes(packet[layer_type].payload.payload)

    raw_layer = packet.getlayer(Raw)
    if raw_layer is not None:
        return raw_layer.load or b""

    return b""


def read_session_bytes(pcap_path: Union[str, Path]) -> bytes:
    payload = bytearray()
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            packet_payload = _packet_payload_bytes(packet)
            if not packet_payload:
                continue
            payload.extend(packet_payload)
    return bytes(payload)


def normalize_session_bytes(raw_bytes: bytes, size: int = 784) -> np.ndarray:
    vector = np.frombuffer(raw_bytes, dtype=np.uint8)
    if vector.size >= size:
        return vector[:size].copy()
    output = np.zeros(size, dtype=np.uint8)
    output[: vector.size] = vector
    return output
