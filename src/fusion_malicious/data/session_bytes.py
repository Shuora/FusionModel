from pathlib import Path
from typing import Union

import numpy as np
from scapy.all import Raw
from scapy.utils import PcapReader


def read_session_bytes(pcap_path: Union[str, Path]) -> bytes:
    payload = bytearray()
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            raw_layer = packet.getlayer(Raw)
            if raw_layer is None:
                continue
            payload.extend(raw_layer.load or b"")
    return bytes(payload)


def normalize_session_bytes(raw_bytes: bytes, size: int = 784) -> np.ndarray:
    vector = np.frombuffer(raw_bytes, dtype=np.uint8)
    if vector.size >= size:
        return vector[:size].copy()
    output = np.zeros(size, dtype=np.uint8)
    output[: vector.size] = vector
    return output
