from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable

from scapy.all import Ether, IP, TCP, UDP, rdpcap, wrpcap


def anonymize_session_pcap(input_path: Path, output_path: Path, seed: int) -> None:
    """
    Rewrite MAC and IPv4 endpoints in a deterministic way while keeping payloads intact.
    """
    packets = rdpcap(str(input_path))
    rnd = random.Random(seed)
    mac_map: dict[str, str] = {}
    ip_map: dict[str, str] = {}

    def map_value(value: str, mapping: dict[str, str], generator: Callable[[], str]) -> str:
        if value not in mapping:
            mapping[value] = generator()
        return mapping[value]

    def random_mac() -> str:
        return ":".join(f"{rnd.randint(0, 255):02x}" for _ in range(6))

    def random_ip() -> str:
        return ".".join(str(rnd.randint(1, 254)) for _ in range(4))

    for packet in packets:
        if Ether in packet:
            eth_layer = packet[Ether]
            eth_layer.src = map_value(eth_layer.src, mac_map, random_mac)
            eth_layer.dst = map_value(eth_layer.dst, mac_map, random_mac)
        if IP in packet:
            ip_layer = packet[IP]
            ip_layer.src = map_value(ip_layer.src, ip_map, random_ip)
            ip_layer.dst = map_value(ip_layer.dst, ip_map, random_ip)
            ip_layer.chksum = None
            if TCP in packet:
                packet[TCP].chksum = None
            if UDP in packet:
                packet[UDP].chksum = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(output_path), packets)


def fingerprint_session_bytes(payload: bytes) -> str:
    """
    Fingerprint bytes belonging to a session using a stable hash.
    """
    return hashlib.sha256(payload).hexdigest()


def should_keep_session(payload: bytes, seen: set[str]) -> bool:
    """
    Filter out empty payloads and duplicates by fingerprint.
    """
    if not payload:
        return False
    fingerprint = fingerprint_session_bytes(payload)
    if fingerprint in seen:
        return False
    seen.add(fingerprint)
    return True
