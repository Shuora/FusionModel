from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import dpkt

from src.data.build_dataset import split_tls_and_non_tls


def _canonical_flow_key(
    src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str = "TCP"
) -> Tuple[str, int, str, int, str]:
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return src_ip, src_port, dst_ip, dst_port, proto
    return dst_ip, dst_port, src_ip, src_port, proto


def _flow_hash(key: Tuple[str, int, str, int, str]) -> str:
    raw = f"{key[0]}:{key[1]}-{key[2]}:{key[3]}-{key[4]}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def read_tcp_sessions(pcap_path: Path | str) -> List[Dict[str, Any]]:
    session_map: Dict[Tuple[str, int, str, int, str], Dict[str, Any]] = {}
    capture_id = Path(pcap_path).name

    with Path(pcap_path).open("rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            if not isinstance(ip.data, dpkt.tcp.TCP):
                continue
            tcp = ip.data
            if not tcp.data:
                continue

            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)
            key = _canonical_flow_key(src_ip, tcp.sport, dst_ip, tcp.dport, "TCP")

            if key not in session_map:
                session_map[key] = {
                    "session_id": _flow_hash(key),
                    "capture_id": capture_id,
                    "protocol": "TCP",
                    "payload_chunks": [],
                    "packet_count": 0,
                    "byte_count": 0,
                    "first_ts": float(ts),
                    "last_ts": float(ts),
                }

            entry = session_map[key]
            entry["payload_chunks"].append(bytes(tcp.data))
            entry["packet_count"] += 1
            entry["byte_count"] += len(tcp.data)
            entry["last_ts"] = float(ts)

    return sorted(session_map.values(), key=lambda x: x["session_id"])


def classify_pcap_sessions(
    pcap_path: Path | str, mode: str = "strict"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sessions = read_tcp_sessions(pcap_path)
    accepted, dropped = split_tls_and_non_tls(sessions, mode=mode)
    for item in dropped:
        item["capture_id"] = Path(pcap_path).name
    return accepted, dropped
