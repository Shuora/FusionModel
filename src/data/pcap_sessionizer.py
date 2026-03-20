from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _open_packet_reader(fp):
    try:
        return dpkt.pcap.Reader(fp)
    except (ValueError, dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
        fp.seek(0)
        return dpkt.pcapng.Reader(fp)


def _iter_packets(reader):
    packet_iter = iter(reader)
    while True:
        try:
            yield next(packet_iter)
        except StopIteration:
            return
        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError):
            return


def _reader_datalink(reader) -> Optional[int]:
    fn = getattr(reader, "datalink", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return None
    return None


def _parse_ip_packet(buf: bytes, datalink: Optional[int]):
    if datalink == 101:
        try:
            return dpkt.ip.IP(buf)
        except Exception:
            return None

    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except Exception:
        pass

    try:
        return dpkt.ip.IP(buf)
    except Exception:
        return None


def _extract_l4_session_fields(buf: bytes, datalink: Optional[int], include_udp: bool):
    ip = _parse_ip_packet(buf, datalink)
    if ip is None:
        return None

    if isinstance(ip, dpkt.ip.IP):
        src_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)
        l4 = ip.data
    elif isinstance(ip, dpkt.ip6.IP6):
        src_ip = socket.inet_ntop(socket.AF_INET6, ip.src)
        dst_ip = socket.inet_ntop(socket.AF_INET6, ip.dst)
        l4 = ip.data
    else:
        return None

    if isinstance(l4, dpkt.tcp.TCP):
        if not l4.data:
            return None
        return src_ip, int(l4.sport), dst_ip, int(l4.dport), "TCP", bytes(l4.data)

    if include_udp and isinstance(l4, dpkt.udp.UDP):
        if not l4.data:
            return None
        return src_ip, int(l4.sport), dst_ip, int(l4.dport), "UDP", bytes(l4.data)

    return None


def read_tcp_sessions(pcap_path: Path | str, include_udp: bool = False) -> List[Dict[str, Any]]:
    session_map: Dict[Tuple[str, int, str, int, str], Dict[str, Any]] = {}
    capture_id = Path(pcap_path).name

    with Path(pcap_path).open("rb") as f:
        reader = _open_packet_reader(f)
        datalink = _reader_datalink(reader)
        for ts, buf in _iter_packets(reader):
            parsed = _extract_l4_session_fields(buf, datalink=datalink, include_udp=include_udp)
            if parsed is None:
                continue
            src_ip, src_port, dst_ip, dst_port, proto_name, payload = parsed
            key = _canonical_flow_key(src_ip, src_port, dst_ip, dst_port, proto_name)

            if key not in session_map:
                session_map[key] = {
                    "session_id": _flow_hash(key),
                    "capture_id": capture_id,
                    "protocol": proto_name,
                    "payload_chunks": [],
                    "packet_count": 0,
                    "byte_count": 0,
                    "first_ts": float(ts),
                    "last_ts": float(ts),
                }

            entry = session_map[key]
            entry["payload_chunks"].append(payload)
            entry["packet_count"] += 1
            entry["byte_count"] += len(payload)
            entry["last_ts"] = float(ts)

    return sorted(session_map.values(), key=lambda x: x["session_id"])


def classify_pcap_sessions(
    pcap_path: Path | str, mode: str = "strict"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sessions = read_tcp_sessions(pcap_path, include_udp=(mode == "session_full"))
    accepted, dropped = split_tls_and_non_tls(sessions, mode=mode)
    for item in dropped:
        item["capture_id"] = Path(pcap_path).name
    return accepted, dropped
