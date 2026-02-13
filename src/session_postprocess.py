"""Session extraction and post-processing helpers for encrypted traffic preprocessing."""

from __future__ import annotations

import hashlib
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def _load_dpkt():
    try:
        import dpkt  # type: ignore

        return dpkt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("dpkt is required. Install with: pip install dpkt") from exc


def _ip_to_str(ip_bytes: bytes) -> Optional[str]:
    try:
        return socket.inet_ntoa(ip_bytes)
    except OSError:
        return None


@dataclass(frozen=True)
class SessionKey:
    proto: str
    ep1_ip: str
    ep1_port: int
    ep2_ip: str
    ep2_port: int


def _normalize_bidir_key(
    proto: str,
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
) -> SessionKey:
    a = (src_ip, int(src_port))
    b = (dst_ip, int(dst_port))
    if a <= b:
        return SessionKey(proto, a[0], a[1], b[0], b[1])
    return SessionKey(proto, b[0], b[1], a[0], a[1])


def iter_pcap_packets(pcap_path: Path) -> Iterable[bytes]:
    dpkt = _load_dpkt()
    with pcap_path.open("rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            reader = dpkt.pcapng.Reader(f)
        else:
            reader = dpkt.pcap.Reader(f)
        for _, buf in reader:
            yield buf


def extract_payload_from_session_pcap(session_pcap: Path) -> bytes:
    dpkt = _load_dpkt()
    payload = bytearray()
    for buf in iter_pcap_packets(session_pcap):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            continue

        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue

        trans = None
        if isinstance(ip.data, dpkt.tcp.TCP):
            trans = ip.data
        elif isinstance(ip.data, dpkt.udp.UDP):
            trans = ip.data

        if trans is None:
            continue

        data = bytes(trans.data or b"")
        if data:
            payload.extend(data)

    return bytes(payload)


def split_sessions_python_fallback(input_pcap: Path) -> Dict[SessionKey, bytes]:
    dpkt = _load_dpkt()
    sessions: Dict[SessionKey, bytearray] = {}

    for buf in iter_pcap_packets(input_pcap):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            continue

        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue

        trans = None
        proto = None
        if isinstance(ip.data, dpkt.tcp.TCP):
            proto = "TCP"
            trans = ip.data
        elif isinstance(ip.data, dpkt.udp.UDP):
            proto = "UDP"
            trans = ip.data

        if trans is None or proto is None:
            continue

        src_ip = _ip_to_str(ip.src)
        dst_ip = _ip_to_str(ip.dst)
        if not src_ip or not dst_ip:
            continue

        data = bytes(trans.data or b"")
        if not data:
            continue

        key = _normalize_bidir_key(proto, src_ip, int(trans.sport), dst_ip, int(trans.dport))
        sessions.setdefault(key, bytearray()).extend(data)

    return {k: bytes(v) for k, v in sessions.items()}


def unify_length(payload: bytes, max_len: int = 784) -> bytes:
    if len(payload) >= max_len:
        return payload[:max_len]
    return payload + (b"\x00" * (max_len - len(payload)))


def sha1_hex(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def short_digest(payload: bytes, n: int = 12) -> str:
    return sha1_hex(payload)[: int(n)]


def make_sample_stem(source_pcap: Path, session_index: int, payload: bytes) -> str:
    digest = short_digest(payload, 12)
    return f"{source_pcap.stem}_s{session_index:05d}_{digest}"
