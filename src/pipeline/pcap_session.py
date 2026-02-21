from __future__ import annotations

import hashlib
import logging
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import dpkt

ByteMode = Literal["payload", "full_packet"]
LOGGER = logging.getLogger("pcap_session")


@dataclass(frozen=True)
class SessionKey:
    proto: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int

    def to_id(self) -> str:
        return f"{self.proto}_{self.src_ip}_{self.src_port}_{self.dst_ip}_{self.dst_port}"


@dataclass(frozen=True)
class SessionRecord:
    key: SessionKey
    payload: bytes
    first_ts: float
    last_ts: float


def ip_to_str(ip_bytes: bytes) -> str:
    try:
        return socket.inet_ntoa(ip_bytes).replace(".", "-")
    except OSError:
        return "0-0-0-0"


def _canonicalize_endpoints(
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    merge_bidirectional: bool,
) -> Tuple[str, int, str, int]:
    if not merge_bidirectional:
        return src_ip, src_port, dst_ip, dst_port
    left = (src_ip, int(src_port))
    right = (dst_ip, int(dst_port))
    if left <= right:
        return left[0], left[1], right[0], right[1]
    return right[0], right[1], left[0], left[1]


def _sanitize_full_packet(raw_buf: bytes) -> bytes:
    """清理可泄漏字段，减少模型对 IP/MAC/端口的投机依赖。"""
    try:
        eth = dpkt.ethernet.Ethernet(raw_buf)
        eth.src = b"\x00" * len(eth.src)
        eth.dst = b"\x00" * len(eth.dst)
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            ip.src = b"\x00" * len(ip.src)
            ip.dst = b"\x00" * len(ip.dst)
            if isinstance(ip.data, dpkt.tcp.TCP):
                ip.data.sport = 0
                ip.data.dport = 0
            elif isinstance(ip.data, dpkt.udp.UDP):
                ip.data.sport = 0
                ip.data.dport = 0
        return bytes(eth)
    except Exception:
        return raw_buf


def _extract_chunk(transport, raw_buf: bytes, byte_mode: ByteMode, sanitize_headers: bool) -> bytes:
    if byte_mode == "payload":
        return bytes(getattr(transport, "data", b""))
    if sanitize_headers:
        return _sanitize_full_packet(raw_buf)
    return raw_buf


def _collect_session_chunks_from_pcap(
    pcap_path: Path,
    *,
    byte_mode: ByteMode = "payload",
    merge_bidirectional: bool = True,
    sanitize_headers: bool = True,
    min_chunk_bytes: int = 1,
) -> Dict[SessionKey, List[Tuple[float, bytes]]]:
    sessions: Dict[SessionKey, List[Tuple[float, bytes]]] = {}
    with pcap_path.open("rb") as f:
        pcap = dpkt.pcap.Reader(f)
        packet_count = 0
        while True:
            try:
                ts, raw_buf = next(pcap)
            except StopIteration:
                break
            except dpkt.NeedData as exc:
                # Keep already parsed packets when only the tail record is truncated.
                if packet_count > 0:
                    LOGGER.warning(
                        "PCAP尾部数据不完整，已保留前序可解析包: file=%s packets=%s err=%s",
                        pcap_path,
                        packet_count,
                        exc,
                    )
                    break
                raise

            packet_count += 1
            try:
                eth = dpkt.ethernet.Ethernet(raw_buf)
            except (dpkt.UnpackError, ValueError):
                continue

            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue

            transport = ip.data
            if isinstance(transport, dpkt.tcp.TCP):
                proto = "TCP"
            elif isinstance(transport, dpkt.udp.UDP):
                proto = "UDP"
            else:
                continue

            src_ip = ip_to_str(ip.src)
            dst_ip = ip_to_str(ip.dst)
            src_port = int(getattr(transport, "sport", 0))
            dst_port = int(getattr(transport, "dport", 0))

            src_ip, src_port, dst_ip, dst_port = _canonicalize_endpoints(
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                merge_bidirectional=merge_bidirectional,
            )
            key = SessionKey(proto=proto, src_ip=src_ip, src_port=src_port, dst_ip=dst_ip, dst_port=dst_port)

            chunk = _extract_chunk(transport, raw_buf, byte_mode=byte_mode, sanitize_headers=sanitize_headers)
            if len(chunk) < int(min_chunk_bytes):
                continue
            sessions.setdefault(key, []).append((float(ts), chunk))
    return sessions


def extract_session_records_from_pcap(
    pcap_path: Path,
    *,
    byte_mode: ByteMode = "payload",
    merge_bidirectional: bool = True,
    sanitize_headers: bool = True,
    min_chunk_bytes: int = 1,
    min_session_bytes: int = 1,
) -> List[SessionRecord]:
    sessions = _collect_session_chunks_from_pcap(
        pcap_path,
        byte_mode=byte_mode,
        merge_bidirectional=merge_bidirectional,
        sanitize_headers=sanitize_headers,
        min_chunk_bytes=min_chunk_bytes,
    )

    records: List[SessionRecord] = []
    for key, chunks in sessions.items():
        chunks.sort(key=lambda item: item[0])
        payload = b"".join(part for _, part in chunks)
        if len(payload) < int(min_session_bytes):
            continue
        records.append(
            SessionRecord(
                key=key,
                payload=payload,
                first_ts=float(chunks[0][0]),
                last_ts=float(chunks[-1][0]),
            )
        )
    records.sort(key=lambda r: (r.first_ts, r.key.to_id()))
    return records


def extract_sessions_from_pcap(
    pcap_path: Path,
    *,
    byte_mode: ByteMode = "payload",
    merge_bidirectional: bool = True,
    sanitize_headers: bool = True,
    min_chunk_bytes: int = 1,
    min_session_bytes: int = 1,
) -> Dict[SessionKey, bytes]:
    records = extract_session_records_from_pcap(
        pcap_path,
        byte_mode=byte_mode,
        merge_bidirectional=merge_bidirectional,
        sanitize_headers=sanitize_headers,
        min_chunk_bytes=min_chunk_bytes,
        min_session_bytes=min_session_bytes,
    )

    merged: Dict[SessionKey, bytes] = {}
    for record in records:
        merged[record.key] = record.payload
    return merged


def normalize_session_bytes(session_bytes: bytes, target_length: int = 784) -> bytes:
    if len(session_bytes) >= target_length:
        return session_bytes[:target_length]
    return session_bytes + (b"\x00" * (target_length - len(session_bytes)))


def session_sha1(session_bytes: bytes) -> str:
    return hashlib.sha1(session_bytes).hexdigest()


def split_class_files(
    class_to_pcaps: Dict[str, List[Path]],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, List[Path]]]:
    rng = random.Random(int(seed))
    splits: Dict[str, Dict[str, List[Path]]] = {}
    for class_name, files in sorted(class_to_pcaps.items(), key=lambda item: item[0].lower()):
        ordered = sorted(set(files))
        rng.shuffle(ordered)
        total = len(ordered)
        if total <= 1:
            train_files = ordered
            test_files = []
        else:
            cut = int(round(total * float(train_ratio)))
            cut = max(1, min(total - 1, cut))
            train_files = ordered[:cut]
            test_files = ordered[cut:]
        splits[class_name] = {"Train": train_files, "Test": test_files}
    return splits


def collect_ustc_pcaps(source_dir: Path) -> Dict[str, List[Path]]:
    class_to_pcaps: Dict[str, List[Path]] = {}
    for pcap_path in sorted(source_dir.glob("*.pcap")):
        class_name = pcap_path.stem
        class_to_pcaps.setdefault(class_name, []).append(pcap_path)
    return class_to_pcaps


def collect_cic_pcaps(
    source_dir: Path,
    *,
    include_majors: Iterable[str] | None = None,
    label_map: Dict[str, str] | None = None,
) -> Dict[str, List[Path]]:
    include = {name.lower() for name in include_majors} if include_majors else None
    mapped = {k.lower(): v for k, v in (label_map or {}).items()}
    class_to_pcaps: Dict[str, List[Path]] = {}

    for major_dir in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
        if not major_dir.is_dir():
            continue
        major_name_raw = major_dir.name
        major_name = mapped.get(major_name_raw.lower(), major_name_raw)
        if include is not None and major_name.lower() not in include and major_name_raw.lower() not in include:
            continue

        pcaps: List[Path] = []
        for child in sorted(major_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_file() and child.suffix.lower() == ".pcap":
                pcaps.append(child)
                continue
            if child.is_dir():
                pcaps.extend(sorted(child.glob("*.pcap")))

        if pcaps:
            class_to_pcaps.setdefault(major_name, []).extend(pcaps)
    return class_to_pcaps


def collect_mfcp_pcaps(
    source_dir: Path,
    *,
    include_families: Iterable[str] | None = None,
    label_map: Dict[str, str] | None = None,
) -> Dict[str, List[Path]]:
    """
    MFCP 采集入口。

    MFCP 目录结构与 CIC 的分层采集逻辑兼容，这里复用同一实现，
    仅在语义上区分参数名（families）。
    """
    return collect_cic_pcaps(
        source_dir,
        include_majors=include_families,
        label_map=label_map,
    )
