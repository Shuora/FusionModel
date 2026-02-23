from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dpkt


FlowKey = Tuple[str, int, str, int, str]
PacketRow = Tuple[float, bytes]


def _canonical_flow_key(
    src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str = "TCP"
) -> FlowKey:
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return src_ip, src_port, dst_ip, dst_port, proto
    return dst_ip, dst_port, src_ip, src_port, proto


def _flow_hash(key: FlowKey) -> str:
    raw = f"{key[0]}:{key[1]}-{key[2]}:{key[3]}-{key[4]}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def split_pcap_to_session_pcaps(pcap_path: Path | str, out_dir: Path | str) -> List[Path]:
    pcap_path = Path(pcap_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[FlowKey, List[PacketRow]] = {}
    with pcap_path.open("rb") as f:
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
            groups.setdefault(key, []).append((float(ts), buf))

    written: List[Path] = []
    stem = pcap_path.stem
    for key in sorted(groups.keys(), key=_flow_hash):
        session_id = _flow_hash(key)
        target = out_dir / f"{stem}_{session_id}.pcap"
        with target.open("wb") as wf:
            writer = dpkt.pcap.Writer(wf)
            for ts, buf in groups[key]:
                writer.writepkt(buf, ts=ts)
            writer.close()
        written.append(target)
    return written


def cleanup_session_pcaps(paths: Sequence[Path | str]) -> int:
    removed = 0
    for item in paths:
        p = Path(item)
        if p.exists():
            p.unlink()
            removed += 1
    return removed

