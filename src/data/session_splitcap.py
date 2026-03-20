from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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


def _open_packet_reader(fp):
    try:
        return dpkt.pcap.Reader(fp)
    except (ValueError, dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
        fp.seek(0)
        return dpkt.pcapng.Reader(fp)


def _reader_datalink(reader) -> Optional[int]:
    fn = getattr(reader, "datalink", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return None
    return None


def _iter_packets(reader) -> Iterator[PacketRow]:
    packet_iter = iter(reader)
    while True:
        try:
            yield next(packet_iter)
        except StopIteration:
            return
        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError):
            # Tolerate truncated/corrupted tail records; keep packets parsed so far.
            return


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


def _extract_flow_fields(buf: bytes, datalink: Optional[int], include_udp: bool):
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
        return src_ip, int(l4.sport), dst_ip, int(l4.dport), "TCP"

    if include_udp and isinstance(l4, dpkt.udp.UDP):
        if not l4.data:
            return None
        return src_ip, int(l4.sport), dst_ip, int(l4.dport), "UDP"

    return None


def split_pcap_to_session_pcaps(
    pcap_path: Path | str, out_dir: Path | str, include_udp: bool = False
) -> List[Path]:
    pcap_path = Path(pcap_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[FlowKey, List[PacketRow]] = {}
    with pcap_path.open("rb") as f:
        reader = _open_packet_reader(f)
        datalink = _reader_datalink(reader)
        for ts, buf in _iter_packets(reader):
            parsed = _extract_flow_fields(buf, datalink=datalink, include_udp=include_udp)
            if parsed is None:
                continue
            src_ip, src_port, dst_ip, dst_port, proto_name = parsed
            key = _canonical_flow_key(src_ip, src_port, dst_ip, dst_port, proto_name)
            groups.setdefault(key, []).append((float(ts), buf))

    written: List[Path] = []
    stem = pcap_path.stem
    with pcap_path.open("rb") as f:
        writer_linktype = _reader_datalink(_open_packet_reader(f))
    for key in sorted(groups.keys(), key=_flow_hash):
        session_id = _flow_hash(key)
        target = out_dir / f"{stem}_{session_id}.pcap"
        with target.open("wb") as wf:
            writer = dpkt.pcap.Writer(wf, linktype=writer_linktype or dpkt.pcap.DLT_EN10MB)
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
