from __future__ import annotations

from typing import Dict, Iterable, List


def sessionize_packets(packets: Iterable[Dict], key: str = "session_id") -> Dict[str, List[Dict]]:
    """Group packet-like rows by session key."""
    sessions: Dict[str, List[Dict]] = {}
    for packet in packets:
        session_id = str(packet.get(key, "unknown"))
        sessions.setdefault(session_id, []).append(packet)
    return sessions
