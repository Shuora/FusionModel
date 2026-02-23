from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


VALID_CONTENT_TYPES = {20, 21, 22, 23}
VALID_VERSIONS = {0x0300, 0x0301, 0x0302, 0x0303, 0x0304}
MAX_RECORD_LEN = 18432
HANDSHAKE_TYPES = {1, 2, 11}
CLEAR_TEXT_SIGNATURES = (
    b"GET ",
    b"POST ",
    b"HTTP/",
    b"PUT ",
    b"DELETE ",
    b"HEAD ",
)


def _parse_tls_records(chunk: bytes) -> Tuple[List[Tuple[int, int, bytes]], str | None]:
    records: List[Tuple[int, int, bytes]] = []
    i = 0
    while i + 5 <= len(chunk):
        content_type = chunk[i]
        version = int.from_bytes(chunk[i + 1 : i + 3], "big")
        length = int.from_bytes(chunk[i + 3 : i + 5], "big")

        if content_type not in VALID_CONTENT_TYPES:
            return records, "bad_header"
        if version not in VALID_VERSIONS:
            return records, "invalid_version"
        if length < 1 or length > MAX_RECORD_LEN:
            return records, "bad_header"
        if i + 5 + length > len(chunk):
            return records, "bad_header"

        payload = chunk[i + 5 : i + 5 + length]
        records.append((content_type, version, payload))
        i += 5 + length

    return records, None


def _looks_like_cleartext(chunk: bytes) -> bool:
    if not chunk:
        return False
    prefix = chunk[:8].upper()
    return any(prefix.startswith(sig) for sig in CLEAR_TEXT_SIGNATURES)


def classify_session_as_tls(
    payload_chunks: Sequence[bytes],
    protocol: str = "TCP",
    mode: str = "strict",
) -> Tuple[bool, str]:
    if protocol.upper() != "TCP":
        return False, "non_tcp"
    if mode not in {"strict", "relaxed"}:
        raise ValueError(f"Unsupported tls filter mode: {mode}")
    if not payload_chunks:
        return False, "bad_header"

    first = payload_chunks[0] or b""
    parsed_probe, probe_reason = _parse_tls_records(first) if first else ([], "bad_header")
    if not parsed_probe and _looks_like_cleartext(first):
        return False, "cleartext_signature"
    if probe_reason == "invalid_version":
        return False, "invalid_version"

    all_records: List[Tuple[int, int, bytes]] = []
    parse_error = None
    for chunk in payload_chunks:
        records, reason = _parse_tls_records(chunk)
        all_records.extend(records)
        if reason == "invalid_version":
            return False, "invalid_version"
        if reason:
            parse_error = reason

    if not all_records:
        return False, parse_error or "bad_header"

    if mode == "relaxed":
        return True, "tls"

    if len(all_records) < 2:
        return False, "no_handshake_evidence"

    has_handshake = any(
        ct == 22 and payload and payload[0] in HANDSHAKE_TYPES
        for ct, _, payload in all_records
    )
    appdata_run = 0
    for content_type, _, _ in all_records:
        if content_type == 23:
            appdata_run += 1
        else:
            appdata_run = 0
        if appdata_run >= 6:
            return True, "tls"

    if has_handshake:
        return True, "tls"

    return False, "no_handshake_evidence"
