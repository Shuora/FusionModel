from src.data.tls_filter import classify_session_as_tls


def _tls_record(content_type: int, payload: bytes, version: int = 0x0303) -> bytes:
    length = len(payload).to_bytes(2, "big")
    return bytes([content_type]) + version.to_bytes(2, "big") + length + payload


def test_reject_non_tcp_session():
    ok, reason = classify_session_as_tls([b""], protocol="UDP", mode="strict")
    assert ok is False
    assert reason == "non_tcp"


def test_reject_cleartext_http_signature():
    ok, reason = classify_session_as_tls(
        [b"GET / HTTP/1.1\r\nHost: a\r\n\r\n"], protocol="TCP", mode="strict"
    )
    assert ok is False
    assert reason == "cleartext_signature"


def test_accept_handshake_session_in_strict_mode():
    handshake_payload = bytes([1]) + b"\x00" * 31
    session_chunks = [
        _tls_record(22, handshake_payload),
        _tls_record(23, b"\x17\x17\x17\x17"),
    ]
    ok, reason = classify_session_as_tls(session_chunks, protocol="TCP", mode="strict")
    assert ok is True
    assert reason == "tls"


def test_reject_invalid_tls_version():
    session_chunks = [_tls_record(22, b"\x01\x00", version=0x0200)]
    ok, reason = classify_session_as_tls(session_chunks, protocol="TCP", mode="strict")
    assert ok is False
    assert reason == "invalid_version"


def test_accept_midstream_appdata_if_enough_records():
    chunk = b"".join([_tls_record(23, b"abcd") for _ in range(6)])
    ok, reason = classify_session_as_tls([chunk], protocol="TCP", mode="strict")
    assert ok is True
    assert reason == "tls"
