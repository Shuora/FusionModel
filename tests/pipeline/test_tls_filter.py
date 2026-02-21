def test_is_tls_record_accepts_valid_header():
    from src.pipeline.tls_filter import is_tls_record

    header = bytes([22, 0x03, 0x03, 0x00, 0x2F])
    assert is_tls_record(header) is True
