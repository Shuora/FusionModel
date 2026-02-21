from __future__ import annotations


def is_tls_record(header: bytes) -> bool:
    """Validate a TLS record header using basic type/version/length rules."""
    if len(header) < 5:
        return False

    content_type = header[0]
    version = (header[1] << 8) | header[2]
    length = (header[3] << 8) | header[4]

    return (
        content_type in {20, 21, 22, 23}
        and 0x0300 <= version <= 0x0304
        and 0 < length <= 18432
    )
