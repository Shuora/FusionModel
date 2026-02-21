from __future__ import annotations


def redact_sensitive_fields(row: dict) -> dict:
    """Mask sensitive TLS fields for leakage-reduced experiments."""
    redacted = dict(row)
    redacted["sni"] = hash(redacted.get("sni", "")) % (2**16)
    redacted.pop("cert_fingerprint", None)
    return redacted
