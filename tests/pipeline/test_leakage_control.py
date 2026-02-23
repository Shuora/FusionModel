def test_sni_is_masked_in_leakage_reduced():
    from src.pipeline.leakage_control import redact_sensitive_fields

    row = {"sni": "malicious.example.com", "cert_fingerprint": "abcd"}
    out = redact_sensitive_fields(row)
    assert out["sni"] != "malicious.example.com"
