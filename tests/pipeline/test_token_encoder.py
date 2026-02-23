def test_token_encoder_has_cls_sep():
    from src.pipeline.token_encoder import encode_tls_tokens

    ids = encode_tls_tokens({"records": [], "handshake": {}}, max_len=32)
    assert ids[0] == 101
    assert ids[-1] == 102
