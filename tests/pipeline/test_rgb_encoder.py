def test_rgb_encoder_output_shape():
    import numpy as np

    from src.pipeline.rgb_encoder import encode_tls_rgb

    sample = {"records": [], "handshake": {}, "stats": {}}
    img = encode_tls_rgb(sample, image_size=28)
    assert isinstance(img, np.ndarray)
    assert img.shape == (28, 28, 3)
