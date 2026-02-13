"""RGB builder for 28x28 encrypted traffic images.

R channel: paper-style 784-byte mapping.
G/B channels: semantic and behavioral mappings adapted from prior experiments.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMG_H = 28
IMG_W = 28
SESSION_SIZE = IMG_H * IMG_W
G_HEAD_SIZE = 1024


def _ensure_session_len(unified_784: bytes) -> np.ndarray:
    arr = np.frombuffer(unified_784[:SESSION_SIZE], dtype=np.uint8)
    if arr.size < SESSION_SIZE:
        arr = np.pad(arr, (0, SESSION_SIZE - arr.size), mode="constant")
    return arr


def extract_r_channel(unified_784: bytes) -> np.ndarray:
    # Paper-style direct byte-to-pixel mapping.
    return _ensure_session_len(unified_784)


def extract_g_channel(raw_payload: bytes) -> np.ndarray:
    # Semantic channel: pseudo handshake/metadata fingerprints from payload head.
    handshake = raw_payload[:G_HEAD_SIZE]
    if len(handshake) < G_HEAD_SIZE:
        handshake = handshake + (b"\x00" * (G_HEAD_SIZE - len(handshake)))

    hs = np.frombuffer(handshake, dtype=np.uint8)

    cipher_suite_diversity = int(np.unique(hs[:32]).size)
    sni_region = hs[32:96]
    if sni_region.size and sni_region.any():
        probs = np.bincount(sni_region, minlength=256).astype(np.float32)
        probs = probs / max(float(sni_region.size), 1.0)
        probs = probs[probs > 0]
        sni_entropy = float(-(probs * np.log2(probs)).sum())
    else:
        sni_entropy = 0.0

    cert_anomaly = int(hs.max() - hs.min()) if hs.size else 0

    g = np.zeros(SESSION_SIZE, dtype=np.uint8)
    g[0] = cipher_suite_diversity % 256
    g[1] = int(sni_entropy * 32.0) % 256
    g[2] = cert_anomaly % 256

    fill_len = min(hs.size, SESSION_SIZE - 3)
    if fill_len > 0:
        g[3 : 3 + fill_len] = hs[:fill_len]
    return g


def extract_b_channel(raw_payload: bytes) -> np.ndarray:
    # Behavioral channel: packet-size-like segmented stats from payload bytes.
    arr = np.frombuffer(raw_payload, dtype=np.uint8)
    pkt_size = 1500
    chunks = [arr[i : i + pkt_size] for i in range(0, len(arr), pkt_size)]

    pkt_lens = [len(c) for c in chunks]
    mean_len = int(np.mean(pkt_lens)) if pkt_lens else 0

    if len(chunks) > 1:
        pseudo_intervals = [int(chunks[i][0]) - int(chunks[i - 1][0]) for i in range(1, len(chunks)) if len(chunks[i]) > 0 and len(chunks[i - 1]) > 0]
        interval_var = int(np.var(pseudo_intervals)) if pseudo_intervals else 0
    else:
        interval_var = 0

    duration = len(chunks)

    b = np.zeros(SESSION_SIZE, dtype=np.uint8)
    b[0] = mean_len % 256
    b[1] = interval_var % 256
    b[2] = duration % 256

    fill_len = min(len(pkt_lens), SESSION_SIZE - 3)
    if fill_len > 0:
        values = np.clip(np.asarray(pkt_lens[:fill_len], dtype=np.int32), 0, 255).astype(np.uint8)
        pos = np.linspace(3, SESSION_SIZE - 1, num=fill_len, dtype=np.int32)
        b[pos] = values

    return b


def build_rgb_array(unified_784: bytes, raw_payload: bytes) -> np.ndarray:
    r = extract_r_channel(unified_784)
    g = extract_g_channel(raw_payload)
    b = extract_b_channel(raw_payload)
    rgb = np.stack([r, g, b], axis=1).reshape(IMG_H, IMG_W, 3)
    return rgb.astype(np.uint8)


def build_rgb_image(unified_784: bytes, raw_payload: bytes) -> Image.Image:
    rgb = build_rgb_array(unified_784, raw_payload)
    return Image.fromarray(rgb, mode="RGB")
