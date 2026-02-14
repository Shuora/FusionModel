from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def normalize_bytes(raw: bytes, target_length: int = 784) -> np.ndarray:
    """将原始字节序列统一到固定长度。"""
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size >= target_length:
        return arr[:target_length].copy()
    out = np.zeros(target_length, dtype=np.uint8)
    out[: arr.size] = arr
    return out


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist = np.bincount(values, minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    if prob.size == 0:
        return 0.0
    return float(-(prob * np.log2(prob)).sum())


def _safe_uint8(value: float) -> np.uint8:
    return np.uint8(max(0, min(255, int(round(value)))))


def _spread_values(values: Iterable[int], size: int, start: int = 0) -> np.ndarray:
    out = np.zeros(size, dtype=np.uint8)
    values = list(values)
    if not values:
        return out
    usable = min(len(values), size - start)
    if usable <= 0:
        return out
    positions = np.linspace(start, size - 1, num=usable, dtype=np.int64)
    clipped = np.clip(np.asarray(values[:usable], dtype=np.int64), 0, 255).astype(np.uint8)
    out[positions] = clipped
    return out


def build_rgb_image(
    raw_session_bytes: bytes,
    *,
    target_length: int = 784,
    image_size: int = 28,
) -> np.ndarray:
    """
    生成 28x28x3 RGB 图像。

    - R: 784 主序列（MVTBA 核心）
    - G: 前缀/握手统计 + 前缀字节
    - B: 会话行为统计 + 分散长度特征
    """
    if image_size * image_size != target_length:
        raise ValueError(f"image_size={image_size} 与 target_length={target_length} 不匹配")

    # R 通道：严格使用统一后的 784 序列
    r = normalize_bytes(raw_session_bytes, target_length=target_length)

    # G 通道：用前 1024 字节构造“握手/前缀语义增强”
    prefix = raw_session_bytes[:1024]
    prefix_arr = np.frombuffer(prefix, dtype=np.uint8)
    g = np.zeros(target_length, dtype=np.uint8)
    if prefix_arr.size:
        uniq = int(np.unique(prefix_arr[:64]).size)
        ent = _entropy(prefix_arr[: min(256, prefix_arr.size)])
        mean_v = float(prefix_arr.mean())
        std_v = float(prefix_arr.std())
        g[0] = _safe_uint8(uniq)
        g[1] = _safe_uint8(ent * 28.0)
        g[2] = _safe_uint8(mean_v)
        g[3] = _safe_uint8(std_v)
        fill_len = min(prefix_arr.size, target_length - 4)
        g[4 : 4 + fill_len] = prefix_arr[:fill_len]

    # B 通道：会话长度统计 + 分段长度离散映射
    b = np.zeros(target_length, dtype=np.uint8)
    total_len = len(raw_session_bytes)
    if total_len:
        blocks = max(1, int(math.ceil(total_len / 32.0)))
        non_zero_ratio = int(100.0 * np.count_nonzero(r) / float(target_length))
        b[0] = _safe_uint8(min(total_len, 255))
        b[1] = _safe_uint8(blocks)
        b[2] = _safe_uint8(non_zero_ratio)
        b[3] = _safe_uint8(_entropy(r) * 28.0)

        lengths = []
        for idx in range(0, total_len, 32):
            lengths.append(min(255, len(raw_session_bytes[idx : idx + 32])))
        spread = _spread_values(lengths, target_length, start=4)
        b = np.maximum(b, spread)

    rgb = np.stack([r, g, b], axis=-1).reshape(image_size, image_size, 3)
    return rgb


def save_rgb_image(rgb_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    img.save(tmp_path, format="PNG")
    tmp_path.replace(output_path)
