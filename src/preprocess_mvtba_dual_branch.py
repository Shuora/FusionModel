"""Preprocess raw PCAP files into dual-branch inputs (RGB images + temporal sequences).

This script follows the MVTBA paper preprocessing sequence:
1) traffic segmentation by five-tuple session,
2) data cleaning (drop empty/duplicate sessions; anonymized session naming),
3) length unification to 784 bytes (trim or zero-pad),
4) format conversion to 28x28 image.

Outputs:
- image branch: 28x28 RGB `.png` files,
- temporal branch: fixed-length 784-step sequences as `.bin` / `.npy` / `.pt`.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


LOGGER = logging.getLogger("mvtba_dual_preprocess")

SESSION_LEN = 784
IMG_W = 28
IMG_H = 28
PCAP_EXTENSIONS = (".pcap", ".pcapng")
SUPPORTED_TEMPORAL_FORMATS = {"bin", "npy", "pt"}


@dataclass(frozen=True)
class SessionKey:
    proto: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int


@dataclass
class Sample:
    label: str
    src_pcap: Path
    session_idx: int
    session_key: SessionKey
    payload: bytes


def _load_dpkt():
    try:
        import dpkt  # type: ignore

        return dpkt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "dpkt is required to parse pcap/pcapng. Install it with: pip install dpkt"
        ) from exc


def _load_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required when temporal format includes 'pt'. Install it with: pip install torch"
        ) from exc




def _load_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "numpy is required for image/tensor conversion. Install it with: pip install numpy"
        ) from exc


def _load_pil_image():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for PNG export. Install it with: pip install pillow"
        ) from exc

def ip_to_str(ip_bytes: bytes) -> Optional[str]:
    try:
        return socket.inet_ntoa(ip_bytes)
    except OSError:
        return None


def iter_label_pcaps(dataset_root: Path) -> Iterator[Tuple[str, Path]]:
    """Iterate `<dataset_root>/<label>/*.pcap*` files."""
    for label_dir in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for file in sorted(label_dir.iterdir(), key=lambda p: p.name):
            if file.is_file() and file.suffix.lower() in PCAP_EXTENSIONS:
                yield label, file


def read_pcap_packets(pcap_path: Path) -> Iterable[bytes]:
    dpkt = _load_dpkt()
    with pcap_path.open("rb") as f:
        header = f.read(4)
        f.seek(0)
        if header == b"\x0a\x0d\x0d\x0a":
            reader = dpkt.pcapng.Reader(f)
        else:
            reader = dpkt.pcap.Reader(f)
        for _, buf in reader:
            yield buf


def extract_sessions(pcap_path: Path) -> Dict[SessionKey, bytearray]:
    dpkt = _load_dpkt()
    sessions: Dict[SessionKey, bytearray] = {}
    for buf in read_pcap_packets(pcap_path):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except (dpkt.UnpackError, ValueError):
            continue

        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue

        if isinstance(ip.data, dpkt.tcp.TCP):
            proto = "TCP"
            trans = ip.data
        elif isinstance(ip.data, dpkt.udp.UDP):
            proto = "UDP"
            trans = ip.data
        else:
            continue

        src_ip = ip_to_str(ip.src)
        dst_ip = ip_to_str(ip.dst)
        if not src_ip or not dst_ip:
            continue

        payload = bytes(trans.data or b"")
        if not payload:
            continue

        key = SessionKey(proto, src_ip, int(trans.sport), dst_ip, int(trans.dport))
        sessions.setdefault(key, bytearray()).extend(payload)

    return sessions


def unify_length(payload: bytes, session_len: int = SESSION_LEN) -> bytes:
    if len(payload) >= session_len:
        return payload[:session_len]
    return payload + (b"\x00" * (session_len - len(payload)))


def bytes_to_rgb_image(unified: bytes):
    np = _load_numpy()
    Image = _load_pil_image()
    arr = np.frombuffer(unified, dtype=np.uint8).reshape((IMG_H, IMG_W))
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def anonymized_stem(label: str, src_pcap: Path, idx: int, key: SessionKey, seed: int) -> str:
    text = f"{seed}|{label}|{src_pcap.name}|{idx}|{key.proto}|{key.src_port}|{key.dst_port}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{src_pcap.stem}_sess{idx:04d}_{digest}"


def build_samples(dataset_root: Path, deduplicate: bool = True) -> List[Sample]:
    samples: List[Sample] = []
    seen = set()

    for label, pcap_path in iter_label_pcaps(dataset_root):
        try:
            sessions = extract_sessions(pcap_path)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to parse %s: %s", pcap_path, exc)
            continue

        for idx, (key, payload_ba) in enumerate(sessions.items(), start=1):
            payload = bytes(payload_ba)
            if not payload:
                continue

            if deduplicate:
                fp = hashlib.sha1(payload).hexdigest()
                if fp in seen:
                    continue
                seen.add(fp)

            samples.append(
                Sample(
                    label=label,
                    src_pcap=pcap_path,
                    session_idx=idx,
                    session_key=key,
                    payload=payload,
                )
            )

    return samples


def split_by_pcap(samples: List[Sample], train_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    grouped: Dict[Tuple[str, Path], List[Sample]] = {}
    for s in samples:
        grouped.setdefault((s.label, s.src_pcap), []).append(s)

    keys = list(grouped.keys())
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * train_ratio)
    train_keys = set(keys[:cut])

    train, test = [], []
    for key, group in grouped.items():
        (train if key in train_keys else test).extend(group)
    return train, test


def parse_temporal_formats(s: str) -> List[str]:
    formats = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    invalid = [x for x in formats if x not in SUPPORTED_TEMPORAL_FORMATS]
    if invalid:
        raise ValueError(
            f"Invalid temporal formats: {invalid}. Supported: {sorted(SUPPORTED_TEMPORAL_FORMATS)}"
        )
    return sorted(set(formats))


def write_temporal_files(base_path: Path, unified: bytes, temporal_formats: List[str]) -> None:
    np = _load_numpy()
    arr = np.frombuffer(unified, dtype=np.uint8)

    if "bin" in temporal_formats:
        bin_path = base_path.with_suffix(".bin")
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with bin_path.open("wb") as f:
            f.write(unified)

    if "npy" in temporal_formats:
        npy_path = base_path.with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, arr)

    if "pt" in temporal_formats:
        torch = _load_torch()
        pt_path = base_path.with_suffix(".pt")
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.from_numpy(arr.copy())
        torch.save(tensor, pt_path)


def write_split(samples: List[Sample], output_root: Path, split: str, seed: int, temporal_formats: List[str]) -> None:
    for s in samples:
        stem = anonymized_stem(s.label, s.src_pcap, s.session_idx, s.session_key, seed)
        unified = unify_length(s.payload, SESSION_LEN)

        temporal_base = output_root / "temporal_data" / split / s.label / stem
        img_path = output_root / "image_data" / split / s.label / f"{stem}.png"

        write_temporal_files(temporal_base, unified, temporal_formats)

        img_path.parent.mkdir(parents=True, exist_ok=True)
        img = bytes_to_rgb_image(unified)
        tmp = img_path.with_suffix(".tmp.png")
        img.save(tmp, format="PNG")
        os.replace(tmp, img_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MVTBA-style dual-branch preprocessing")
    p.add_argument("--dataset_root", required=True, help="Input root: <root>/<label>/*.pcap[ng]")
    p.add_argument("--output_root", required=True, help="Output root directory")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio by source pcap")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_deduplicate", action="store_true", help="Disable session payload deduplication")
    p.add_argument(
        "--temporal_formats",
        default="bin,npy",
        help="Comma-separated temporal output formats: bin,npy,pt (default: bin,npy)",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_argparser().parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root does not exist or is not a directory: {dataset_root}")

    temporal_formats = parse_temporal_formats(args.temporal_formats)
    samples = build_samples(dataset_root, deduplicate=not args.no_deduplicate)
    if not samples:
        LOGGER.warning("No valid session samples extracted from %s", dataset_root)
        return

    train, test = split_by_pcap(samples, train_ratio=args.train_ratio, seed=args.seed)

    write_split(train, output_root, "Train", args.seed, temporal_formats)
    write_split(test, output_root, "Test", args.seed, temporal_formats)

    LOGGER.info(
        "Done. total=%d, train=%d, test=%d, temporal_formats=%s, output=%s",
        len(samples),
        len(train),
        len(test),
        ",".join(temporal_formats),
        output_root,
    )


if __name__ == "__main__":
    main()
