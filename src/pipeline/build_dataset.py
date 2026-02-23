from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from src.common.config import load_yaml
from src.common.io_utils import ensure_dir, write_json, write_npy


def build_dataset(cfg: Dict[str, Any]) -> Path:
    dataset_name = cfg["dataset_name"]
    output_root = Path(cfg.get("output_root", "dataset"))
    dataset_root = ensure_dir(output_root / dataset_name)

    image_dir = ensure_dir(dataset_root / "image_data")
    pcap_dir = ensure_dir(dataset_root / "pcap_data")

    samples: Iterable[Dict[str, Any]] = cfg.get("samples", [])
    for idx, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", f"sample_{idx:06d}"))
        image = np.zeros((28, 28, 3), dtype=np.float32)
        write_npy(image_dir / f"{sample_id}.npy", image)
        write_json(pcap_dir / f"{sample_id}.json", sample)

    return dataset_root


def main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Build paired TLS dataset artifacts")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    return build_dataset(cfg)


if __name__ == "__main__":
    main()
