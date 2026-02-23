from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np

from src.common.config import load_yaml
from src.common.io_utils import ensure_dir, write_json, write_npy
from src.common.logging_utils import build_file_logger

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


def _sample_id_from_path(path: Path, base_root: Path) -> str:
    rel = path.relative_to(base_root)
    return "__".join(rel.with_suffix("").parts)


def _discover_samples_from_source_root(source_root: Path) -> List[Dict[str, Any]]:
    patterns = ("*.pcap", "*.pcapng", "*.cap")
    pcap_files: List[Path] = []
    for pattern in patterns:
        pcap_files.extend(sorted(source_root.rglob(pattern)))

    samples: List[Dict[str, Any]] = []
    for pcap_path in pcap_files:
        samples.append(
            {
                "sample_id": _sample_id_from_path(pcap_path, source_root),
                "source_path": str(pcap_path),
            }
        )
    return samples


def _resolve_samples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_samples: Iterable[Dict[str, Any]] = cfg.get("samples", [])
    samples = [dict(x) for x in raw_samples]
    if samples:
        return samples

    source_root_value = cfg.get("source_root")
    if not source_root_value:
        return []

    source_root = Path(str(source_root_value))
    if not source_root.exists():
        return []

    return _discover_samples_from_source_root(source_root)


def _progress_items(
    items: Sequence[Any],
    desc: str,
    unit: str,
) -> Iterator[Any]:
    if tqdm is not None:
        yield from tqdm(items, total=len(items), desc=desc, unit=unit, dynamic_ncols=True)
        return

    total = len(items)
    for idx, item in enumerate(items, start=1):
        print(f"[{desc}] {idx}/{total}")
        yield item


def build_dataset(
    cfg: Dict[str, Any],
    logger: logging.Logger | None = None,
    show_progress: bool = False,
) -> tuple[Path, int]:
    dataset_name = cfg["dataset_name"]
    output_root = Path(cfg.get("output_root", "dataset"))
    dataset_root = ensure_dir(output_root / dataset_name)

    image_dir = ensure_dir(dataset_root / "image_data")
    pcap_dir = ensure_dir(dataset_root / "pcap_data")

    samples = _resolve_samples(cfg)
    if logger:
        logger.info("dataset_name=%s output_root=%s sample_count=%d", dataset_name, dataset_root, len(samples))

    sample_iter: Iterable[Dict[str, Any]]
    if show_progress:
        sample_iter = _progress_items(samples, desc=f"{dataset_name} samples", unit="sample")
    else:
        sample_iter = samples

    for idx, sample in enumerate(sample_iter):
        sample_id = str(sample.get("sample_id", f"sample_{idx:06d}"))
        image = np.zeros((28, 28, 3), dtype=np.float32)
        write_npy(image_dir / f"{sample_id}.npy", image)
        write_json(pcap_dir / f"{sample_id}.json", sample)
        if logger:
            logger.info("wrote sample_id=%s", sample_id)

    return dataset_root, len(samples)


def _build_one_config(cfg_path: str, show_progress: bool = True) -> Path:
    cfg = load_yaml(cfg_path)
    dataset_name = str(cfg.get("dataset_name", Path(cfg_path).stem))
    log_root = Path(str(cfg.get("log_root", "outputs")))
    log_path = log_root / "logs" / "preprocess" / f"{dataset_name}.log"
    logger = build_file_logger(log_path, name="fusion.preprocess")

    logger.info("start preprocessing config=%s", cfg_path)
    dataset_root, sample_count = build_dataset(cfg, logger=logger, show_progress=show_progress)
    logger.info("finish preprocessing dataset_root=%s sample_count=%d", dataset_root, sample_count)

    image_count = len(list((dataset_root / "image_data").glob("*.npy")))
    pcap_count = len(list((dataset_root / "pcap_data").glob("*.json")))
    print(f"[build_dataset] config={cfg_path}")
    print(f"[build_dataset] dataset={dataset_name} dataset_root={dataset_root}")
    print(f"[build_dataset] image_count={image_count} pcap_count={pcap_count}")
    print(f"[build_dataset] log_path={log_path}")
    if image_count == 0:
        print("[build_dataset][warn] no samples generated; check `samples` or `source_root`.")
    return dataset_root


def main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Build paired TLS dataset artifacts")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to dataset YAML config. Repeat this flag to process multiple datasets.",
    )
    args = parser.parse_args(argv)

    config_paths: List[str] = list(args.config)
    last_dataset_root: Path | None = None
    for cfg_path in _progress_items(config_paths, desc="datasets", unit="dataset"):
        last_dataset_root = _build_one_config(cfg_path, show_progress=True)

    if last_dataset_root is None:
        raise RuntimeError("No dataset config provided")
    return last_dataset_root


if __name__ == "__main__":
    main()
