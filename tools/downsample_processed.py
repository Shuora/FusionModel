#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = BASE_DIR.parent / 'ProcessedData' / 'binary_benign_vs_malicious'


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def collect_samples(processed_root: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """Collect all .bin session files under processed_root/pcap_data/{Train,Test}/{label}.
    Returns mapping label -> list of (path, split).
    """
    samples: Dict[str, List[Tuple[Path, str]]] = {}
    for split in ('Train', 'Test'):
        split_dir = processed_root / 'pcap_data' / split
        if not split_dir.exists():
            logging.warning('Missing split dir: %s', split_dir)
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for entry in os.scandir(label_dir):
                if not entry.is_file():
                    continue
                if not entry.name.endswith('.bin'):
                    continue
                samples.setdefault(label, []).append((Path(entry.path), split))
    return samples


def collect_image_index(processed_root: Path) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """Build image lookup index once to avoid rescanning large label directories."""
    image_index: DefaultDict[str, DefaultDict[str, DefaultDict[str, List[Path]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for split in ('Train', 'Test'):
        split_dir = processed_root / 'image_data' / split
        if not split_dir.exists():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for entry in label_dir.iterdir():
                if entry.is_file():
                    image_index[split][label][entry.stem].append(entry)
    return {
        split: {
            label: dict(stem_map)
            for label, stem_map in label_map.items()
        }
        for split, label_map in image_index.items()
    }


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if copy:
            shutil.copy2(src, dst)
        else:
            # prefer hard link to avoid data duplication
            os.link(str(src), str(dst))
    except Exception as exc:
        logging.warning('Hardlink failed (%s), falling back to copy: %s -> %s', exc, src, dst)
        shutil.copy2(src, dst)


def find_related_image_files(
    image_index: Dict[str, Dict[str, Dict[str, List[Path]]]],
    split: str,
    label: str,
    stem: str,
) -> List[Path]:
    return list(image_index.get(split, {}).get(label, {}).get(stem, []))


def downsample_processed(
    processed_root: Path,
    dest_root: Path,
    fraction: float,
    seed: int,
    copy: bool = False,
    force: bool = False,
) -> Path:
    rng = random.Random(seed)
    processed_root = processed_root.resolve()
    dest_root = dest_root.resolve()

    if dest_root.exists():
        if not force:
            raise FileExistsError(f'destination exists: {dest_root}; use --force to overwrite')
        shutil.rmtree(dest_root)
    (dest_root / 'pcap_data').mkdir(parents=True, exist_ok=True)

    samples = collect_samples(processed_root)
    image_index = collect_image_index(processed_root)
    labels = sorted(samples.keys())
    
    if not samples:
        raise RuntimeError('No samples found in processed_root')

    manifest_rows = []
    total_linked = 0
    
    # We want to maintain relative split ratios, so we group by (split, label)
    grouped: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for label, items in samples.items():
        for path, split in items:
            grouped[(split, label)].append(path)
            
    for (split, label), items in sorted(grouped.items()):
        rng.shuffle(items)
        keep_n = max(1, int(len(items) * fraction))
        selected = items[:keep_n]
        
        logging.info('Split=%s Label=%s downsampling %s -> %s', split, label, len(items), len(selected))

        for src_path in selected:
            stem = src_path.stem
            relative_dir = dest_root / 'pcap_data' / split / label
            relative_dir.mkdir(parents=True, exist_ok=True)
            dst_bin = relative_dir / f"{stem}{src_path.suffix}"
            link_or_copy(src_path, dst_bin, copy=copy)

            # also link/copy sidecar json if present
            src_json = src_path.with_suffix('.json')
            if src_json.exists():
                dst_json = dst_bin.with_suffix('.json')
                link_or_copy(src_json, dst_json, copy=copy)
                try:
                    with open(dst_json, 'r', encoding='utf-8') as fh:
                        meta = json.load(fh)
                except Exception:
                    meta = {}
            else:
                meta = {}

            # attempt to link related image files
            related_images = find_related_image_files(image_index, split, label, stem)
            for img in related_images:
                dst_img_dir = dest_root / 'image_data' / split / label
                dst_img_dir.mkdir(parents=True, exist_ok=True)
                dst_img = dst_img_dir / img.name
                link_or_copy(img, dst_img, copy=copy)

            manifest_rows.append({
                'split': split,
                'label': label,
                'dataset_name': meta.get('dataset_name', ''),
                'raw_path': meta.get('raw_path', str(src_path)),
                'session_name': stem,
                'bin_path': str(dst_bin),
            })
            total_linked += 1

    metadata_dir = dest_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / 'manifest.json').write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding='utf-8')

    logging.info('Downsampled dataset written to %s; total_samples=%s', dest_root, total_linked)
    return dest_root


def build_parser():
    parser = argparse.ArgumentParser(description='Downsample an existing ProcessedData task.')
    parser.add_argument('--processed_root', default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument('--dest_root', default='')
    parser.add_argument('--fraction', type=float, default=0.1, help='Fraction of samples to keep (e.g., 0.1 for 10x reduction)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy', action='store_true', help='Copy files instead of creating hardlinks')
    parser.add_argument('--force', action='store_true', help='Remove dest_root if it exists')
    return parser


def main():
    setup_logging()
    args = build_parser().parse_args()
    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f'processed_root not found: {processed_root}')
    if args.dest_root:
        dest_root = Path(args.dest_root)
    else:
        dest_root = processed_root.parent / f"{processed_root.name}_downsampled_f{args.fraction}"

    downsample_processed(
        processed_root=processed_root,
        dest_root=dest_root,
        fraction=args.fraction,
        seed=args.seed,
        copy=args.copy,
        force=args.force
    )


if __name__ == '__main__':
    main()
