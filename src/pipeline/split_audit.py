from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _read_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _source_from_sample_name(sample_path: Path) -> str:
    stem = sample_path.stem
    if "." in stem:
        return stem.split(".", 1)[0]
    return stem


def _list_class_names(dataset_dir: Path, split: str) -> List[str]:
    image_split = dataset_dir / "image_data" / split
    pcap_split = dataset_dir / "pcap_data" / split
    image_classes = {p.name for p in image_split.iterdir() if p.is_dir()} if image_split.exists() else set()
    pcap_classes = {p.name for p in pcap_split.iterdir() if p.is_dir()} if pcap_split.exists() else set()
    return sorted(image_classes | pcap_classes)


def _audit_split(
    dataset_dir: Path,
    split: str,
) -> Tuple[Dict[str, dict], Set[str], Dict[str, Set[str]], List[str]]:
    class_report: Dict[str, dict] = {}
    split_digests: Set[str] = set()
    class_digests: Dict[str, Set[str]] = {}
    warnings: List[str] = []

    for class_name in _list_class_names(dataset_dir, split):
        image_dir = dataset_dir / "image_data" / split / class_name
        pcap_dir = dataset_dir / "pcap_data" / split / class_name
        image_count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0
        bin_files = sorted(pcap_dir.glob("*.bin")) if pcap_dir.exists() else []
        pcap_count = len(bin_files)

        source_counter: Counter[str] = Counter()
        digest_set: Set[str] = set()
        for bin_file in bin_files:
            source_counter[_source_from_sample_name(bin_file)] += 1
            digest = _read_sha1(bin_file)
            digest_set.add(digest)
            split_digests.add(digest)

        if image_count != pcap_count:
            warnings.append(
                f"count_mismatch split={split} class={class_name} image={image_count} pcap={pcap_count}"
            )

        class_report[class_name] = {
            "image_count": image_count,
            "pcap_count": pcap_count,
            "unique_sha1_count": len(digest_set),
            "source_pcap_count": len(source_counter),
            "source_pcap_distribution": dict(sorted(source_counter.items())),
        }
        class_digests[class_name] = digest_set
    return class_report, split_digests, class_digests, warnings


def build_split_audit(dataset_dir: Path) -> dict:
    train_report, train_digests, train_class_digests, warnings_train = _audit_split(dataset_dir, "Train")
    test_report, test_digests, test_class_digests, warnings_test = _audit_split(dataset_dir, "Test")

    all_classes = sorted(set(train_class_digests) | set(test_class_digests))
    class_overlap = {
        class_name: len(train_class_digests.get(class_name, set()) & test_class_digests.get(class_name, set()))
        for class_name in all_classes
    }

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {
            "Train": train_report,
            "Test": test_report,
        },
        "cross_split_overlap": {
            "global_sha1_overlap": len(train_digests & test_digests),
            "by_class_sha1_overlap": class_overlap,
        },
        "warnings": warnings_train + warnings_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Train/Test split leakage for fusion dataset.")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory, e.g. dataset/USTC-TFC2016")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output json path. Defaults to <dataset_dir>/reports/split_audit_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

    report = build_split_audit(dataset_dir)
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        report_dir = dataset_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / f"split_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[split_audit] saved: {output_path}")
    print(f"[split_audit] global_sha1_overlap={report['cross_split_overlap']['global_sha1_overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
