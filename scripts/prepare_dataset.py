from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fusion_malicious.data.cache import write_cached_sample
from fusion_malicious.data.cleaning import anonymize_session_pcap, should_keep_session
from fusion_malicious.data.etbert_tokens import load_etbert_tokenizer, tokenize_session_bytes
from fusion_malicious.data.image_features import bytes_to_rgb_image
from fusion_malicious.data.manifest import build_manifest_dataframe
from fusion_malicious.data.records import SessionRecord
from fusion_malicious.data.session_bytes import normalize_session_bytes, read_session_bytes
from fusion_malicious.data.split import stratified_split_records
from fusion_malicious.data.splitcap import build_splitcap_command

TASK_DATASETS = {
    "binary": {"ISCX-VPN-NonVPN-2016", "MTA", "MFCP"},
    "mta": {"MTA"},
    "mfcp": {"MFCP"},
    "ustc": {"USTC-TFC2016"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cached multimodal samples from raw traffic captures.")
    parser.add_argument("--task", choices=sorted(TASK_DATASETS), required=True)
    parser.add_argument("--source-root", type=Path, default=repo_root / "SourceData")
    parser.add_argument("--output-root", type=Path, default=repo_root / "dataset")
    parser.add_argument("--splitcap-exe", type=Path, default=repo_root / "Tools" / "SplitCap.exe")
    parser.add_argument("--skip-splitcap", action="store_true")
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.2)
    return parser


def discover_capture_files(source_root: Path, task: str) -> list[Path]:
    allowed = TASK_DATASETS[task]
    files: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pcap", ".pcapng"}:
            continue
        if not any(part in allowed for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def run_splitcap(splitcap_exe: Path, input_pcap: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_splitcap_command(splitcap_exe, input_pcap, output_dir)
    subprocess.run(command, check=True)
    return sorted(output_dir.rglob("*.pcap"))


def collect_session_paths(
    raw_paths: list[Path],
    *,
    task: str,
    output_root: Path,
    splitcap_exe: Path,
    skip_splitcap: bool,
) -> list[Path]:
    if skip_splitcap:
        return raw_paths

    session_root = output_root / task / "sessions_raw"
    session_paths: list[Path] = []
    for raw_path in raw_paths:
        try:
            relative = raw_path.relative_to(repo_root / "SourceData")
        except ValueError:
            relative = Path(raw_path.name)
        target_dir = session_root / relative.with_suffix("")
        session_paths.extend(run_splitcap(splitcap_exe, raw_path, target_dir))
    return session_paths


def _task_root(output_root: Path, task: str) -> Path:
    return output_root / task


def build_cached_manifest(
    session_paths: list[Path],
    *,
    task: str,
    output_root: Path,
    tokenizer_model: str,
    tokenizer_max_length: int,
    seed: int,
) -> pd.DataFrame:
    tokenizer = load_etbert_tokenizer(tokenizer_model)
    manifest = build_manifest_dataframe(session_paths, task_name=task)
    task_root = _task_root(output_root, task)
    cleaned_root = task_root / "sessions_clean"
    cache_root = task_root / "cache"
    seen_payloads: set[str] = set()
    rows: list[dict[str, object]] = []

    for row in manifest.to_dict(orient="records"):
        source_path = Path(str(row["source_path"]))
        sample_id = str(row["sample_id"])
        cleaned_path = cleaned_root / f"{sample_id}.pcap"
        anonymize_session_pcap(source_path, cleaned_path, seed=seed)

        payload_bytes = read_session_bytes(cleaned_path)
        if not should_keep_session(payload_bytes, seen_payloads):
            continue

        normalized = normalize_session_bytes(payload_bytes, size=784).tobytes()
        image = bytes_to_rgb_image(normalized, size=784)
        token_text, input_ids, attention_mask = tokenize_session_bytes(
            normalized,
            tokenizer=tokenizer,
            max_length=tokenizer_max_length,
        )
        cache_path = cache_root / f"{sample_id}.npz"
        write_cached_sample(
            cache_path=cache_path,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=int(row["label_id"]),
            token_text=token_text,
        )

        row["cleaned_path"] = str(cleaned_path)
        row["cache_path"] = str(cache_path)
        rows.append(row)

    return pd.DataFrame(rows)


def write_split_manifests(
    frame: pd.DataFrame,
    *,
    task: str,
    output_root: Path,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> None:
    if frame.empty:
        raise ValueError("No usable samples were produced.")

    records = [
        SessionRecord(
            sample_id=str(row.sample_id),
            dataset=str(row.dataset),
            family=str(row.family),
            source_path=str(row.source_path),
            label_name=str(row.label_name),
            label_id=int(row.label_id),
        )
        for row in frame.itertuples()
    ]
    split = stratified_split_records(
        records,
        train_size=train_split,
        val_size=val_split,
        test_size=test_split,
        seed=seed,
    )

    task_root = _task_root(output_root, task)
    task_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(task_root / "manifest.csv", index=False)
    for subset_name, subset_records in split.items():
        subset_ids = {record.sample_id for record in subset_records}
        subset_frame = frame[frame["sample_id"].isin(subset_ids)].copy()
        subset_frame["subset"] = subset_name
        subset_frame.to_csv(task_root / f"{subset_name}.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()
    if not args.source_root.exists():
        print(f"SourceData directory not found at {args.source_root}; nothing to prepare.")
        return

    raw_paths = discover_capture_files(args.source_root, args.task)
    if not raw_paths:
        print(f"No capture files found for task {args.task} under {args.source_root}")
        return

    session_paths = collect_session_paths(
        raw_paths,
        task=args.task,
        output_root=args.output_root,
        splitcap_exe=args.splitcap_exe,
        skip_splitcap=args.skip_splitcap,
    )
    frame = build_cached_manifest(
        session_paths,
        task=args.task,
        output_root=args.output_root,
        tokenizer_model=args.tokenizer_model,
        tokenizer_max_length=args.tokenizer_max_length,
        seed=args.seed,
    )
    write_split_manifests(
        frame,
        task=args.task,
        output_root=args.output_root,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    print(
        f"Prepared {len(frame)} cached samples for task {args.task} under "
        f"{_task_root(args.output_root, args.task)}"
    )


if __name__ == "__main__":
    main()
