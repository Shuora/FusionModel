from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fusion_malicious.data.manifest import build_manifest_dataframe


def main() -> None:
    source_root = repo_root / "SourceData"
    source_root = repo_root / "SourceData"
    splitcap_path = repo_root / "Tools" / "SplitCap.exe"
    if not source_root.exists():
        print(f"SourceData directory not found at {source_root}; nothing to prepare.")
        return

    print(f"Using SplitCap located at: {splitcap_path}")
    session_paths = sorted(source_root.rglob("*.pcap"))
    if not session_paths:
        print(f"No PCAP files found under {source_root}")
        return

    manifest = build_manifest_dataframe(session_paths, task_name="binary")
    dataset_dir = repo_root / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    target_path = dataset_dir / "binary_manifest.csv"
    manifest.to_csv(target_path, index=False)
    print(f"Wrote binary manifest with {len(manifest)} entries to {target_path}")


if __name__ == "__main__":
    main()
