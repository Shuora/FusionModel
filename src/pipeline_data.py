from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DATASET_ALIASES = {
    "ISCX-VPN-NonVPN-2016": "ISCX",
}


def _normalize_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(str(name), str(name))


def _resolve_label(label_mode: str, dataset_name: str, shard_label: int) -> int:
    if label_mode == "multiclass":
        return int(shard_label)
    if label_mode == "binary":
        return 0 if dataset_name == "ISCX" else 1
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def load_policy_multimodal_data(
    processed_root: str | Path,
    policy: str,
    datasets: Sequence[str] | None = None,
    label_mode: str = "multiclass",
    session_filter_manifest: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    processed_root = Path(processed_root)
    dataset_filter = {_normalize_dataset_name(x) for x in datasets} if datasets else None
    allowed_session_ids, session_filter_meta = _load_session_filter_manifest(session_filter_manifest)

    rgbs: List[np.ndarray] = []
    input_ids_list: List[np.ndarray] = []
    attention_list: List[np.ndarray] = []
    token_type_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    session_ids: List[np.ndarray] = []
    splits: List[np.ndarray] = []
    dataset_rows: List[np.ndarray] = []

    for dataset_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        default_dataset = _normalize_dataset_name(dataset_dir.name)
        if dataset_filter is not None and default_dataset not in dataset_filter:
            continue
        policy_dir = dataset_dir / policy
        if not policy_dir.exists():
            continue

        rgb_files = sorted((policy_dir / "rgb").glob("rgb_shard_*.npz"))
        etbert_files = sorted((policy_dir / "etbert").glob("etbert_shard_*.npz"))
        if not rgb_files or not etbert_files:
            continue

        manifest_dir = policy_dir / "manifest"
        manifest_map = _load_manifest_meta_map(manifest_dir, default_dataset=default_dataset)
        seq_map = _load_etbert_map(etbert_files)

        for rgb_file in rgb_files:
            rgb_npz = np.load(rgb_file, allow_pickle=False)
            rgb_sid = rgb_npz["session_id"]
            rgb_label = rgb_npz["label"].astype(np.int32)
            rgb_arr = rgb_npz["rgb"].astype(np.float32) / 255.0

            rgb_rows = []
            input_rows = []
            attn_rows = []
            type_rows = []
            y_rows = []
            sid_rows = []
            split_rows = []
            dataset_name_rows = []
            for i in range(len(rgb_sid)):
                sid = str(rgb_sid[i])
                if sid not in seq_map:
                    continue
                if allowed_session_ids is not None and sid not in allowed_session_ids:
                    continue
                meta = dict(manifest_map.get(sid, {}))
                if sid in session_filter_meta:
                    meta.update(session_filter_meta[sid])
                dataset_name = str(meta.get("dataset", default_dataset))
                if dataset_filter is not None and dataset_name not in dataset_filter:
                    continue
                input_ids, attention, token_type_ids = seq_map[sid]
                rgb_rows.append(rgb_arr[i])
                input_rows.append(input_ids.astype(np.int32))
                attn_rows.append(attention.astype(np.uint8))
                type_rows.append(token_type_ids.astype(np.uint8))
                y_rows.append(_resolve_label(label_mode=label_mode, dataset_name=dataset_name, shard_label=int(rgb_label[i])))
                sid_rows.append(sid)
                split_rows.append(str(meta.get("split", "train")))
                dataset_name_rows.append(dataset_name)

            if rgb_rows:
                rgbs.append(np.stack(rgb_rows, axis=0))
                input_ids_list.append(np.stack(input_rows, axis=0))
                attention_list.append(np.stack(attn_rows, axis=0))
                token_type_list.append(np.stack(type_rows, axis=0))
                labels.append(np.asarray(y_rows, dtype=np.int32))
                session_ids.append(np.asarray(sid_rows, dtype="U64"))
                splits.append(np.asarray(split_rows, dtype="U16"))
                dataset_rows.append(np.asarray(dataset_name_rows, dtype="U64"))

    if not rgbs:
        return {
            "rgb": np.zeros((0, 3, 28, 28), dtype=np.float32),
            "input_ids": np.zeros((0, 128), dtype=np.int32),
            "attention_mask": np.zeros((0, 256), dtype=np.uint8),
            "token_type_ids": np.zeros((0, 128), dtype=np.uint8),
            "y": np.zeros((0,), dtype=np.int32),
            "session_id": np.zeros((0,), dtype="U64"),
            "split": np.zeros((0,), dtype="U16"),
            "dataset": np.zeros((0,), dtype="U64"),
        }

    return {
        "rgb": np.concatenate(rgbs, axis=0).astype(np.float32, copy=False),
        "input_ids": np.concatenate(input_ids_list, axis=0).astype(np.int32, copy=False),
        "attention_mask": np.concatenate(attention_list, axis=0).astype(np.uint8, copy=False),
        "token_type_ids": np.concatenate(token_type_list, axis=0).astype(np.uint8, copy=False),
        "y": np.concatenate(labels, axis=0).astype(np.int32, copy=False),
        "session_id": np.concatenate(session_ids, axis=0),
        "split": np.concatenate(splits, axis=0),
        "dataset": np.concatenate(dataset_rows, axis=0),
    }


def load_policy_data(processed_root: str | Path, policy: str) -> Dict[str, np.ndarray]:
    mm = load_policy_multimodal_data(processed_root, policy)
    if mm["rgb"].shape[0] == 0:
        return {
            "X": np.zeros((0, 1), dtype=np.float32),
            "y": mm["y"],
            "session_id": mm["session_id"],
            "split": mm["split"],
            "dataset": mm["dataset"],
        }
    rgb_flat = mm["rgb"].reshape(mm["rgb"].shape[0], -1)
    token_part = mm["input_ids"][:, :64].astype(np.float32) / 30522.0
    mask_part = mm["attention_mask"][:, :64].astype(np.float32)
    seg_part = mm["token_type_ids"][:, :64].astype(np.float32)
    X = np.concatenate([rgb_flat, token_part, mask_part, seg_part], axis=1).astype(np.float32, copy=False)
    return {
        "X": X,
        "y": mm["y"],
        "session_id": mm["session_id"],
        "split": mm["split"],
        "dataset": mm["dataset"],
    }


def _load_etbert_map(seq_files: List[Path]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    seq_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seq_file in seq_files:
        npz = np.load(seq_file, allow_pickle=False)
        sids = npz["session_id"]
        input_ids = npz["input_ids"]
        attention = npz["attention_mask"]
        token_type_ids = npz["token_type_ids"]
        for i in range(len(sids)):
            sid = str(sids[i])
            seq_map[sid] = (input_ids[i], attention[i], token_type_ids[i])
    return seq_map


def _load_manifest_meta_map(manifest_dir: Path, default_dataset: str) -> Dict[str, Dict[str, str]]:
    csv_path = manifest_dir / "session_manifest.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "session_id" not in reader.fieldnames:
            return {}
        meta: Dict[str, Dict[str, str]] = {}
        for r in reader:
            sid = str(r["session_id"])
            dataset_name = _normalize_dataset_name(str(r.get("dataset", default_dataset)))
            family = str(r.get("family", ""))
            split = str(r.get("split", "train"))
            meta[sid] = {"dataset": dataset_name, "family": family, "split": split}
    return meta


def _load_session_filter_manifest(
    session_filter_manifest: str | Path | None,
) -> tuple[set[str] | None, Dict[str, Dict[str, str]]]:
    if not session_filter_manifest:
        return None, {}
    manifest_path = Path(session_filter_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"session filter manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() == ".parquet":
        raise ValueError("parquet session filter manifests require pandas and are not supported in this environment")
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "session_id" not in reader.fieldnames:
            return None, {}
        allowed = set()
        meta_overrides: Dict[str, Dict[str, str]] = {}
        for row in reader:
            sid = str(row["session_id"])
            allowed.add(sid)
            override: Dict[str, str] = {}
            split_value = str(row.get("split", "")).strip()
            if split_value:
                override["split"] = split_value
            dataset_value = str(row.get("dataset", "")).strip()
            if dataset_value:
                override["dataset"] = _normalize_dataset_name(dataset_value)
            family_value = str(row.get("family", "")).strip()
            if family_value:
                override["family"] = family_value
            if override:
                meta_overrides[sid] = override
        return allowed, meta_overrides
