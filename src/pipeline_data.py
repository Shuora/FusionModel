from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_policy_multimodal_data(processed_root: str | Path, policy: str) -> Dict[str, np.ndarray]:
    processed_root = Path(processed_root)
    rgbs: List[np.ndarray] = []
    token_ids_list: List[np.ndarray] = []
    attention_list: List[np.ndarray] = []
    segment_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    session_ids: List[np.ndarray] = []
    splits: List[np.ndarray] = []

    for dataset_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        policy_dir = dataset_dir / policy
        if not policy_dir.exists():
            continue

        rgb_files = sorted((policy_dir / "rgb").glob("rgb_shard_*.npz"))
        seq_files = sorted((policy_dir / "seq").glob("seq_shard_*.npz"))
        if not rgb_files or not seq_files:
            continue

        manifest_dir = policy_dir / "manifest"
        manifest_map = _load_manifest_split_map(manifest_dir)
        seq_map = _load_seq_map(seq_files)

        for rgb_file in rgb_files:
            rgb_npz = np.load(rgb_file, allow_pickle=False)
            rgb_sid = rgb_npz["session_id"]
            rgb_label = rgb_npz["label"].astype(np.int32)
            rgb_arr = rgb_npz["rgb"].astype(np.float32) / 255.0

            rgb_rows = []
            token_rows = []
            attn_rows = []
            seg_rows = []
            y_rows = []
            sid_rows = []
            split_rows = []
            for i in range(len(rgb_sid)):
                sid = str(rgb_sid[i])
                if sid not in seq_map:
                    continue
                token_ids, attention, segment = seq_map[sid]
                rgb_rows.append(rgb_arr[i])
                token_rows.append(token_ids.astype(np.int32))
                attn_rows.append(attention.astype(np.uint8))
                seg_rows.append(segment.astype(np.uint8))
                y_rows.append(int(rgb_label[i]))
                sid_rows.append(sid)
                split_rows.append(manifest_map.get(sid, "train"))

            if rgb_rows:
                rgbs.append(np.stack(rgb_rows, axis=0))
                token_ids_list.append(np.stack(token_rows, axis=0))
                attention_list.append(np.stack(attn_rows, axis=0))
                segment_list.append(np.stack(seg_rows, axis=0))
                labels.append(np.asarray(y_rows, dtype=np.int32))
                session_ids.append(np.asarray(sid_rows, dtype="U64"))
                splits.append(np.asarray(split_rows, dtype="U16"))

    if not rgbs:
        return {
            "rgb": np.zeros((0, 3, 28, 28), dtype=np.float32),
            "token_ids": np.zeros((0, 256), dtype=np.int32),
            "attention_mask": np.zeros((0, 256), dtype=np.uint8),
            "segment_ids": np.zeros((0, 256), dtype=np.uint8),
            "y": np.zeros((0,), dtype=np.int32),
            "session_id": np.zeros((0,), dtype="U64"),
            "split": np.zeros((0,), dtype="U16"),
        }

    return {
        "rgb": np.concatenate(rgbs, axis=0).astype(np.float32, copy=False),
        "token_ids": np.concatenate(token_ids_list, axis=0).astype(np.int32, copy=False),
        "attention_mask": np.concatenate(attention_list, axis=0).astype(np.uint8, copy=False),
        "segment_ids": np.concatenate(segment_list, axis=0).astype(np.uint8, copy=False),
        "y": np.concatenate(labels, axis=0).astype(np.int32, copy=False),
        "session_id": np.concatenate(session_ids, axis=0),
        "split": np.concatenate(splits, axis=0),
    }


def load_policy_data(processed_root: str | Path, policy: str) -> Dict[str, np.ndarray]:
    mm = load_policy_multimodal_data(processed_root, policy)
    if mm["rgb"].shape[0] == 0:
        return {
            "X": np.zeros((0, 1), dtype=np.float32),
            "y": mm["y"],
            "session_id": mm["session_id"],
            "split": mm["split"],
        }
    rgb_flat = mm["rgb"].reshape(mm["rgb"].shape[0], -1)
    token_part = mm["token_ids"][:, :64].astype(np.float32) / 8192.0
    mask_part = mm["attention_mask"][:, :64].astype(np.float32)
    seg_part = mm["segment_ids"][:, :64].astype(np.float32)
    X = np.concatenate([rgb_flat, token_part, mask_part, seg_part], axis=1).astype(np.float32, copy=False)
    return {
        "X": X,
        "y": mm["y"],
        "session_id": mm["session_id"],
        "split": mm["split"],
    }


def _load_seq_map(seq_files: List[Path]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    seq_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seq_file in seq_files:
        npz = np.load(seq_file, allow_pickle=False)
        sids = npz["session_id"]
        token_ids = npz["token_ids"]
        attention = npz["attention_mask"]
        segment = npz["segment_ids"]
        for i in range(len(sids)):
            sid = str(sids[i])
            seq_map[sid] = (token_ids[i], attention[i], segment[i])
    return seq_map


def _load_manifest_split_map(manifest_dir: Path) -> Dict[str, str]:
    csv_path = manifest_dir / "session_manifest.csv"
    parquet_path = manifest_dir / "session_manifest.parquet"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        return {}
    if "session_id" not in df.columns or "split" not in df.columns:
        return {}
    return {str(r["session_id"]): str(r["split"]) for _, r in df.iterrows()}
