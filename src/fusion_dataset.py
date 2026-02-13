"""Dataset utilities for CharBERT + MobileViT fusion training."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    CACHE_VERSION = 1

    def __init__(
        self,
        image_dir: str,
        pcap_dir: str,
        *,
        image_size: int = 28,
        char_seq_len: int = 786,
        r_only: bool = False,
        use_index_cache: bool = True,
        rebuild_index_cache: bool = False,
    ):
        self.image_dir = str(image_dir)
        self.pcap_dir = str(pcap_dir)
        self.image_size = int(image_size)
        self.char_seq_len = int(char_seq_len)
        self.r_only = bool(r_only)
        self.use_index_cache = bool(use_index_cache)
        self.rebuild_index_cache = bool(rebuild_index_cache)

        self.classes = sorted([d.name for d in Path(self.image_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        cache_path = self._cache_file_path(self.image_dir, self.pcap_dir)
        if self.use_index_cache and not self.rebuild_index_cache:
            payload = self._load_index_cache(cache_path)
            if payload is not None:
                self.classes = payload["classes"]
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                self.samples = [tuple(row) for row in payload["samples"]]
                self._refresh_label_stats()
                return

        self.samples: List[Tuple[str, str, int]] = []
        for cls in self.classes:
            img_cls_dir = Path(self.image_dir) / cls
            pcap_cls_dir = Path(self.pcap_dir) / cls
            if (not img_cls_dir.exists()) or (not pcap_cls_dir.exists()):
                continue

            pcap_files = [p.name for p in pcap_cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".bin", ".pcap", ".pcapng"}]
            exact_index: Dict[str, str] = {}
            norm_index: Dict[str, str] = {}
            for f in pcap_files:
                stem = Path(f).stem
                exact_index.setdefault(stem, f)
                norm_index.setdefault(self._normalize_stem(stem), f)

            for img in img_cls_dir.iterdir():
                if not img.is_file() or img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                stem = img.stem
                pcap_match = self._find_matching_pcap(stem, pcap_files, exact_index, norm_index)
                if pcap_match:
                    self.samples.append((str(img), str(pcap_cls_dir / pcap_match), self.class_to_idx[cls]))

        if self.use_index_cache:
            self._save_index_cache(cache_path)

        self._refresh_label_stats()

    def _refresh_label_stats(self) -> None:
        self.targets = [int(x[2]) for x in self.samples]
        self.class_counts = [0 for _ in self.classes]
        for t in self.targets:
            if 0 <= t < len(self.class_counts):
                self.class_counts[t] += 1

    @staticmethod
    def _normalize_stem(stem: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", stem.lower())

    @classmethod
    def _cache_file_path(cls, image_dir: str, pcap_dir: str) -> str:
        image_key = cls._normalize_stem(os.path.abspath(image_dir))
        pcap_key = cls._normalize_stem(os.path.abspath(pcap_dir))
        return str(Path(image_dir) / f".fusion_index_cache_v{cls.CACHE_VERSION}_{image_key[:28]}_{pcap_key[:28]}.json")

    def _load_index_cache(self, cache_path: str) -> Optional[dict]:
        try:
            cp = Path(cache_path)
            if not cp.exists() or not cp.is_file():
                return None
            with cp.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if int(payload.get("cache_version", -1)) != int(self.CACHE_VERSION):
                return None
            if payload.get("image_dir") != os.path.abspath(self.image_dir):
                return None
            if payload.get("pcap_dir") != os.path.abspath(self.pcap_dir):
                return None
            if not isinstance(payload.get("classes", None), list):
                return None
            if not isinstance(payload.get("samples", None), list):
                return None
            return payload
        except Exception:
            return None

    def _save_index_cache(self, cache_path: str) -> None:
        try:
            payload = {
                "cache_version": int(self.CACHE_VERSION),
                "image_dir": os.path.abspath(self.image_dir),
                "pcap_dir": os.path.abspath(self.pcap_dir),
                "classes": self.classes,
                "samples": self.samples,
            }
            cp = Path(cache_path)
            tmp = cp.with_suffix(cp.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.replace(cp)
        except Exception:
            pass

    @classmethod
    def _find_matching_pcap(
        cls,
        img_stem: str,
        pcap_files: Sequence[str],
        exact_index: Dict[str, str],
        norm_index: Dict[str, str],
    ) -> Optional[str]:
        if img_stem in exact_index:
            return exact_index[img_stem]

        norm = cls._normalize_stem(img_stem)
        if norm in norm_index:
            return norm_index[norm]

        for suffix in ("_img", "_image", "_pcap", "_bin", "_png", "_jpg", "_jpeg"):
            if img_stem.endswith(suffix):
                c = img_stem[: -len(suffix)]
                if c in exact_index:
                    return exact_index[c]
                cn = cls._normalize_stem(c)
                if cn in norm_index:
                    return norm_index[cn]

        for pcap_file in pcap_files:
            stem = Path(pcap_file).stem
            if img_stem in stem or stem in img_stem:
                return pcap_file

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32)
        if self.r_only:
            r = arr[:, :, 0:1]
            arr = np.concatenate([r, r, r], axis=2)

        arr = arr / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()

    def _load_pcap_tokens(self, pcap_path: str) -> torch.Tensor:
        max_len = int(self.char_seq_len)
        pad_id = 256
        cls_id = 257
        sep_id = 258

        if max_len < 2:
            return torch.tensor([cls_id], dtype=torch.long)

        with open(pcap_path, "rb") as f:
            raw = f.read(max_len - 2)

        ids = [cls_id] + list(raw) + [sep_id]
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        img_path, pcap_path, label = self.samples[idx]
        image = self._load_image(img_path)
        tokens = self._load_pcap_tokens(pcap_path)
        return image, tokens, int(label)


def _is_flat_layout(dataset_dir: Path) -> bool:
    req = [
        dataset_dir / "image_data" / "Train",
        dataset_dir / "image_data" / "Test",
        dataset_dir / "pcap_data" / "Train",
        dataset_dir / "pcap_data" / "Test",
    ]
    return all(p.exists() and p.is_dir() for p in req)


def resolve_dataset_dirs(dataset_root: str, dataset_name: str = "") -> Tuple[str, str, str, str, str]:
    root = Path(dataset_root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    if _is_flat_layout(root) and (not dataset_name or dataset_name == root.name):
        ds_dir = root
        resolved_name = root.name
    else:
        if not dataset_name:
            candidates = sorted([d.name for d in root.iterdir() if d.is_dir()])
            if not candidates:
                raise FileNotFoundError(f"No dataset directory under: {root}")
            dataset_name = candidates[0]

        ds_dir = root / dataset_name
        if not ds_dir.exists() or not ds_dir.is_dir():
            raise FileNotFoundError(f"dataset_name '{dataset_name}' not found under {root}")
        resolved_name = dataset_name

    train_img = ds_dir / "image_data" / "Train"
    train_pcap = ds_dir / "pcap_data" / "Train"
    test_img = ds_dir / "image_data" / "Test"
    test_pcap = ds_dir / "pcap_data" / "Test"

    for p in (train_img, train_pcap, test_img, test_pcap):
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Required dataset directory missing: {p}")

    return str(train_img), str(train_pcap), str(test_img), str(test_pcap), resolved_name
