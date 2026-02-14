import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


SPECIAL_TOKENS = {
    "PAD": 256,
    "CLS": 257,
    "SEP": 258,
}


def _bytes_to_tensor(path: Path, max_len: int) -> torch.Tensor:
    data = path.read_bytes()
    # 转成整数列表并截断/填充
    arr = list(data[: max_len - 2])  # 预留 CLS/SEP
    arr = [SPECIAL_TOKENS["CLS"]] + arr + [SPECIAL_TOKENS["SEP"]]
    if len(arr) < max_len:
        arr += [SPECIAL_TOKENS["PAD"]] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.long)


class ByteDataset(Dataset):
    def __init__(self, root: Path, max_len: int, label_to_id: Dict[str, int]):
        self.samples: List[Tuple[Path, int]] = []
        root = Path(root)
        for label_dir in sorted(root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in label_to_id:
                continue
            label_id = label_to_id[label]
            for f in sorted(label_dir.glob("*.bin")):
                self.samples.append((f, label_id))

        if not self.samples:
            raise ValueError(f"在 {root} 下未找到样本")

        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        input_ids = _bytes_to_tensor(path, self.max_len)
        return input_ids, label


def build_label_map(train_dir: Path) -> Dict[str, int]:
    labels = [p.name for p in sorted(train_dir.iterdir()) if p.is_dir()]
    if not labels:
        raise ValueError(f"在 {train_dir} 未找到标签目录")
    return {label: i for i, label in enumerate(labels)}


def create_dataloaders(train_dir: Path, test_dir: Path, max_len: int, batch_size: int, num_workers: int, label_list: List[str] = None):
    if label_list:
        label_to_id = {l: i for i, l in enumerate(label_list)}
    else:
        label_to_id = build_label_map(train_dir)

    train_ds = ByteDataset(train_dir, max_len, label_to_id)
    test_ds = ByteDataset(test_dir, max_len, label_to_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    id_to_label = {v: k for k, v in label_to_id.items()}
    return train_loader, test_loader, label_to_id, id_to_label

