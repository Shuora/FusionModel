"""
Shared utilities for CharBERT + MobileViT fusion experiments.
"""

from __future__ import annotations

import inspect
import json
import csv
import logging
import math
import os
import re
import sys
import copy
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Iterable, List, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:
    torch = None

    class _TorchModulePlaceholder:
        pass

    class _NNPlaceholder:
        Module = object

        def __getattr__(self, name):
            raise ModuleNotFoundError("torch is required for training functionality")

    nn = _NNPlaceholder()
    optim = None
    F = None
    DataLoader = object
    Dataset = object
    WeightedRandomSampler = object

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
except ModuleNotFoundError:
    def _missing_sklearn(*args, **kwargs):
        raise ModuleNotFoundError("scikit-learn is required for evaluation functionality")

    accuracy_score = classification_report = confusion_matrix = f1_score = _missing_sklearn

    class ConfusionMatrixDisplay:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _missing_sklearn()

try:
    from torchvision import transforms
except ModuleNotFoundError:
    transforms = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from transformers import MobileViTForImageClassification, MobileViTConfig
except ModuleNotFoundError:
    MobileViTForImageClassification = None
    MobileViTConfig = None

logger = logging.getLogger(__name__)
os.environ.setdefault("MPLBACKEND", "Agg")


def resolve_charbert_src() -> str:
    return str(Path(__file__).resolve().parent / 'CharBERT' / 'src')


def load_pyplot_headless():
    import matplotlib

    if "matplotlib.pyplot" in sys.modules:
        import matplotlib.pyplot as plt

        if "agg" not in str(plt.get_backend()).lower():
            plt.switch_backend("Agg")
        return plt

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _autocast_ctx(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def _make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def setup_logging(log_file: Optional[Union[str, os.PathLike]] = None, *, level: int = logging.INFO, force: bool = False) -> Path:
    logs_dir = Path(__file__).resolve().parent / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"train_{ts}.log"
    else:
        log_file = Path(log_file)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    if "force" in inspect.signature(logging.basicConfig).parameters:
        kwargs["force"] = force

    logging.basicConfig(**kwargs)
    return Path(log_file)


def set_seed(seed: int) -> None:
    try:
        import random

        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_available_datasets(dataset_root: Union[str, os.PathLike]) -> List[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {root}")
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def find_grouped_split_pairs(dataset_dir: Union[str, os.PathLike], split: str) -> List[Tuple[str, str]]:
    root = Path(dataset_dir)
    pairs: List[Tuple[str, str]] = []
    for item in sorted(root.iterdir(), key=lambda p: p.name):
        if not item.is_dir():
            continue
        # Preferred layout: <group>/<image_data|pcap_data>/<Train|Test>/<subclass>
        image_dir = item / "image_data" / split
        pcap_dir = item / "pcap_data" / split
        if image_dir.is_dir() and pcap_dir.is_dir():
            pairs.append((str(image_dir), str(pcap_dir)))
            continue
        # Backward-compatible grouped layout: <group>/<Train|Test>/<image_data|pcap_data>/<subclass>
        split_dir = item / split
        image_dir = split_dir / "image_data"
        pcap_dir = split_dir / "pcap_data"
        if image_dir.is_dir() and pcap_dir.is_dir():
            pairs.append((str(image_dir), str(pcap_dir)))
    return pairs


def parse_csv_values(value: str) -> List[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _is_flat_dataset_layout(dataset_dir: Union[str, os.PathLike]) -> bool:
    root = Path(dataset_dir)
    required_dirs = [
        root / "image_data" / "Train",
        root / "image_data" / "Test",
        root / "pcap_data" / "Train",
        root / "pcap_data" / "Test",
    ]
    return all(d.exists() and d.is_dir() for d in required_dirs)


def resolve_dataset_dirs(
    dataset_root: Union[str, os.PathLike],
    dataset_name: Optional[str] = None,
) -> tuple[str, str, str, str, str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {root}")

    # Flat layout support: dataset_root/{image_data,pcap_data}/{Train,Test}/...
    # Example: --dataset_root dataset2 (without --dataset_name)
    if _is_flat_dataset_layout(root) and (
        not dataset_name or dataset_name == root.name
    ):
        dataset_dir = root
        resolved_name = root.name
    else:
        available = list_available_datasets(root)
        if not available:
            raise FileNotFoundError(
                f"No dataset directories found under: {root}. "
                "If dataset_root is itself a dataset, ensure it contains "
                "image_data/Train, image_data/Test, pcap_data/Train, pcap_data/Test."
            )

        resolved_name = dataset_name if dataset_name else available[0]
        dataset_dir = root / resolved_name
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset '{resolved_name}' not found under {root}. "
                f"Available datasets: {available}. "
                f"For flat layout, use --dataset_root {root} and leave --dataset_name empty."
            )

    train_image_dir = dataset_dir / "image_data" / "Train"
    train_pcap_dir = dataset_dir / "pcap_data" / "Train"
    test_image_dir = dataset_dir / "image_data" / "Test"
    test_pcap_dir = dataset_dir / "pcap_data" / "Test"
    required_dirs = [train_image_dir, train_pcap_dir, test_image_dir, test_pcap_dir]
    missing = [str(d) for d in required_dirs if not d.exists() or not d.is_dir()]
    if missing:
        grouped_train_pairs = find_grouped_split_pairs(dataset_dir, "Train")
        grouped_test_pairs = find_grouped_split_pairs(dataset_dir, "Test")
        if grouped_train_pairs and grouped_test_pairs:
            root = str(dataset_dir)
            return (root, root, root, root, resolved_name)
        raise FileNotFoundError(
            "Dataset directory structure is incomplete for "
            f"'{resolved_name}'. Missing directories: {missing}"
        )

    return (
        str(train_image_dir),
        str(train_pcap_dir),
        str(test_image_dir),
        str(test_pcap_dir),
        resolved_name,
    )


def default_dirs() -> tuple[str, str, str, str]:
    base = Path(__file__).resolve().parent
    train_img, train_pcap, test_img, test_pcap, _ = resolve_dataset_dirs(base / "dataset")
    return train_img, train_pcap, test_img, test_pcap


def resolve_task_dataset_dirs(
    processed_root: Union[str, os.PathLike],
    task_name: str,
) -> tuple[str, str, str, str, str]:
    task_root = Path(processed_root) / task_name
    if not task_root.exists() or not task_root.is_dir():
        raise FileNotFoundError(f"task dataset root does not exist: {task_root}")

    train_image_dir = task_root / "image_data" / "Train"
    train_pcap_dir = task_root / "pcap_data" / "Train"
    test_image_dir = task_root / "image_data" / "Test"
    test_pcap_dir = task_root / "pcap_data" / "Test"
    required_dirs = [train_image_dir, train_pcap_dir, test_image_dir, test_pcap_dir]
    missing = [str(d) for d in required_dirs if not d.exists() or not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Task dataset structure is incomplete for {task_name}. Missing directories: {missing}"
        )

    return (
        str(train_image_dir),
        str(train_pcap_dir),
        str(test_image_dir),
        str(test_pcap_dir),
        task_name,
    )


def device_from_arg(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_run_output_dir(output_dir: Path, run_name: str) -> Path:
    ensure_output_dirs(output_dir)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(run_name).strip())
    base_name = safe_name or "run"

    run_dir = output_dir / base_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    suffix = 2
    while True:
        candidate = output_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def build_run_artifact_paths(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    return {
        "train_log": run_dir / "train.log",
        "metrics_json": run_dir / "metrics.json",
        "epoch_metrics_csv": run_dir / "epoch_metrics.csv",
        "metrics_curve": run_dir / "metrics_curve.png",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "attention_curve": run_dir / "attention_curve.png",
        "report_md": run_dir / "report.md",
        "model": run_dir / "fusion_model.pth",
        "base_model": run_dir / "fusion_model_base.pth",
    }


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _history_epoch_rows(history: dict) -> List[dict]:
    columns = ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]
    max_len = max((len(history.get(col, [])) for col in columns), default=0)
    rows: List[dict] = []
    for idx in range(max_len):
        row = {"epoch": idx + 1}
        for col in columns:
            seq = history.get(col, [])
            row[col] = seq[idx] if idx < len(seq) else ""
        rows.append(row)
    return rows


def export_metrics_artifacts(run_dir: Path, history: dict, metrics_payload: dict) -> Tuple[Path, Path]:
    paths = build_run_artifact_paths(run_dir)

    metrics_path = paths["metrics_json"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics_payload), f, ensure_ascii=False, indent=2)

    epoch_csv_path = paths["epoch_metrics_csv"]
    fieldnames = ["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]
    rows = _history_epoch_rows(history)
    with open(epoch_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return metrics_path, epoch_csv_path


def log_saved(logger_obj, path: Path, what: str) -> None:
    try:
        logger_obj.info("saved %s: %s (exists=%s)", what, path, path.exists())
    except Exception:
        pass


class EarlyStopping:
    """
    早停机制类
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> None:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict()) if self.restore_best_weights else None
            return

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)


def convert_grayscale_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    将单通道灰度图转换为三通道RGB图
    """
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


class FusionDataset(Dataset):
    """
    融合数据集类，同时加载图像数据和Pcap数据
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        image_dir: str,
        pcap_dir: str,
        transform=None,
        max_pcap_length: int = 784,
        *,
        use_index_cache: bool = True,
        rebuild_index_cache: bool = False,
    ):
        self.image_dir = image_dir
        self.pcap_dir = pcap_dir
        self.transform = transform
        self.max_pcap_length = max_pcap_length
        self.use_index_cache = bool(use_index_cache)
        self.rebuild_index_cache = bool(rebuild_index_cache)

        self.classes = sorted(
            [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        cache_path = self._cache_file_path(image_dir, pcap_dir)
        if self.use_index_cache and not self.rebuild_index_cache:
            cached = self._load_index_cache(cache_path)
            if cached is not None:
                self.classes = cached["classes"]
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
                self.samples = [tuple(row) for row in cached["samples"]]
                self._refresh_label_stats()
                logger.info("融合索引缓存命中: %s, 样本数: %s", cache_path, len(self.samples))
                return

        self.samples = []
        total_classes = len(self.classes)
        for class_idx, class_name in enumerate(self.classes):
            image_class_dir = os.path.join(image_dir, class_name)
            pcap_class_dir = os.path.join(pcap_dir, class_name)

            if os.path.exists(image_class_dir) and os.path.exists(pcap_class_dir):
                image_files = [
                    e.name
                    for e in os.scandir(image_class_dir)
                    if e.is_file() and e.name.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                pcap_files = [
                    e.name
                    for e in os.scandir(pcap_class_dir)
                    if e.is_file() and (e.name.lower().endswith((".bin", ".pcap")) or "pcap" in e.name.lower())
                ]

                pcap_exact_index = {}
                pcap_norm_index = {}
                for pcap_file in pcap_files:
                    pcap_base = os.path.splitext(pcap_file)[0]
                    pcap_exact_index.setdefault(pcap_base, pcap_file)
                    pcap_norm_index.setdefault(self._normalize_stem(pcap_base), pcap_file)

                for img_file in image_files:
                    img_base = os.path.splitext(img_file)[0]
                    matching_pcap = self._find_matching_pcap(
                        img_base=img_base,
                        pcap_files=pcap_files,
                        pcap_exact_index=pcap_exact_index,
                        pcap_norm_index=pcap_norm_index,
                    )

                    if matching_pcap:
                        self.samples.append(
                            (
                                os.path.join(image_class_dir, img_file),
                                os.path.join(pcap_class_dir, matching_pcap),
                                class_idx,
                            )
                        )

            logger.info("融合数据索引进度: %s/%s 类别 (%s)", class_idx + 1, total_classes, class_name)

        if self.use_index_cache:
            self._save_index_cache(cache_path)

        self._refresh_label_stats()
        logger.info("融合数据集加载完成，总样本数: %s", len(self.samples))

    def _refresh_label_stats(self) -> None:
        self.targets = [int(s[2]) for s in self.samples]
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
        filename = f".fusion_index_cache_v{cls.CACHE_VERSION}_{image_key[:28]}_{pcap_key[:28]}.json"
        return os.path.join(image_dir, filename)

    def _load_index_cache(self, cache_path: str) -> Optional[dict]:
        try:
            if not os.path.isfile(cache_path):
                return None
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if int(payload.get("cache_version", -1)) != int(self.CACHE_VERSION):
                return None
            if payload.get("image_dir") != os.path.abspath(self.image_dir):
                return None
            if payload.get("pcap_dir") != os.path.abspath(self.pcap_dir):
                return None
            classes = payload.get("classes", [])
            samples = payload.get("samples", [])
            if not isinstance(classes, list) or not isinstance(samples, list):
                return None
            return payload
        except Exception as e:
            logger.warning("读取融合索引缓存失败 %s: %s", cache_path, e)
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
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, cache_path)
            logger.info("融合索引缓存已保存: %s", cache_path)
        except Exception as e:
            logger.warning("保存融合索引缓存失败 %s: %s", cache_path, e)

    @classmethod
    def _find_matching_pcap(
        cls,
        img_base: str,
        pcap_files: List[str],
        pcap_exact_index: dict,
        pcap_norm_index: dict,
    ) -> Optional[str]:
        if img_base in pcap_exact_index:
            return pcap_exact_index[img_base]

        img_norm = cls._normalize_stem(img_base)
        if img_norm in pcap_norm_index:
            return pcap_norm_index[img_norm]

        for suffix in ("_img", "_image", "_png", "_jpg", "_jpeg", "_pcap", "_bin"):
            if img_base.endswith(suffix):
                candidate = img_base[: -len(suffix)]
                if candidate in pcap_exact_index:
                    return pcap_exact_index[candidate]
                candidate_norm = cls._normalize_stem(candidate)
                if candidate_norm in pcap_norm_index:
                    return pcap_norm_index[candidate_norm]

        for pcap_file in pcap_files:
            pcap_base = os.path.splitext(pcap_file)[0]
            if img_base in pcap_base or pcap_base in img_base:
                return pcap_file

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, pcap_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        pcap_data = self.load_pcap_data(pcap_path)

        return image, pcap_data, label

    def load_pcap_data(self, pcap_path: str) -> torch.Tensor:
        """
        加载和预处理pcap数据，并返回字节ID序列（LongTensor）。

        - 序列 = [CLS] + bytes[:max_len-2] + [SEP]
        - 不足 max_len 用 PAD 填充

        约定：byte id 取值 [0,255]；特殊 token：PAD=256, CLS=257, SEP=258
        """
        try:
            max_len = int(self.max_pcap_length)
            pad_token = 256
            cls_token = 257
            sep_token = 258
            if max_len < 2:
                return torch.tensor([cls_token], dtype=torch.long)

            # 只读取模型会用到的字节，避免大 pcap 文件整包读取导致 I/O 成为瓶颈。
            with open(pcap_path, "rb") as f:
                raw = f.read(max_len - 2)

            arr = [cls_token] + list(raw) + [sep_token]
            if len(arr) < max_len:
                arr.extend([pad_token] * (max_len - len(arr)))
            return torch.tensor(arr, dtype=torch.long)
        except Exception as e:
            logger.warning("读取pcap文件失败 %s: %s", pcap_path, e)
            max_len = int(self.max_pcap_length)
            pad_token = 256
            cls_token = 257
            sep_token = 258
            if max_len < 2:
                return torch.tensor([cls_token], dtype=torch.long)
            arr = [cls_token, sep_token] + [pad_token] * (max_len - 2)
            return torch.tensor(arr, dtype=torch.long)


class MergedFusionDataset(FusionDataset):
    def __init__(self, samples: List[Tuple[str, str, int]], classes: List[str], transform=None, max_pcap_length: int = 784):
        self.image_dir = ""
        self.pcap_dir = ""
        self.transform = transform
        self.max_pcap_length = max_pcap_length
        self.use_index_cache = False
        self.rebuild_index_cache = False
        self.classes = list(classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = list(samples)
        self._refresh_label_stats()


def compute_class_weights(class_counts: List[int], *, beta: float = 0.9999) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    valid = counts > 0
    if np.any(valid):
        effective_num = 1.0 - np.power(beta, counts[valid])
        effective_num = np.clip(effective_num, 1e-12, None)
        weights[valid] = (1.0 - beta) / effective_num
        weights_sum = weights[valid].sum()
        if weights_sum > 0:
            weights[valid] = weights[valid] * (valid.sum() / weights_sum)
    return torch.tensor(weights, dtype=torch.float32)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(max(label_smoothing, 0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def load_fusion_data(
    image_dir: str,
    pcap_dir: str,
    batch_size: int = 64,
    image_size: int = 28,
    max_pcap_length: int = 784,
    *,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    use_index_cache: bool = True,
    rebuild_index_cache: bool = False,
    is_train: bool = True,
    balance_mode: str = "none",
    selected_groups: Optional[List[str]] = None,
):
    logger.info("加载融合数据 - 图像目录: %s, Pcap目录: %s", image_dir, pcap_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(convert_grayscale_to_rgb),
        ]
    )

    grouped_pairs: List[Tuple[str, str]] = []
    if os.path.abspath(image_dir) == os.path.abspath(pcap_dir):
        split = "Train" if is_train else "Test"
        grouped_pairs = find_grouped_split_pairs(image_dir, split)
        all_group_names = [Path(img).parents[1].name for img, _ in grouped_pairs]
        if selected_groups:
            wanted = {g.lower() for g in selected_groups}
            filtered_pairs: List[Tuple[str, str]] = []
            for img_dir, p_dir in grouped_pairs:
                group_name = Path(img_dir).parents[1].name.lower()
                if group_name in wanted:
                    filtered_pairs.append((img_dir, p_dir))
            grouped_pairs = filtered_pairs
            if not grouped_pairs:
                raise FileNotFoundError(
                    f"--cic_group={selected_groups} 未匹配到任何分组。可用分组: {sorted(set(all_group_names))}"
                )

    if grouped_pairs:
        logger.info("检测到分层CIC结构，按 %s 聚合 %s 个大类目录", "Train" if is_train else "Test", len(grouped_pairs))
        if selected_groups:
            logger.info("启用大类过滤: %s", ",".join(selected_groups))
        datasets = [
            FusionDataset(
                img_dir,
                p_dir,
                transform,
                max_pcap_length,
                use_index_cache=use_index_cache,
                rebuild_index_cache=rebuild_index_cache,
            )
            for img_dir, p_dir in grouped_pairs
        ]
        all_class_names = sorted({cls for ds in datasets for cls in ds.classes})
        global_class_to_idx = {cls: idx for idx, cls in enumerate(all_class_names)}
        merged_samples: List[Tuple[str, str, int]] = []
        for ds in datasets:
            for img_path, p_path, local_idx in ds.samples:
                cls_name = ds.classes[int(local_idx)]
                merged_samples.append((img_path, p_path, global_class_to_idx[cls_name]))
        dataset = MergedFusionDataset(
            merged_samples,
            all_class_names,
            transform=transform,
            max_pcap_length=max_pcap_length,
        )
    else:
        dataset = FusionDataset(
            image_dir,
            pcap_dir,
            transform,
            max_pcap_length,
            use_index_cache=use_index_cache,
            rebuild_index_cache=rebuild_index_cache,
        )

    dl_kwargs = dict(batch_size=batch_size, shuffle=bool(is_train), num_workers=int(num_workers), pin_memory=bool(pin_memory))
    if int(num_workers) > 0:
        dl_kwargs["persistent_workers"] = bool(persistent_workers)
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    if is_train and balance_mode in ("weighted_sampler", "weighted_sampler_loss"):
        class_weights = compute_class_weights(dataset.class_counts)
        sample_weights = [float(class_weights[t]) for t in dataset.targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        dl_kwargs["sampler"] = sampler
        dl_kwargs["shuffle"] = False

    dataloader = DataLoader(dataset, **dl_kwargs)
    dataloader.class_counts = dataset.class_counts  # type: ignore[attr-defined]
    dataloader.classes = dataset.classes  # type: ignore[attr-defined]
    logger.info("融合数据加载完成，类别数: %s, 样本总数: %s", len(dataset.classes), len(dataset))
    return dataloader, dataset.classes


class CharBERTTextEncoder(nn.Module):
    """CharBERT text/byte sequence encoder.

    input:  x (LongTensor) shape (B, S) with byte ids in [0,255] and padding id.
    output: features (FloatTensor) shape (B, feature_dim)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        seq_len: int = 784,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len

        self.charbert = None
        self.pad_id = 256
        self.proj = None
        self.char_hidden_size = hidden_size

        try:
            charbert_src = resolve_charbert_src()
            if charbert_src not in sys.path:
                sys.path.insert(0, charbert_src)

            from model import build_model  # type: ignore
            from config import TrainingConfig  # type: ignore

            cfg = TrainingConfig()
            cfg.vocab_size = getattr(cfg, "vocab_size", 259)
            cfg.hidden_size = hidden_size
            cfg.num_layers = num_layers
            cfg.num_heads = num_heads
            cfg.dropout = dropout
            cfg.max_len = seq_len

            self.charbert = build_model(cfg, num_labels=feature_dim)
            self.pad_id = getattr(cfg, "pad_id", cfg.vocab_size - 3)

            self.char_hidden_size = getattr(cfg, "hidden_size", hidden_size)
            self.proj = nn.Linear(self.char_hidden_size, feature_dim)
        except Exception as e:
            logger.warning("CharBERT 不可用，使用降级特征提取: %s", e)
            self.charbert = None
            self.proj = nn.Linear(1, feature_dim)

    def encode_tokens(self, x: torch.Tensor):
        if self.charbert is None:
            return None, None

        attention_mask = (x != self.pad_id).long()
        if hasattr(self.charbert, "embedding") and hasattr(self.charbert, "encoder"):
            emb = self.charbert.embedding(x)
            if hasattr(self.charbert, "pos_encoder"):
                emb = self.charbert.pos_encoder(emb)
            pad_mask = (attention_mask == 0)
            enc = self.charbert.encoder(emb, src_key_padding_mask=pad_mask)
            return enc, pad_mask

        return None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.charbert is None:
            x_mean = x.float().mean(dim=1, keepdim=True)
            return self.proj(x_mean)

        attention_mask = (x != self.pad_id).long()

        enc, pad_mask = self.encode_tokens(x)
        if enc is not None:
            enc = enc * attention_mask.unsqueeze(-1).to(enc.dtype)
            denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(enc.dtype)
            pooled = enc.sum(dim=1) / denom
            return self.proj(pooled)

        try:
            out = self.charbert(x, attention_mask=attention_mask)
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, "logits"):
                return out.logits
            if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                return out[0]
        except Exception as e:
            logger.warning("CharBERT forward 失败，降级到均值投影: %s", e)

        x_mean = x.float().mean(dim=1, keepdim=True)
        return self.proj(x_mean)


class AttentionFusionModel(nn.Module):
    """Cross-attention fusion model."""

    def __init__(self, num_classes: int = 10, attention_dim: int = 256, char_hidden_size: int = 128):
        super().__init__()

        mv_cfg = MobileViTConfig()
        mobilevit_feature_dim = mv_cfg.neck_hidden_sizes[-1] if hasattr(mv_cfg, "neck_hidden_sizes") else 640
        mv_cfg.num_labels = mobilevit_feature_dim
        self.mobilevit = MobileViTForImageClassification(mv_cfg)
        self.mobilevit.classifier = nn.Linear(mobilevit_feature_dim, mobilevit_feature_dim)

        self.text_encoder = CharBERTTextEncoder(
            feature_dim=char_hidden_size,
            seq_len=784,
            hidden_size=char_hidden_size,
            num_layers=2,
            num_heads=4,
            dropout=0.3,
        )
        self.pad_id = getattr(self.text_encoder, "pad_id", 256)

        self.q_proj = nn.Linear(mobilevit_feature_dim, attention_dim)
        self.k_proj = nn.Linear(self.text_encoder.char_hidden_size, attention_dim)
        self.v_proj = nn.Linear(self.text_encoder.char_hidden_size, attention_dim)
        self.pcap_linear = nn.Linear(1, attention_dim)

        self.out = nn.Sequential(
            nn.Linear(mobilevit_feature_dim + attention_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.attention_dim = attention_dim

    def forward(self, images: torch.Tensor, pcap_ids: torch.Tensor, return_attention: bool = False):
        img_feats = self.mobilevit(images).logits
        attn_weights = None

        if self.text_encoder.charbert is not None:
            enc, pad_mask = self.text_encoder.encode_tokens(pcap_ids)
            if enc is not None and pad_mask is not None:
                Q = self.q_proj(img_feats).unsqueeze(1)
                K = self.k_proj(enc)
                V = self.v_proj(enc)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)
                scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                attended = torch.matmul(weights, V).squeeze(1)
                attn_weights = weights.squeeze(1)
            else:
                pcap_mean = pcap_ids.float().mean(dim=1, keepdim=True)
                attended = self.pcap_linear(pcap_mean)
        else:
            pcap_mean = pcap_ids.float().mean(dim=1, keepdim=True)
            attended = self.pcap_linear(pcap_mean)

        fused = torch.cat([img_feats, attended], dim=1)
        logits = self.out(fused)
        if return_attention:
            return logits, attn_weights
        return logits


def initialize_fusion_model(num_classes: int, fusion_mode: str = "attention", attention_dim: int = 256) -> nn.Module:
    logger.info("初始化融合模型，融合模式: %s", fusion_mode)
    if fusion_mode != "attention":
        raise ValueError(f"unsupported fusion mode: {fusion_mode}")
    if torch is None or MobileViTForImageClassification is None or MobileViTConfig is None:
        raise ModuleNotFoundError("torch and transformers are required for attention fusion training")
    model = AttentionFusionModel(num_classes=num_classes, attention_dim=attention_dim)
    logger.info("融合模型初始化完成，分类头设置为 %s 个类别", num_classes)
    return model


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
):
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_corrects = 0
    all_labels = []
    all_predictions = []

    use_amp = bool(use_amp and device.type == "cuda")
    non_blocking = bool(device.type == "cuda")
    with torch.no_grad():
        eval_progress = tqdm(data_loader, desc="评估", leave=False)
        for images, pcap_data, labels in eval_progress:
            images = images.to(device, non_blocking=non_blocking)
            pcap_data = pcap_data.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            with _autocast_ctx(device, use_amp):
                outputs = model(images, pcap_data)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_corrects += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            acc = 100.0 * val_corrects / max(val_total, 1)
            eval_progress.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

    epoch_val_loss = val_loss / max(len(data_loader.dataset), 1)
    accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0
    macro_f1 = f1_score(all_labels, all_predictions, average="macro") if all_labels else 0.0
    return epoch_val_loss, accuracy, macro_f1, all_labels, all_predictions


def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int = 7,
    use_amp: bool = True,
    class_balance: str = "none",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    early_stop_metric: str = "val_loss",
    early_stop_mode: str = "auto",
    lr_scheduler_mode: str = "none",
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    grad_clip_norm: float = 0.0,
    val_every: int = 1,
):
    logger.info("开始训练融合模型")
    logger.info(
        "训练参数 - Epochs: %s, Learning Rate: %s, Early Stopping Patience: %s", num_epochs, learning_rate, patience
    )
    logger.info(
        "训练策略 - class_balance: %s, loss_type: %s, focal_gamma: %.3f, weight_decay: %.6f, label_smoothing: %.4f, early_stop_metric: %s, lr_scheduler: %s, val_every: %s",
        class_balance,
        loss_type,
        focal_gamma,
        weight_decay,
        label_smoothing,
        early_stop_metric,
        lr_scheduler_mode,
        val_every,
    )
    logger.info(
        "DataLoader参数 - train_batches: %s, val_batches: %s, batch_size: %s, num_workers: %s, pin_memory: %s",
        len(train_loader),
        len(val_loader),
        getattr(train_loader, "batch_size", "unknown"),
        getattr(train_loader, "num_workers", "unknown"),
        getattr(train_loader, "pin_memory", "unknown"),
    )

    model.to(device)
    use_amp = bool(use_amp and device.type == "cuda")
    non_blocking = bool(device.type == "cuda")
    class_weights = None
    if class_balance in ("weighted_loss", "weighted_sampler_loss"):
        class_counts = getattr(train_loader, "class_counts", None)
        if class_counts:
            class_weights = compute_class_weights(class_counts).to(device)
            logger.info("使用 class-weighted CrossEntropyLoss")
        else:
            logger.warning("weighted_loss 已启用，但未获取到 class_counts，回退到普通 CrossEntropyLoss")
    if loss_type == "focal":
        criterion = FocalCrossEntropyLoss(
            gamma=float(focal_gamma),
            weight=class_weights,
            label_smoothing=float(max(label_smoothing, 0.0)),
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(max(label_smoothing, 0.0)))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(max(weight_decay, 0.0)))
    scaler = _make_grad_scaler(device, use_amp)
    mode = early_stop_mode
    if mode == "auto":
        mode = "min" if early_stop_metric == "val_loss" else "max"
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, mode=mode)

    scheduler = None
    if lr_scheduler_mode == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(lr_factor),
            patience=int(lr_patience),
            min_lr=float(min_lr),
        )
    elif lr_scheduler_mode == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(num_epochs), 1),
            eta_min=float(min_lr),
        )

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    for epoch in range(num_epochs):
        logger.info("Epoch %s/%s", epoch + 1, num_epochs)
        model.train()
        train_loss = 0.0
        train_total = 0
        train_corrects = 0
        train_labels = []
        train_preds = []

        train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
        for images, pcap_data, labels in train_progress:
            images = images.to(device, non_blocking=non_blocking)
            pcap_data = pcap_data.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            with _autocast_ctx(device, use_amp):
                outputs = model(images, pcap_data)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_corrects += (predicted == labels).sum().item()
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

            acc = 100.0 * train_corrects / max(train_total, 1)
            train_progress.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

        epoch_train_loss = train_loss / max(len(train_loader.dataset), 1)
        epoch_train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0.0
        epoch_train_f1 = f1_score(train_labels, train_preds, average="macro") if train_labels else 0.0

        run_validation = ((epoch + 1) % max(int(val_every), 1) == 0) or ((epoch + 1) == num_epochs)
        if run_validation:
            val_loss, val_acc, val_f1, _, _ = evaluate_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        else:
            val_loss, val_acc, val_f1 = float("nan"), float("nan"), float("nan")

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["train_f1"].append(epoch_train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if run_validation:
            logger.info(
                "Epoch %s 结果: 训练 Loss: %.4f, 训练 Acc: %.4f, 训练 F1: %.4f | 验证 Loss: %.4f, 验证 Acc: %.4f, 验证 F1: %.4f",
                epoch + 1,
                epoch_train_loss,
                epoch_train_acc,
                epoch_train_f1,
                val_loss,
                val_acc,
                val_f1,
            )

            if early_stop_metric == "val_acc":
                monitor_value = val_acc
            elif early_stop_metric == "val_f1":
                monitor_value = val_f1
            else:
                monitor_value = val_loss

            early_stopping(float(monitor_value), model)
            logger.info("早停计数器: %s/%s", early_stopping.counter, early_stopping.patience)

            if scheduler is not None:
                if lr_scheduler_mode == "reduce":
                    scheduler.step(float(monitor_value))
                else:
                    scheduler.step()
            logger.info("当前学习率: %.8f", optimizer.param_groups[0]["lr"])

            if early_stopping.early_stop:
                logger.info("早停机制触发，在第 %s 轮后停止训练", epoch + 1)
                break
        else:
            logger.info(
                "Epoch %s 结果: 训练 Loss: %.4f, 训练 Acc: %.4f, 训练 F1: %.4f | 跳过验证 (val_every=%s)",
                epoch + 1,
                epoch_train_loss,
                epoch_train_acc,
                epoch_train_f1,
                val_every,
            )
            if scheduler is not None and lr_scheduler_mode == "cosine":
                scheduler.step()
                logger.info("当前学习率: %.8f", optimizer.param_groups[0]["lr"])

    logger.info("融合模型训练完成")
    return model, history


def evaluate_full(model: nn.Module, data_loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    loss, acc, macro_f1, labels, preds = evaluate_epoch(
        model,
        data_loader,
        criterion,
        device,
        use_amp=(device.type == "cuda"),
    )
    report = classification_report(labels, preds, digits=4) if labels else ""
    cm = confusion_matrix(labels, preds) if labels else np.zeros((0, 0), dtype=int)
    per_class_f1 = f1_score(labels, preds, average=None) if labels else np.array([])
    return dict(
        loss=loss,
        acc=acc,
        macro_f1=macro_f1,
        report=report,
        cm=cm,
        per_class_f1=per_class_f1,
        labels=labels,
        preds=preds,
    )


def plot_training_curves(history: dict, path: Path, title: str) -> None:
    plt = load_pyplot_headless()

    epochs = range(1, len(history.get("train_acc", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history.get("train_acc", []), marker="o", label="Train Acc")
    axes[0].plot(epochs, history.get("val_acc", []), marker="o", label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Acc")
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_f1", []), marker="o", label="Train F1")
    axes[1].plot(epochs, history.get("val_f1", []), marker="o", label="Val F1")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: List[str], path: Path, title: str) -> None:
    plt = load_pyplot_headless()

    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax_cm, cmap="Blues", xticks_rotation=45)
    plt.title(title)
    fig_cm.savefig(path)
    plt.close(fig_cm)


def save_report_md(
    path: Path,
    *,
    title: str,
    acc: float,
    macro_f1: float,
    report: str,
    cm: np.ndarray,
    confusion_image: str,
    curve_image: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Test Accuracy:** {acc:.4f}\n\n")
        f.write(f"**Macro F1:** {macro_f1:.4f}\n\n")
        if report:
            f.write("**分类报告:**\n\n")
            f.write(report)
            f.write("\n\n")
        f.write("**混淆矩阵:**\n\n")
        f.write(str(cm))
        f.write("\n\n")
        if confusion_image:
            f.write(f"![Confusion Matrix]({confusion_image})\n")
        if curve_image:
            f.write(f"![Metrics Curve]({curve_image})\n")


def summarize_attention(attn: np.ndarray, pad_mask: Optional[np.ndarray] = None) -> dict:
    eps = 1e-12
    a = np.asarray(attn, dtype=np.float64)
    a = np.clip(a, eps, 1.0)

    nonpad = None
    pad = None
    if pad_mask is not None:
        pad = np.asarray(pad_mask, dtype=bool)
        nonpad = ~pad
        denom = (a * nonpad).sum(axis=1, keepdims=True)
        denom = np.clip(denom, eps, None)
        a_nonpad = (a * nonpad) / denom
    else:
        a_nonpad = a / np.clip(a.sum(axis=1, keepdims=True), eps, None)

    mean = float(a_nonpad.mean())
    mx = float(a_nonpad.max())
    mn = float(a_nonpad.min())

    safe_log = np.zeros_like(a_nonpad)
    np.log(a_nonpad, out=safe_log, where=a_nonpad > 0)
    ent = -(a_nonpad * safe_log).sum(axis=1)
    ent_mean = float(ent.mean())

    def topk_mass(k: int) -> float:
        kk = min(k, a_nonpad.shape[1])
        part = np.partition(a_nonpad, -kk, axis=1)[:, -kk:]
        return float(part.sum(axis=1).mean())

    out = dict(
        mean=mean,
        max=mx,
        min=mn,
        entropy=ent_mean,
        top1=topk_mass(1),
        top5=topk_mass(5),
        top10=topk_mass(10),
    )

    if nonpad is not None and pad is not None:
        out["nonpad_mass_mean"] = float((a * nonpad).sum(axis=1).mean())
        out["pad_mass_mean"] = float((a * pad).sum(axis=1).mean())
        out["nonpad_fraction_mean"] = float(nonpad.mean())

    return out


def collect_attention_diagnostics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    prefix: str,
    logger_obj,
    max_batches: int = 6,
    filename: Optional[str] = None,
) -> Optional[Path]:
    try:
        model.eval()
        attn_collect = []
        pad_collect = []
        seen = 0
        with torch.no_grad():
            for images, pcaps, _ in data_loader:
                images = images.to(device)
                pcaps = pcaps.to(device)
                try:
                    out = model(images, pcaps, return_attention=True)
                except TypeError:
                    return None
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    _, attn = out
                else:
                    attn = None
                if attn is None:
                    break
                attn_np = attn.detach().cpu().numpy()
                pad_mask = pcaps.detach().cpu().numpy() == getattr(model, "pad_id", 256)
                attn_collect.append(attn_np)
                pad_collect.append(pad_mask)
                seen += 1
                if seen >= max_batches:
                    break

        if not attn_collect:
            logger_obj.warning("[AttentionDiag] 未采集到注意力权重")
            return None

        attn_all = np.concatenate(attn_collect, axis=0)
        pad_all = np.concatenate(pad_collect, axis=0)
        stats = summarize_attention(attn_all, pad_all)
        logger_obj.info(
            "[AttentionDiag] mean=%.6f min=%.6f max=%.6f entropy=%.4f top1=%.4f top5=%.4f top10=%.4f nonpad_mass=%.4f pad_mass=%.4f nonpad_frac=%.4f",
            stats.get("mean", float("nan")),
            stats.get("min", float("nan")),
            stats.get("max", float("nan")),
            stats.get("entropy", float("nan")),
            stats.get("top1", float("nan")),
            stats.get("top5", float("nan")),
            stats.get("top10", float("nan")),
            stats.get("nonpad_mass_mean", float("nan")),
            stats.get("pad_mass_mean", float("nan")),
            stats.get("nonpad_fraction_mean", float("nan")),
        )

        eps = 1e-12
        nonpad = ~pad_all
        denom = (attn_all * nonpad).sum(axis=1, keepdims=True)
        denom = np.clip(denom, eps, None)
        a_nonpad = (attn_all * nonpad) / denom
        mean_curve = a_nonpad.mean(axis=0)

        plt = load_pyplot_headless()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(mean_curve)
        ax.set_title("Mean attention over pcap positions")
        ax.set_xlabel("Token index")
        ax.set_ylabel("Attention")
        attn_filename = filename or f"attention_curve_{prefix}.png"
        attn_fig_path = output_dir / attn_filename
        fig.tight_layout()
        fig.savefig(attn_fig_path)
        plt.close(fig)
        logger_obj.info("[AttentionDiag] saved attention curve: %s", attn_fig_path)
        return attn_fig_path
    except Exception as e:
        logger_obj.warning("[AttentionDiag] failed: %s", e)
        return None


def generate_meta_features(
    text_model: nn.Module,
    mobilevit_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    use_softmax: bool = True,
):
    text_model.eval()
    mobilevit_model.eval()
    meta_features = []
    meta_labels = []
    non_blocking = bool(device.type == "cuda")
    with torch.no_grad():
        for images, pcap_data, labels in tqdm(data_loader, desc="生成元特征"):
            images = images.to(device, non_blocking=non_blocking)
            pcap_data = pcap_data.to(device, non_blocking=non_blocking)

            text_logits = text_model(pcap_data)
            if isinstance(text_logits, (tuple, list)):
                text_logits = text_logits[0]
            if use_softmax:
                text_out = torch.softmax(text_logits, dim=1).cpu().numpy()
            else:
                text_out = text_logits.cpu().numpy()

            mobilevit_logits = mobilevit_model(images)
            if hasattr(mobilevit_logits, "logits"):
                mobilevit_logits = mobilevit_logits.logits
            if use_softmax:
                img_out = torch.softmax(mobilevit_logits, dim=1).cpu().numpy()
            else:
                img_out = mobilevit_logits.cpu().numpy()

            meta_features.append(np.concatenate([text_out, img_out], axis=1))
            meta_labels.append(labels.cpu().numpy())

    if meta_features:
        return np.concatenate(meta_features, axis=0), np.concatenate(meta_labels, axis=0)
    return np.array([]), np.array([])


def train_xgboost(meta_features: np.ndarray, meta_labels: np.ndarray):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost 未安装") from e

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    clf.fit(meta_features, meta_labels)
    return clf


def train_meta_learner(meta_features: np.ndarray, meta_labels: np.ndarray, method: str = "xgboost"):
    if method == "xgboost":
        return train_xgboost(meta_features, meta_labels)
    if method == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm 未安装") from e
        clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05)
        clf.fit(meta_features, meta_labels)
        return clf
    if method == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError("catboost 未安装") from e
        clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)
        clf.fit(meta_features, meta_labels)
        return clf
    if method == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=500, early_stopping=True)
        clf.fit(meta_features, meta_labels)
        return clf
    raise ValueError(f"Unknown meta-learner: {method}")


def parse_methods(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def add_common_args(p):
    base = Path(__file__).resolve().parent
    p.add_argument(
        "--dataset_root",
        default=str(base / "dataset"),
        help=(
            "Dataset root. Supports two layouts: "
            "(1) dataset/<dataset_name>/... ; "
            "(2) flat root with image_data/ and pcap_data/, e.g. dataset2"
        ),
    )
    p.add_argument(
        "--dataset_name",
        default="",
        help="Dataset folder name under dataset_root; empty means auto-select first by name",
    )
    p.add_argument(
        "--task_name",
        default="",
        help="ProcessedData task name, e.g. binary_benign_vs_malicious or ustc_multiclass",
    )
    p.add_argument(
        "--cic_group",
        default="",
        help="For grouped CIC dataset, choose one or more top-level groups, e.g. Adware or Adware,Ransomware",
    )

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=28)
    p.add_argument("--max_pcap_length", type=int, default=784)

    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--preset", choices=["none", "cic_balanced"], default="none")

    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision training")
    p.add_argument("--no_index_cache", action="store_true", help="Disable sample index cache")
    p.add_argument("--rebuild_index_cache", action="store_true", help="Force rebuild sample index cache")
    p.add_argument(
        "--class_balance",
        choices=["none", "weighted_loss", "weighted_sampler", "weighted_sampler_loss"],
        default="none",
        help="Class imbalance strategy used for train loader/loss",
    )
    p.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--early_stop_metric", choices=["val_loss", "val_acc", "val_f1"], default="val_loss")
    p.add_argument("--early_stop_mode", choices=["auto", "min", "max"], default="auto")
    p.add_argument("--lr_scheduler", choices=["none", "reduce", "cosine"], default="none")
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    p.add_argument("--val_every", type=int, default=1)

    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "outputs"))
    p.add_argument("--attention_dim", type=int, default=256)
    return p


def _arg_explicitly_set(flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in sys.argv[1:])


def _apply_preset_defaults(args, resolved_dataset_name: str) -> None:
    if getattr(args, "preset", "none") != "cic_balanced":
        return

    preset_values = {
        "--class_balance": "weighted_sampler_loss",
        "--loss_type": "focal",
        "--focal_gamma": 1.5,
        "--weight_decay": 1e-4,
        "--label_smoothing": 0.03,
        "--early_stop_metric": "val_f1",
        "--early_stop_mode": "max",
        "--lr_scheduler": "reduce",
        "--lr_patience": 2,
        "--lr_factor": 0.5,
        "--min_lr": 1e-6,
        "--grad_clip_norm": 1.0,
        "--val_every": 1,
    }
    for flag, value in preset_values.items():
        if not _arg_explicitly_set(flag):
            setattr(args, flag[2:], value)

    if resolved_dataset_name == "CICAndMal2017":
        cic_values = {
            "--lr": 3e-4,
            "--patience": 14,
            "--num_workers": 6,
            "--prefetch_factor": 2,
        }
        for flag, value in cic_values.items():
            if not _arg_explicitly_set(flag):
                setattr(args, flag[2:], value)


def build_common_kwargs(args):
    device = device_from_arg(args.device)
    set_seed(args.seed)
    if device.type == "cuda" and not args.pin_memory:
        args.pin_memory = True
    if int(args.num_workers) > 0 and not args.persistent_workers:
        args.persistent_workers = True
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    if getattr(args, "task_name", ""):
        train_image_dir, train_pcap_dir, test_image_dir, test_pcap_dir, resolved_dataset_name = resolve_task_dataset_dirs(
            args.dataset_root,
            args.task_name,
        )
    else:
        train_image_dir, train_pcap_dir, test_image_dir, test_pcap_dir, resolved_dataset_name = resolve_dataset_dirs(
            args.dataset_root,
            args.dataset_name or None,
        )
    _apply_preset_defaults(args, resolved_dataset_name)
    logger.info("Using dataset: %s (root=%s)", resolved_dataset_name, args.dataset_root)
    print(f"[Data] dataset={resolved_dataset_name}")
    print(f"[Data] dataset_root={args.dataset_root}")
    print(f"[Data] train_image_dir={train_image_dir}")
    print(f"[Data] train_pcap_dir={train_pcap_dir}")
    print(f"[Data] test_image_dir={test_image_dir}")
    print(f"[Data] test_pcap_dir={test_pcap_dir}")
    if args.cic_group:
        print(f"[Data] cic_group={args.cic_group}")

    return dict(
        train_image_dir=train_image_dir,
        train_pcap_dir=train_pcap_dir,
        test_image_dir=test_image_dir,
        test_pcap_dir=test_pcap_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_pcap_length=args.max_pcap_length,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        output_dir=Path(args.output_dir),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        attention_dim=args.attention_dim,
        use_amp=(not args.no_amp),
        use_index_cache=(not args.no_index_cache),
        rebuild_index_cache=args.rebuild_index_cache,
        class_balance=args.class_balance,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        early_stop_metric=args.early_stop_metric,
        early_stop_mode=args.early_stop_mode,
        lr_scheduler_mode=args.lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        val_every=args.val_every,
        selected_groups=parse_csv_values(args.cic_group) if args.cic_group else None,
    )


def make_tag(fusion_mode: str, attention_dim: int) -> str:
    if fusion_mode == "attention":
        return f"attention_dim{attention_dim}"
    return fusion_mode


def run_fusion_experiment(
    *,
    fusion_mode: str,
    train_image_dir: str,
    train_pcap_dir: str,
    test_image_dir: str,
    test_pcap_dir: str,
    batch_size: int,
    image_size: int,
    max_pcap_length: int,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    output_dir: Path,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    attention_dim: int = 256,
    use_amp: bool = True,
    use_index_cache: bool = True,
    rebuild_index_cache: bool = False,
    class_balance: str = "none",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    early_stop_metric: str = "val_loss",
    early_stop_mode: str = "auto",
    lr_scheduler_mode: str = "none",
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    grad_clip_norm: float = 0.0,
    val_every: int = 1,
    selected_groups: Optional[List[str]] = None,
) -> None:
    ensure_output_dirs(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = make_tag(fusion_mode, attention_dim)
    run_dir = prepare_run_output_dir(output_dir, f"{tag}_{ts}")
    artifact_paths = build_run_artifact_paths(run_dir)

    log_path = artifact_paths["train_log"]
    setup_logging(log_path, force=True)
    run_logger = logging.getLogger(f"run_{fusion_mode}")
    run_logger.info("start %s: output_root=%s run_dir=%s", fusion_mode, output_dir, run_dir)

    train_loader, train_classes = load_fusion_data(
        train_image_dir,
        train_pcap_dir,
        batch_size,
        image_size,
        max_pcap_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_index_cache=use_index_cache,
        rebuild_index_cache=rebuild_index_cache,
        is_train=True,
        balance_mode=class_balance,
        selected_groups=selected_groups,
    )
    test_loader, test_classes = load_fusion_data(
        test_image_dir,
        test_pcap_dir,
        batch_size,
        image_size,
        max_pcap_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_index_cache=use_index_cache,
        rebuild_index_cache=rebuild_index_cache,
        is_train=False,
        balance_mode="none",
        selected_groups=selected_groups,
    )

    assert train_classes == test_classes, "训练集和测试集类别不一致"
    num_classes = len(train_classes)

    model = initialize_fusion_model(num_classes, fusion_mode, attention_dim=attention_dim)
    model, history = train_fusion_model(
        model,
        train_loader,
        test_loader,
        epochs,
        lr,
        device,
        patience,
        use_amp=use_amp,
        class_balance=class_balance,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        early_stop_metric=early_stop_metric,
        early_stop_mode=early_stop_mode,
        lr_scheduler_mode=lr_scheduler_mode,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        min_lr=min_lr,
        grad_clip_norm=grad_clip_norm,
        val_every=val_every,
    )

    attention_curve_path: Optional[Path] = None
    if fusion_mode == "attention":
        attention_curve_path = collect_attention_diagnostics(
            model,
            test_loader,
            device,
            run_dir,
            prefix=f"{tag}_{ts}",
            logger_obj=run_logger,
            filename=artifact_paths["attention_curve"].name,
        )
        if attention_curve_path is not None:
            log_saved(run_logger, attention_curve_path, "attention_curve")

    eval_result = evaluate_full(model, test_loader, device)
    run_logger.info("评估结果:")
    run_logger.info("  准确率: %.4f", eval_result["acc"])
    run_logger.info("  Macro F1: %.4f", eval_result["macro_f1"])
    if eval_result["report"]:
        run_logger.info("  分类报告:\n%s", eval_result["report"])
    run_logger.info("  混淆矩阵:\n%s", eval_result["cm"])

    curve_path = artifact_paths["metrics_curve"]
    plot_training_curves(history, curve_path, title=f"Training Curves - {tag}")
    log_saved(run_logger, curve_path, "metrics_curve")

    cm_path = artifact_paths["confusion_matrix"]
    plot_confusion(eval_result["cm"], train_classes, cm_path, f"Confusion Matrix - {tag}")
    log_saved(run_logger, cm_path, "confusion_matrix")

    report_path = artifact_paths["report_md"]
    save_report_md(
        report_path,
        title=f"融合方式: {fusion_mode}",
        acc=eval_result["acc"],
        macro_f1=eval_result["macro_f1"],
        report=eval_result["report"],
        cm=eval_result["cm"],
        confusion_image=cm_path.name,
        curve_image=curve_path.name,
    )
    log_saved(run_logger, report_path, "report")

    model_path = artifact_paths["model"]
    torch.save(model.state_dict(), model_path)
    log_saved(run_logger, model_path, "model")

    metrics_payload = {
        "mode": fusion_mode,
        "tag": tag,
        "timestamp": ts,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "output_root": str(output_dir),
        "classes": train_classes,
        "history": history,
        "eval": {
            "loss": eval_result["loss"],
            "acc": eval_result["acc"],
            "macro_f1": eval_result["macro_f1"],
            "report": eval_result["report"],
            "confusion_matrix": eval_result["cm"],
            "per_class_f1": eval_result["per_class_f1"],
        },
        "artifacts": {
            "train_log": log_path.name,
            "metrics_curve": curve_path.name,
            "confusion_matrix": cm_path.name,
            "attention_curve": attention_curve_path.name if attention_curve_path else None,
            "report": report_path.name,
            "model": model_path.name,
        },
    }
    metrics_path, epoch_csv_path = export_metrics_artifacts(
        run_dir=run_dir,
        history=history,
        metrics_payload=metrics_payload,
    )
    log_saved(run_logger, metrics_path, "metrics_json")
    log_saved(run_logger, epoch_csv_path, "epoch_metrics_csv")

    run_logger.info("done %s: run_dir=%s log=%s", fusion_mode, run_dir, log_path)
    print(f"[{fusion_mode}] done. acc={eval_result['acc']:.4f}, run_dir={run_dir}, saved={model_path}")


def run_stacking_experiment(
    *,
    base_fusion_mode: str,
    meta_methods: Iterable[str],
    train_image_dir: str,
    train_pcap_dir: str,
    test_image_dir: str,
    test_pcap_dir: str,
    batch_size: int,
    image_size: int,
    max_pcap_length: int,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    output_dir: Path,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    attention_dim: int = 256,
    ensemble_tag: Optional[str] = None,
    use_amp: bool = True,
    use_index_cache: bool = True,
    rebuild_index_cache: bool = False,
    class_balance: str = "none",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    early_stop_metric: str = "val_loss",
    early_stop_mode: str = "auto",
    lr_scheduler_mode: str = "none",
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    grad_clip_norm: float = 0.0,
    val_every: int = 1,
    selected_groups: Optional[List[str]] = None,
) -> None:
    ensure_output_dirs(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_tag = make_tag(base_fusion_mode, attention_dim)
    ensemble_tag = ensemble_tag or f"{base_tag}_stacking"
    run_dir = prepare_run_output_dir(output_dir, f"{ensemble_tag}_{ts}")
    artifact_paths = build_run_artifact_paths(run_dir)

    log_path = artifact_paths["train_log"]
    setup_logging(log_path, force=True)
    run_logger = logging.getLogger(f"run_{ensemble_tag}")
    run_logger.info("start stacking: base=%s output_root=%s run_dir=%s", base_fusion_mode, output_dir, run_dir)
    methods = list(meta_methods)

    train_loader, train_classes = load_fusion_data(
        train_image_dir,
        train_pcap_dir,
        batch_size,
        image_size,
        max_pcap_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_index_cache=use_index_cache,
        rebuild_index_cache=rebuild_index_cache,
        is_train=True,
        balance_mode=class_balance,
        selected_groups=selected_groups,
    )
    test_loader, test_classes = load_fusion_data(
        test_image_dir,
        test_pcap_dir,
        batch_size,
        image_size,
        max_pcap_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_index_cache=use_index_cache,
        rebuild_index_cache=rebuild_index_cache,
        is_train=False,
        balance_mode="none",
        selected_groups=selected_groups,
    )

    assert train_classes == test_classes, "训练集和测试集类别不一致"
    num_classes = len(train_classes)

    model = initialize_fusion_model(num_classes, base_fusion_mode, attention_dim=attention_dim)
    model, history = train_fusion_model(
        model,
        train_loader,
        test_loader,
        epochs,
        lr,
        device,
        patience,
        use_amp=use_amp,
        class_balance=class_balance,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        early_stop_metric=early_stop_metric,
        early_stop_mode=early_stop_mode,
        lr_scheduler_mode=lr_scheduler_mode,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        min_lr=min_lr,
        grad_clip_norm=grad_clip_norm,
        val_every=val_every,
    )

    attention_curve_path: Optional[Path] = None
    if base_fusion_mode == "attention":
        attention_curve_path = collect_attention_diagnostics(
            model,
            test_loader,
            device,
            run_dir,
            prefix=f"{ensemble_tag}_{ts}",
            logger_obj=run_logger,
            filename=artifact_paths["attention_curve"].name,
        )
        if attention_curve_path is not None:
            log_saved(run_logger, attention_curve_path, "attention_curve")

    curve_path = artifact_paths["metrics_curve"]
    plot_training_curves(history, curve_path, title=f"Training Curves - {ensemble_tag}")
    log_saved(run_logger, curve_path, "metrics_curve")

    base_eval = evaluate_full(model, test_loader, device)
    cm_path = artifact_paths["confusion_matrix"]
    plot_confusion(base_eval["cm"], train_classes, cm_path, f"Confusion Matrix - {ensemble_tag}")
    log_saved(run_logger, cm_path, "confusion_matrix")

    report_path = artifact_paths["report_md"]
    save_report_md(
        report_path,
        title=f"融合方式: {base_fusion_mode}+stacking ({ensemble_tag})",
        acc=base_eval["acc"],
        macro_f1=base_eval["macro_f1"],
        report=base_eval["report"],
        cm=base_eval["cm"],
        confusion_image=cm_path.name,
        curve_image=curve_path.name,
    )
    log_saved(run_logger, report_path, "report")

    meta_features, meta_labels = generate_meta_features(model.text_encoder, model.mobilevit, train_loader, device)
    test_meta_features, test_meta_labels = generate_meta_features(model.text_encoder, model.mobilevit, test_loader, device)

    method_results = []
    for method in methods:
        try:
            meta_model = train_meta_learner(meta_features, meta_labels, method=method)
        except ImportError as e:
            run_logger.warning("跳过 %s: %s", method, e)
            method_results.append({"method": method, "skipped": True, "reason": str(e)})
            continue
        except Exception as e:
            run_logger.warning("训练 %s 失败: %s", method, e)
            method_results.append({"method": method, "failed": True, "reason": str(e)})
            continue

        preds = meta_model.predict(test_meta_features)
        acc = accuracy_score(test_meta_labels, preds) if len(test_meta_labels) else 0.0
        macro_f1 = f1_score(test_meta_labels, preds, average="macro") if len(test_meta_labels) else 0.0
        report = classification_report(test_meta_labels, preds, digits=4) if len(test_meta_labels) else ""
        cm = confusion_matrix(test_meta_labels, preds) if len(test_meta_labels) else np.zeros((0, 0), dtype=int)

        tag = f"{ensemble_tag}_{method}"
        run_logger.info("[%s] acc=%.4f macro_f1=%.4f", tag, acc, macro_f1)
        if report:
            run_logger.info("[%s] 分类报告:\n%s", tag, report)
        run_logger.info("[%s] 混淆矩阵:\n%s", tag, cm)

        method_cm_path = run_dir / f"confusion_matrix_{method}.png"
        plot_confusion(cm, train_classes, method_cm_path, f"Confusion Matrix - {tag}")
        log_saved(run_logger, method_cm_path, f"confusion_matrix_{method}")

        method_report_path = run_dir / f"report_{method}.md"
        save_report_md(
            method_report_path,
            title=f"融合方式: {base_fusion_mode}+stacking ({method})",
            acc=acc,
            macro_f1=macro_f1,
            report=report,
            cm=cm,
            confusion_image=method_cm_path.name,
            curve_image=curve_path.name,
        )
        log_saved(run_logger, method_report_path, f"report_{method}")

        meta_model_path = None
        try:
            import pickle

            meta_model_path = run_dir / f"meta_model_{method}.pkl"
            with open(meta_model_path, "wb") as f:
                pickle.dump(meta_model, f)
            log_saved(run_logger, meta_model_path, f"meta_model_{method}")
        except Exception as e:
            run_logger.warning("保存 %s 失败: %s", method, e)

        method_results.append(
            {
                "method": method,
                "acc": acc,
                "macro_f1": macro_f1,
                "report": report,
                "confusion_matrix": cm,
                "confusion_matrix_path": method_cm_path.name,
                "report_path": method_report_path.name,
                "meta_model_path": meta_model_path.name if meta_model_path else None,
            }
        )

    base_model_path = artifact_paths["base_model"]
    torch.save(model.state_dict(), base_model_path)
    log_saved(run_logger, base_model_path, "model_base")

    metrics_payload = {
        "mode": "attention_stacking",
        "base_fusion_mode": base_fusion_mode,
        "tag": ensemble_tag,
        "timestamp": ts,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "output_root": str(output_dir),
        "classes": train_classes,
        "history": history,
        "base_eval": {
            "loss": base_eval["loss"],
            "acc": base_eval["acc"],
            "macro_f1": base_eval["macro_f1"],
            "report": base_eval["report"],
            "confusion_matrix": base_eval["cm"],
            "per_class_f1": base_eval["per_class_f1"],
        },
        "meta_methods": methods,
        "method_results": method_results,
        "artifacts": {
            "train_log": log_path.name,
            "metrics_curve": curve_path.name,
            "confusion_matrix": cm_path.name,
            "attention_curve": attention_curve_path.name if attention_curve_path else None,
            "report": report_path.name,
            "base_model": base_model_path.name,
        },
    }
    metrics_path, epoch_csv_path = export_metrics_artifacts(
        run_dir=run_dir,
        history=history,
        metrics_payload=metrics_payload,
    )
    log_saved(run_logger, metrics_path, "metrics_json")
    log_saved(run_logger, epoch_csv_path, "epoch_metrics_csv")

    run_logger.info("done stacking: run_dir=%s log=%s", run_dir, log_path)
    print(f"[{ensemble_tag}] done. run_dir={run_dir}, saved_base={base_model_path}")
