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

from itertools import product

from datetime import datetime

from pathlib import Path

from typing import Optional, Union, Iterable, List, Tuple, Dict, Callable, Any



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



    class ConfusionMatrixDisplay:                          

        def __init__(self, *args, **kwargs):

            _missing_sklearn()



try:

    from sklearn.metrics import log_loss, recall_score

except ModuleNotFoundError:

    def _missing_sklearn_extra(*args, **kwargs):

        raise ModuleNotFoundError("scikit-learn is required for evaluation functionality")



    log_loss = recall_score = _missing_sklearn_extra



try:

    from sklearn.model_selection import StratifiedKFold

except ModuleNotFoundError:

    StratifiedKFold = None



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



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"



MTA_CLASS_SIGNATURES = (

    frozenset(["dridex", "emotet", "hancitor", "qakbot", "trickbot", "ursnif"]),

    frozenset(["dridex", "emotet", "hancitor", "icedid", "qakbot", "trickbot", "ursnif"]),

)

MFCP_CLASS_SIGNATURES = (

    frozenset(["artemis", "dridex", "pua", "trickbot", "ursnif"]),

    frozenset(["artemis", "cobalt", "dridex", "pua", "trickbot", "ursnif"]),

    frozenset(["artemis", "cobalt", "dridex", "pua", "emotet", "ursnif"]),

)

MFCP_PAIR_CLASS_NAMES = ("artemis", "ursnif")

KNOWN_TASK_NAMES = {

    "binary_benign_vs_malicious",

    "ustc_multiclass",

    "mta_multiclass",

    "mfcp_multiclass",

}





def infer_task_name_from_data_dir(path: Optional[Union[str, os.PathLike]]) -> str:

    if not path:

        return ""

    try:

        parts = [str(p).lower() for p in Path(path).resolve().parts]

    except Exception:

        parts = [str(p).lower() for p in Path(path).parts]

    for part in parts:

        if part in KNOWN_TASK_NAMES:

            return part

    return ""





def detect_stacking_special_tasks(

    *,

    train_classes: List[str],

    train_image_dir: Optional[Union[str, os.PathLike]] = None,

    train_pcap_dir: Optional[Union[str, os.PathLike]] = None,

    explicit_task_name: str = "",

) -> Tuple[bool, bool]:

    class_names_lower = [str(c).strip().lower() for c in train_classes]

    class_set = frozenset(class_names_lower)

    task_hint = str(explicit_task_name or "").strip().lower()

    if not task_hint:

        task_hint = infer_task_name_from_data_dir(train_image_dir) or infer_task_name_from_data_dir(train_pcap_dir)



    is_mta_task = bool(task_hint == "mta_multiclass" or class_set in MTA_CLASS_SIGNATURES)

    is_mfcp_task = bool(task_hint == "mfcp_multiclass" or class_set in MFCP_CLASS_SIGNATURES)

    return is_mta_task, is_mfcp_task





def resolve_pair_class_indices(class_names: List[str], class_a_name: str, class_b_name: str) -> Optional[Tuple[int, int]]:

    name_to_idx = {str(name).strip().lower(): idx for idx, name in enumerate(class_names)}

    class_a = name_to_idx.get(str(class_a_name).strip().lower())

    class_b = name_to_idx.get(str(class_b_name).strip().lower())

    if class_a is None or class_b is None or class_a == class_b:

        return None

    return int(class_a), int(class_b)





def resolve_mfcp_pair_class_indices(class_names: List[str]) -> Optional[Tuple[int, int]]:

    return resolve_pair_class_indices(class_names, MFCP_PAIR_CLASS_NAMES[0], MFCP_PAIR_CLASS_NAMES[1])





def select_confusion_pair(

    *,

    labels: np.ndarray,

    preds: np.ndarray,

    preferred_pair: Optional[Tuple[int, int]] = None,

) -> Optional[Tuple[int, int]]:

    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)

    y_pred = np.asarray(preds, dtype=np.int64).reshape(-1)

    if y_true.size == 0 or y_pred.size != y_true.size:

        return None



    max_cls = int(max(np.max(y_true), np.max(y_pred)))

    num_classes = max_cls + 1

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))



    if preferred_pair is not None:

        pa, pb = int(preferred_pair[0]), int(preferred_pair[1])

        if 0 <= pa < num_classes and 0 <= pb < num_classes and pa != pb:

            if int(cm[pa, pb] + cm[pb, pa]) > 0:

                return (pa, pb)



    best_pair: Optional[Tuple[int, int]] = None

    best_confusion = 0

    for i in range(num_classes):

        for j in range(i + 1, num_classes):

            confusion_mass = int(cm[i, j] + cm[j, i])

            if confusion_mass > best_confusion:

                best_confusion = confusion_mass

                best_pair = (i, j)

    return best_pair





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

    logs_dir = DEFAULT_OUTPUT_ROOT / "logs"

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

                                                                                  

        image_dir = item / "image_data" / split

        pcap_dir = item / "pcap_data" / split

        if image_dir.is_dir() and pcap_dir.is_dir():

            pairs.append((str(image_dir), str(pcap_dir)))

            continue

                                                                                                    

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

    image_mode: str = "rgb",

) -> tuple[str, str, str, str, str]:

    task_root = Path(processed_root) / task_name

    if not task_root.exists() or not task_root.is_dir():

        raise FileNotFoundError(f"task dataset root does not exist: {task_root}")



    img_dir_name = "image_data" if image_mode == "rgb" else f"image_{image_mode}"

    train_image_dir = task_root / img_dir_name / "Train"

    train_pcap_dir = task_root / "pcap_data" / "Train"

    test_image_dir = task_root / img_dir_name / "Test"

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

        patience: int = 4,

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





def _resolve_early_stop_mode(early_stop_metric: str, early_stop_mode: str) -> str:

    metric_to_mode = {

        "val_loss": "min",

        "val_acc": "max",

        "val_f1": "max",

    }

    if early_stop_metric not in metric_to_mode:

        raise ValueError(f"Unsupported early_stop_metric: {early_stop_metric}")



    expected_mode = metric_to_mode[early_stop_metric]

    if early_stop_mode == "auto":

        return expected_mode

    if early_stop_mode != expected_mode:

        raise ValueError(

            f"early_stop_mode={early_stop_mode} 与 early_stop_metric={early_stop_metric} 不一致，"

            f"请使用 '{expected_mode}' 或 'auto'"

        )

    return early_stop_mode





def _select_monitor_value(early_stop_metric: str, val_loss: float, val_acc: float, val_f1: float) -> float:

    if early_stop_metric == "val_acc":

        return float(val_acc)

    if early_stop_metric == "val_f1":

        return float(val_f1)

    return float(val_loss)





def _has_non_finite_gradients(model: nn.Module) -> bool:

    for p in model.parameters():

        if p.grad is not None and not torch.isfinite(p.grad).all():

            return True

    return False





def _has_non_finite_parameters(model: nn.Module) -> bool:

    for p in model.parameters():

        if not torch.isfinite(p).all():

            return True

    return False





def _should_log_invalid_batch(counter: int) -> bool:

    return counter <= 20 or (counter % 200 == 0)





def convert_grayscale_to_rgb(x: torch.Tensor) -> torch.Tensor:

    """
    将单通道灰度图转换为三通道RGB图
    """

    if x.shape[0] == 1:

        return x.repeat(3, 1, 1)

    return x





def build_temporal_pcap_token_ids(

    packet_records: list[dict],

    max_pcap_length: int,

    *,

    pad_token: int = 256,

    cls_token: int = 257,

    sep_token: int = 258,

) -> list[int]:

    max_len = int(max_pcap_length)

    if max_len < 2:

        return [cls_token]



    token_ids: list[int] = [cls_token]

    previous_timestamp: float | None = None

    for packet in packet_records:

        try:

            timestamp = float(packet.get('timestamp', 0.0))

            direction = int(packet.get('direction', 0))

            packet_length = int(packet.get('packet_length', 0))

            payload_hex = str(packet.get('payload_hex', ''))

            payload = bytes.fromhex(payload_hex)

        except (TypeError, ValueError) as exc:

            logger.warning('跳过格式不合法的 packet 记录: %s', exc)

            continue



        delta_t = 0.0 if previous_timestamp is None else max(timestamp - previous_timestamp, 0.0)

        previous_timestamp = timestamp

        header = f'<PKT dir={direction} len={packet_length} dt={delta_t:.6f}>'.encode('ascii')

        footer = b'</PKT>'

        block = list(header) + list(payload) + list(footer)

        if len(token_ids) + len(block) + 1 > max_len:

            break

        token_ids.extend(block)



    token_ids.append(sep_token)

    if len(token_ids) < max_len:

        token_ids.extend([pad_token] * (max_len - len(token_ids)))

    return token_ids





_TEMPORAL_PACKET_HEADER_RE = re.compile(

    r"^<PKT dir=(?P<direction>\d+) len=(?P<length>\d+) dt=(?P<delta>[0-9]+(?:\.[0-9]+)?)>$"

)





def _decode_temporal_packet_header(header_tokens: list[int]) -> dict[str, float | int]:

    header_text = bytes(int(token) for token in header_tokens if 0 <= int(token) <= 255).decode(

        'ascii',

        errors='ignore',

    )

    match = _TEMPORAL_PACKET_HEADER_RE.match(header_text)

    if match:

        return {

            'direction': int(match.group('direction')),

            'packet_length': int(match.group('length')),

            'delta_t': float(match.group('delta')),

        }

    return {'direction': 0, 'packet_length': 0, 'delta_t': 0.0}





def _extract_temporal_packet_records(input_ids: list[int], pad_id: int) -> list[dict[str, int | float]]:

    header_prefix = [60, 80, 75, 84, 32]           

    footer = [60, 47, 80, 75, 84, 62]            

    records: list[dict[str, int | float]] = []

    idx = 0

    n = len(input_ids)



    while idx <= n - len(header_prefix):

        if input_ids[idx : idx + len(header_prefix)] != header_prefix:

            idx += 1

            continue



        header_start = idx

        header_end = None

        cursor = idx + len(header_prefix)

        while cursor < n:

            token = input_ids[cursor]

            if token == pad_id:

                break

            if token == 62:       

                header_end = cursor

                break

            cursor += 1



        if header_end is None:

            idx += 1

            continue



        payload_start = header_end + 1

        footer_start = None

        cursor = payload_start

        while cursor <= n - len(footer):

            if input_ids[cursor : cursor + len(footer)] == footer:

                footer_start = cursor

                break

            if input_ids[cursor] == pad_id:

                break

            cursor += 1



        if footer_start is None:

            idx += 1

            continue



        header_meta = _decode_temporal_packet_header(input_ids[header_start : header_end + 1])

        records.append(

            {

                'header_start': header_start,

                'header_end': header_end,

                'payload_start': payload_start,

                'payload_end': footer_start,

                'direction': header_meta['direction'],

                'packet_length': header_meta['packet_length'],

                'delta_t': header_meta['delta_t'],

            }

        )

        idx = footer_start + len(footer)



    return records





def _build_temporal_packet_meta_tensor(record: dict[str, int | float], *, device, dtype) -> torch.Tensor:

    direction = float(record.get('direction', 0))

    packet_length = float(record.get('packet_length', 0))

    delta_t = float(record.get('delta_t', 0.0))

    meta = torch.tensor(

        [

            direction,

            math.log1p(max(packet_length, 0.0)) / 10.0,

            math.log1p(max(delta_t, 0.0)) / 10.0,

        ],

        device=device,

        dtype=dtype,

    )

    return meta





def _find_temporal_packet_spans(input_ids: list[int], pad_id: int) -> list[tuple[int, int]]:

    header_prefix = [60, 80, 75, 84, 32]           

    footer = [60, 47, 80, 75, 84, 62]            

    spans: list[tuple[int, int]] = []

    idx = 0

    n = len(input_ids)

    while idx <= n - len(header_prefix):

        if input_ids[idx : idx + len(header_prefix)] != header_prefix:

            idx += 1

            continue



        header_end = None

        cursor = idx + len(header_prefix)

        while cursor < n:

            token = input_ids[cursor]

            if token == pad_id:

                break

            if token == 62:       

                header_end = cursor

                break

            cursor += 1



        if header_end is None:

            idx += 1

            continue



        payload_start = header_end + 1

        cursor = payload_start

        while cursor <= n - len(footer):

            if input_ids[cursor : cursor + len(footer)] == footer:

                if payload_start < cursor:

                    spans.append((payload_start, cursor))

                idx = cursor + len(footer)

                break

            if input_ids[cursor] == pad_id:

                idx = n

                break

            cursor += 1

        else:

            idx += 1

    return spans





def pool_temporal_packet_blocks(

    encoded: torch.Tensor,

    input_ids: torch.Tensor,

    pad_id: int,

    attention_mask: Optional[torch.Tensor] = None,

) -> Optional[torch.Tensor]:

    if encoded.ndim != 3 or input_ids.ndim != 2 or encoded.size(0) != input_ids.size(0):

        raise ValueError('encoded/input_ids batch dimensions do not match')



    device = encoded.device

    summaries: list[torch.Tensor] = []

    for batch_idx in range(encoded.size(0)):

        token_ids = input_ids[batch_idx].detach().cpu().tolist()

        spans = _find_temporal_packet_spans(token_ids, pad_id)

        packet_vecs: list[torch.Tensor] = []

        for start, end in spans:

            block = encoded[batch_idx, start:end]

            if attention_mask is not None:

                block_mask = attention_mask[batch_idx, start:end].to(dtype=encoded.dtype)

                block = block * block_mask.unsqueeze(-1)

                denom = block_mask.sum().clamp(min=1.0)

                packet_vec = block.sum(dim=0) / denom

            else:

                packet_vec = block.mean(dim=0)

            packet_vecs.append(packet_vec)



        if packet_vecs:

            summaries.append(torch.stack(packet_vecs, dim=0).mean(dim=0))

        else:

            summaries.append(encoded[batch_idx, 0])



    if not summaries:

        return None

    return torch.stack(summaries, dim=0).to(device=device, dtype=encoded.dtype)





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

        use_temporal_sidecar: bool = True,

        image_mode: str = "rgb",

    ):

        self.image_dir = image_dir

        self.pcap_dir = pcap_dir

        self.transform = transform

        self.max_pcap_length = max_pcap_length

        self.use_index_cache = bool(use_index_cache)

        self.rebuild_index_cache = bool(rebuild_index_cache)

        self.use_temporal_sidecar = bool(use_temporal_sidecar)

        self.image_mode = str(image_mode).lower()



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



        image = Image.open(image_path)

        if self.image_mode == "gray":

            image = image.convert("L")

        else:

            image = image.convert("RGB")



        if self.transform:

            image = self.transform(image)



        pcap_data = self.load_pcap_data(pcap_path)



        return image, pcap_data, label



    def load_pcap_data(self, pcap_path: str) -> torch.Tensor:

        """
        加载和预处理pcap数据，并返回字节ID序列（LongTensor）。

        - Legacy: 序列 = [CLS] + bytes[:max_len-2] + [SEP]
        - Hierarchical: 若存在同名 `.json` sidecar，则序列 = [CLS] + packet tag + payload + [/PKT] + ... + [SEP]
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



            if self.use_temporal_sidecar:

                meta_path = Path(pcap_path).with_suffix('.json')

                if meta_path.is_file():

                    try:

                        with meta_path.open('r', encoding='utf-8') as f:

                            payload = json.load(f)

                        if not isinstance(payload, dict):

                            raise ValueError('temporal sidecar payload must be a JSON object')

                        version = int(payload.get('version', 0))

                        if version != 1:

                            logger.warning('不支持的时序 sidecar 版本 %s，回退 legacy `.bin`', version)

                        else:

                            packets = payload.get('packets', [])

                            if isinstance(packets, list) and packets:

                                tokens = build_temporal_pcap_token_ids(

                                    packets,

                                    max_pcap_length=max_len,

                                    pad_token=pad_token,

                                    cls_token=cls_token,

                                    sep_token=sep_token,

                                )

                                return torch.tensor(tokens, dtype=torch.long)

                    except Exception as exc:

                        logger.warning('读取时序 sidecar 失败 %s: %s', meta_path, exc)



                                                     

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

    def __init__(self, samples: List[Tuple[str, str, int]], classes: List[str], transform=None, max_pcap_length: int = 784, use_temporal_sidecar: bool = True, image_mode: str = "rgb"):

        self.image_dir = ""

        self.pcap_dir = ""

        self.transform = transform

        self.max_pcap_length = max_pcap_length

        self.use_index_cache = False

        self.rebuild_index_cache = False

        self.use_temporal_sidecar = bool(use_temporal_sidecar)

        self.image_mode = str(image_mode).lower()

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

    use_temporal_sidecar: bool = True,

    image_mode: str = "rgb",

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

                use_temporal_sidecar=use_temporal_sidecar,

                image_mode=image_mode,

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

            use_temporal_sidecar=use_temporal_sidecar,

            image_mode=image_mode,

        )

    else:

        dataset = FusionDataset(

            image_dir,

            pcap_dir,

            transform,

            max_pcap_length,

            use_index_cache=use_index_cache,

            rebuild_index_cache=rebuild_index_cache,

            use_temporal_sidecar=use_temporal_sidecar,

            image_mode=image_mode,

        )



    dl_kwargs = dict(

        batch_size=batch_size,

        shuffle=bool(is_train),

        num_workers=int(num_workers),

        pin_memory=bool(pin_memory),

        drop_last=bool(is_train),

    )

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

    dataloader.class_counts = dataset.class_counts                              

    dataloader.classes = dataset.classes                              

    logger.info("融合数据加载完成，类别数: %s, 样本总数: %s", len(dataset.classes), len(dataset))

    return dataloader, dataset.classes





def build_deterministic_meta_loader(data_loader: DataLoader) -> DataLoader:

    """Build an evaluation-style loader from an existing loader's dataset.

    This removes train-time sampler/shuffle/drop_last effects so stacking meta
    features are generated exactly once per underlying sample.
    """



    dataset = data_loader.dataset

    batch_size = int(getattr(data_loader, "batch_size", 1) or 1)

    num_workers = int(getattr(data_loader, "num_workers", 0) or 0)

    pin_memory = bool(getattr(data_loader, "pin_memory", False))

    persistent_workers = bool(getattr(data_loader, "persistent_workers", False))



    kwargs: Dict[str, Any] = {

        "batch_size": batch_size,

        "shuffle": False,

        "num_workers": num_workers,

        "pin_memory": pin_memory,

        "drop_last": False,

    }

    collate_fn = getattr(data_loader, "collate_fn", None)

    if collate_fn is not None:

        kwargs["collate_fn"] = collate_fn

    if num_workers > 0:

        kwargs["persistent_workers"] = persistent_workers

        prefetch_factor = getattr(data_loader, "prefetch_factor", None)

        if prefetch_factor is not None:

            kwargs["prefetch_factor"] = int(prefetch_factor)



    deterministic_loader = DataLoader(dataset, **kwargs)

    if hasattr(data_loader, "class_counts"):

        deterministic_loader.class_counts = data_loader.class_counts                              

    if hasattr(data_loader, "classes"):

        deterministic_loader.classes = data_loader.classes                              

    return deterministic_loader





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

        charbert_mode: str = "charaware",

        char_vocab: str = "hex",

        char_emb_dim: int = 32,

        char_cnn_channels: int = 64,

        char_fusion: str = "gated",

        char_fusion_layers: str = "all",

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



            from model import build_model                

            from config import TrainingConfig                



            cfg = TrainingConfig()

            cfg.vocab_size = getattr(cfg, "vocab_size", 259)

            cfg.hidden_size = hidden_size

            cfg.num_layers = num_layers

            cfg.num_heads = num_heads

            cfg.dropout = dropout

            cfg.max_len = seq_len

            cfg.mode = charbert_mode

            cfg.char_vocab = char_vocab

            cfg.char_emb_dim = int(char_emb_dim)

            cfg.char_cnn_channels = int(char_cnn_channels)

            cfg.char_fusion = char_fusion

            cfg.char_fusion_layers = char_fusion_layers



            self.charbert = build_model(cfg, num_labels=feature_dim)

            self.pad_id = getattr(cfg, "pad_id", cfg.vocab_size - 3)



            self.char_hidden_size = getattr(cfg, "hidden_size", hidden_size)

            self.proj = nn.Linear(self.char_hidden_size, feature_dim)

        except Exception as e:

            logger.warning("CharBERT 不可用，使用降级特征提取: %s", e)

            self.charbert = None

            self.proj = nn.Linear(1, feature_dim)



        self.temporal_packet_proj = nn.Linear(self.char_hidden_size + 3, self.char_hidden_size)

        self.temporal_packet_activation = nn.GELU()

        self.temporal_packet_norm = nn.LayerNorm(self.char_hidden_size)

        temporal_packet_layer = nn.TransformerEncoderLayer(

            d_model=self.char_hidden_size,

            nhead=num_heads,

            dim_feedforward=self.char_hidden_size * 4,

            dropout=dropout,

            batch_first=True,

        )

        self.temporal_packet_encoder = nn.TransformerEncoder(temporal_packet_layer, num_layers=1)

        self.temporal_packet_pos_emb = nn.Embedding(max(2, seq_len + 1), self.char_hidden_size)



    def encode_tokens(self, x: torch.Tensor):

        if self.charbert is None:

            return None, None



        attention_mask = (x != self.pad_id).long()

        if hasattr(self.charbert, "encode_tokens"):

            enc, pad_mask = self.charbert.encode_tokens(x, attention_mask=attention_mask)

            return enc, pad_mask



        if hasattr(self.charbert, "embedding") and callable(getattr(self.charbert, "encoder", None)):

            emb = self.charbert.embedding(x)

            if hasattr(self.charbert, "pos_encoder"):

                emb = self.charbert.pos_encoder(emb)

            pad_mask = (attention_mask == 0)

            enc = self.charbert.encoder(emb, src_key_padding_mask=pad_mask)

            return enc, pad_mask



        return None, None



    def _encode_temporal_session_summary(

        self,

        encoded: torch.Tensor,

        input_ids: torch.Tensor,

        attention_mask: Optional[torch.Tensor] = None,

    ) -> Optional[torch.Tensor]:

        if encoded.ndim != 3 or input_ids.ndim != 2 or encoded.size(0) != input_ids.size(0):

            raise ValueError('encoded/input_ids batch dimensions do not match')



        summaries: list[torch.Tensor] = []

        for batch_idx in range(encoded.size(0)):

            token_ids = input_ids[batch_idx].detach().cpu().tolist()

            packet_records = _extract_temporal_packet_records(token_ids, self.pad_id)

            packet_vecs: list[torch.Tensor] = []

            for packet_idx, record in enumerate(packet_records):

                payload_start = int(record['payload_start'])

                payload_end = int(record['payload_end'])

                if payload_end <= payload_start:

                    continue



                block = encoded[batch_idx, payload_start:payload_end]

                if attention_mask is not None:

                    block_mask = attention_mask[batch_idx, payload_start:payload_end].to(dtype=encoded.dtype)

                    block = block * block_mask.unsqueeze(-1)

                    denom = block_mask.sum().clamp(min=1.0)

                    payload_vec = block.sum(dim=0) / denom

                else:

                    payload_vec = block.mean(dim=0)



                meta_vec = _build_temporal_packet_meta_tensor(record, device=encoded.device, dtype=encoded.dtype)

                packet_input = torch.cat([payload_vec, meta_vec], dim=-1).unsqueeze(0)

                packet_vec = self.temporal_packet_proj(packet_input)

                packet_vec = self.temporal_packet_activation(packet_vec)

                packet_vec = self.temporal_packet_norm(packet_vec)

                pos_idx = min(packet_idx, self.temporal_packet_pos_emb.num_embeddings - 1)

                packet_vec = packet_vec + self.temporal_packet_pos_emb(

                    torch.tensor([pos_idx], device=encoded.device, dtype=torch.long)

                )

                packet_vecs.append(packet_vec.squeeze(0))



            if packet_vecs:

                packet_seq = torch.stack(packet_vecs, dim=0).unsqueeze(0)

                packet_seq = self.temporal_packet_encoder(packet_seq)

                summaries.append(packet_seq.mean(dim=1).squeeze(0))

            else:

                                                                              

                                                                    

                summaries.append(encoded[batch_idx, 0])



        if not summaries:

            return None

        return torch.stack(summaries, dim=0)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.charbert is None:

            x_mean = x.float().mean(dim=1, keepdim=True)

            return self.proj(x_mean)



        attention_mask = (x != self.pad_id).long()



        enc, pad_mask = self.encode_tokens(x)

        if enc is not None:

            enc = enc * attention_mask.unsqueeze(-1).to(enc.dtype)

            cls_summary = enc[:, 0]

            packet_summary = self._encode_temporal_session_summary(

                enc,

                x,

                attention_mask=attention_mask,

            )

            if packet_summary is not None and packet_summary.size(0) == cls_summary.size(0):

                pooled = 0.5 * cls_summary + 0.5 * packet_summary

            else:

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

    """Fusion model supporting attention and concat modes."""



    def __init__(

        self,

        num_classes: int = 10,

        attention_dim: int = 256,

        char_hidden_size: int = 128,

        charbert_mode: str = "charaware",

        char_vocab: str = "hex",

        char_emb_dim: int = 32,

        char_cnn_channels: int = 64,

        char_fusion: str = "gated",

        char_fusion_layers: str = "all",

        fusion_mode: str = "attention",

    ):

        super().__init__()

        self.fusion_mode = fusion_mode



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

            charbert_mode=charbert_mode,

            char_vocab=char_vocab,

            char_emb_dim=char_emb_dim,

            char_cnn_channels=char_cnn_channels,

            char_fusion=char_fusion,

            char_fusion_layers=char_fusion_layers,

        )

        self.pad_id = getattr(self.text_encoder, "pad_id", 256)



        if self.fusion_mode == "attention":

            self.q_proj = nn.Linear(mobilevit_feature_dim, attention_dim)

            self.k_proj = nn.Linear(self.text_encoder.char_hidden_size, attention_dim)

            self.v_proj = nn.Linear(self.text_encoder.char_hidden_size, attention_dim)

            self.pcap_linear = nn.Linear(1, attention_dim)

            fusion_out_dim = mobilevit_feature_dim + attention_dim

        elif self.fusion_mode == "weighted":

            self.img_proj = nn.Linear(mobilevit_feature_dim, attention_dim)

            self.pcap_proj = nn.Linear(self.text_encoder.char_hidden_size, attention_dim)

            self.pcap_linear = nn.Linear(1, attention_dim)

            self.fusion_weights = nn.Parameter(torch.ones(2))

            fusion_out_dim = attention_dim

        else:

                         

            self.pcap_proj = nn.Linear(self.text_encoder.char_hidden_size, attention_dim)

            self.pcap_linear = nn.Linear(1, attention_dim)

            fusion_out_dim = mobilevit_feature_dim + attention_dim



        self.out = nn.Sequential(

            nn.Linear(fusion_out_dim, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes),

        )



        self.attention_dim = attention_dim



    def forward(self, images: torch.Tensor, pcap_ids: torch.Tensor, return_attention: bool = False):

        img_feats = self.mobilevit(images).logits

        attn_weights = None



        if self.text_encoder.charbert is not None:

            if self.fusion_mode == "attention":

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

                fused = torch.cat([img_feats, attended], dim=1)

            elif self.fusion_mode == "weighted":

                text_feats = self.text_encoder(pcap_ids)

                img_proj = self.img_proj(img_feats)

                text_proj = self.pcap_proj(text_feats)

                

                                                                     

                w = torch.softmax(self.fusion_weights, dim=0)

                fused = w[0] * img_proj + w[1] * text_proj

            else:

                             

                text_feats = self.text_encoder(pcap_ids)

                attended = self.pcap_proj(text_feats)

                fused = torch.cat([img_feats, attended], dim=1)

        else:

            pcap_mean = pcap_ids.float().mean(dim=1, keepdim=True)

            attended = self.pcap_linear(pcap_mean)

            if self.fusion_mode == "weighted":

                img_proj = self.img_proj(img_feats)

                w = torch.softmax(self.fusion_weights, dim=0)

                fused = w[0] * img_proj + w[1] * attended

            else:

                fused = torch.cat([img_feats, attended], dim=1)



        logits = self.out(fused)

        if return_attention:

            return logits, attn_weights

        return logits





def initialize_fusion_model(

    num_classes: int,

    fusion_mode: str = "attention",

    attention_dim: int = 256,

    charbert_mode: str = "charaware",

    char_vocab: str = "hex",

    char_emb_dim: int = 32,

    char_cnn_channels: int = 64,

    char_fusion: str = "gated",

    char_fusion_layers: str = "all",

) -> nn.Module:

    logger.info("初始化融合模型，融合模式: %s", fusion_mode)

    if fusion_mode not in ("attention", "concat", "weighted"):

        raise ValueError(f"unsupported fusion mode: {fusion_mode}")

    if torch is None or MobileViTForImageClassification is None or MobileViTConfig is None:

        raise ModuleNotFoundError("torch and transformers are required for attention fusion training")

    model = AttentionFusionModel(

        num_classes=num_classes,

        attention_dim=attention_dim,

        charbert_mode=charbert_mode,

        char_vocab=char_vocab,

        char_emb_dim=char_emb_dim,

        char_cnn_channels=char_cnn_channels,

        char_fusion=char_fusion,

        char_fusion_layers=char_fusion_layers,

        fusion_mode=fusion_mode,

    )

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

    patience: int = 4,

    use_amp: bool = True,

    class_balance: str = "none",

    loss_type: str = "ce",

    focal_gamma: float = 2.0,

    weight_decay: float = 1e-4,

    label_smoothing: float = 0.0,

    early_stop_metric: str = "val_loss",

    early_stop_mode: str = "auto",

    lr_scheduler_mode: str = "reduce",

    lr_patience: int = 2,

    lr_factor: float = 0.5,

    min_lr: float = 1e-6,

    grad_clip_norm: float = 1.0,

    val_every: int = 1,

    max_consecutive_invalid_batches: int = 128,

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

    logger.info("训练保护 - grad_clip_norm: %.3f, max_consecutive_invalid_batches: %s", grad_clip_norm, max_consecutive_invalid_batches)

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

    mode = _resolve_early_stop_mode(early_stop_metric, early_stop_mode)

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

    health = {

        "run_status": "ok",

        "stop_reason": "completed",

        "invalid_loss_batches": 0,

        "invalid_grad_batches": 0,

        "invalid_param_events": 0,

        "processed_train_batches": 0,

        "skipped_train_batches": 0,

    }

    max_consecutive_invalid_batches = max(int(max_consecutive_invalid_batches), 1)

    stop_training_now = False



    for epoch in range(num_epochs):

        logger.info("Epoch %s/%s", epoch + 1, num_epochs)

        model.train()

        train_loss = 0.0

        train_total = 0

        train_corrects = 0

        train_labels = []

        train_preds = []

        processed_train_samples = 0

        consecutive_invalid_batches = 0

        invalid_stop_reason = ""



        train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")

        for batch_idx, (images, pcap_data, labels) in enumerate(train_progress):

            images = images.to(device, non_blocking=non_blocking)

            pcap_data = pcap_data.to(device, non_blocking=non_blocking)

            labels = labels.to(device, non_blocking=non_blocking)



            optimizer.zero_grad()

            with _autocast_ctx(device, use_amp):

                outputs = model(images, pcap_data)

                if isinstance(outputs, (tuple, list)):

                    outputs = outputs[0]

                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):

                health["invalid_loss_batches"] += 1

                health["skipped_train_batches"] += 1

                consecutive_invalid_batches += 1

                if _should_log_invalid_batch(health["invalid_loss_batches"]):

                    logger.warning(

                        "训练损失无效（NaN/Inf），跳过该 batch: epoch=%s batch=%s/%s",

                        epoch + 1,

                        batch_idx + 1,

                        len(train_loader),

                    )

                if consecutive_invalid_batches >= max_consecutive_invalid_batches:

                    invalid_stop_reason = f"consecutive_invalid_batches(loss)>={max_consecutive_invalid_batches}"

                    logger.error(

                        "连续无效 batch 达到阈值，终止训练: epoch=%s batch=%s/%s reason=%s",

                        epoch + 1,

                        batch_idx + 1,

                        len(train_loader),

                        invalid_stop_reason,

                    )

                    stop_training_now = True

                    break

                continue

            if use_amp:

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                if grad_clip_norm and grad_clip_norm > 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()

                if grad_clip_norm and grad_clip_norm > 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

                if _has_non_finite_gradients(model):

                    health["invalid_grad_batches"] += 1

                    health["skipped_train_batches"] += 1

                    consecutive_invalid_batches += 1

                    optimizer.zero_grad(set_to_none=True)

                    if _should_log_invalid_batch(health["invalid_grad_batches"]):

                        logger.warning(

                            "梯度无效（NaN/Inf），跳过该 batch: epoch=%s batch=%s/%s",

                            epoch + 1,

                            batch_idx + 1,

                            len(train_loader),

                        )

                    if consecutive_invalid_batches >= max_consecutive_invalid_batches:

                        invalid_stop_reason = f"consecutive_invalid_batches(grad)>={max_consecutive_invalid_batches}"

                        logger.error(

                            "连续无效 batch 达到阈值，终止训练: epoch=%s batch=%s/%s reason=%s",

                            epoch + 1,

                            batch_idx + 1,

                            len(train_loader),

                            invalid_stop_reason,

                        )

                        stop_training_now = True

                        break

                    continue

                optimizer.step()

            if _has_non_finite_parameters(model):

                health["invalid_param_events"] += 1

                invalid_stop_reason = "non_finite_parameters_after_step"

                logger.error(

                    "检测到模型参数出现 NaN/Inf，终止训练: epoch=%s batch=%s/%s",

                    epoch + 1,

                    batch_idx + 1,

                    len(train_loader),

                )

                stop_training_now = True

                break



            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)

            train_corrects += (predicted == labels).sum().item()

            train_labels.extend(labels.cpu().numpy())

            train_preds.extend(predicted.cpu().numpy())

            processed_train_samples += labels.size(0)

            health["processed_train_batches"] += 1

            consecutive_invalid_batches = 0



            acc = 100.0 * train_corrects / max(train_total, 1)

            train_progress.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})



        epoch_train_loss = train_loss / processed_train_samples if processed_train_samples > 0 else float("nan")

        epoch_train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0.0

        epoch_train_f1 = f1_score(train_labels, train_preds, average="macro") if train_labels else 0.0



        run_validation = (not stop_training_now) and (

            ((epoch + 1) % max(int(val_every), 1) == 0) or ((epoch + 1) == num_epochs)

        )

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



        if stop_training_now:

            health["run_status"] = "failed"

            health["stop_reason"] = invalid_stop_reason or "invalid_training_state"

            if early_stopping.restore_best_weights and early_stopping.best_weights is not None:

                model.load_state_dict(early_stopping.best_weights)

                logger.warning("检测到训练异常，已恢复最佳权重并提前停止训练。")

            logger.error("训练失败保护触发，在第 %s 轮停止。reason=%s", epoch + 1, health["stop_reason"])

            break



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



            monitor_value = _select_monitor_value(early_stop_metric, val_loss, val_acc, val_f1)

            monitor_is_finite = math.isfinite(monitor_value)



            if monitor_is_finite:

                early_stopping(monitor_value, model)

                logger.info("早停计数器: %s/%s", early_stopping.counter, early_stopping.patience)

            else:

                logger.warning(

                    "跳过本轮早停更新：%s=%.6f 不是有限值（可能是 NaN/Inf）",

                    early_stop_metric,

                    monitor_value,

                )

                early_stopping.counter += 1

                logger.info("早停计数器: %s/%s (无效监控值按未改善处理)", early_stopping.counter, early_stopping.patience)

                if early_stopping.counter >= early_stopping.patience:

                    early_stopping.early_stop = True

                    if early_stopping.restore_best_weights and early_stopping.best_weights is not None:

                        model.load_state_dict(early_stopping.best_weights)



            if scheduler is not None:

                if lr_scheduler_mode == "reduce":

                    if monitor_is_finite:

                        scheduler.step(monitor_value)

                    else:

                        logger.warning("跳过 ReduceLROnPlateau 更新：监控指标无效")

                else:

                    scheduler.step()

            logger.info("当前学习率: %.8f", optimizer.param_groups[0]["lr"])



            if early_stopping.early_stop:

                health["stop_reason"] = "early_stop"

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



    if health["run_status"] != "failed":

        has_invalid = (health["invalid_loss_batches"] + health["invalid_grad_batches"] + health["invalid_param_events"]) > 0

        health["run_status"] = "degraded" if has_invalid else "ok"

    history["health"] = health

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





def visualize_attention_heatmap(attn_weights: np.ndarray, output_path: Path, title: str = "Cross-Modal Attention Heatmap"):

    """
    Visualizes attention weights as a heatmap.
    attn_weights: (seq_len,) or (1, seq_len)
    """

    plt = load_pyplot_headless()

    weights = np.asarray(attn_weights).reshape(-1)

    

                                                                              

    seq_len = len(weights)

    side = int(np.ceil(np.sqrt(seq_len)))

    padded_weights = np.zeros(side * side)

    padded_weights[:seq_len] = weights

    grid = padded_weights.reshape(side, side)

    

    plt.figure(figsize=(8, 7))

    sns.heatmap(grid, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Attention Weight'})

    plt.title(title)

    plt.xlabel("Sequence Column")

    plt.ylabel("Sequence Row")

    

    plt.savefig(output_path)

    plt.close()

    print(f"Saved Attention Heatmap: {output_path}")





def _normalize_probs(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    out = np.asarray(arr, dtype=np.float64)

    if out.ndim != 2:

        raise ValueError(f"Expected 2D probabilities, got shape={out.shape}")

    out = np.clip(out, 0.0, None)

    denom = np.clip(out.sum(axis=1, keepdims=True), eps, None)

    return out / denom





def _entropy_features(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    return -np.sum(probs * np.log(np.clip(probs, eps, None)), axis=1, keepdims=True)





def _margin_features(probs: np.ndarray) -> np.ndarray:

    if probs.shape[1] < 2:

        return np.zeros((probs.shape[0], 1), dtype=np.float64)

    sorted_probs = np.sort(probs, axis=1)

    return (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)





def build_meta_features_from_probs(

    text_probs: np.ndarray,

    image_probs: np.ndarray,

    *,

    fusion_probs: Optional[np.ndarray] = None,

) -> np.ndarray:

    text_probs = _normalize_probs(text_probs)

    image_probs = _normalize_probs(image_probs)

    if fusion_probs is None:

        fusion_probs = (text_probs + image_probs) / 2.0

    fusion_probs = _normalize_probs(fusion_probs)



    text_pred = np.argmax(text_probs, axis=1)

    image_pred = np.argmax(image_probs, axis=1)

    fusion_pred = np.argmax(fusion_probs, axis=1)



    agreement = np.stack(

        [

            (text_pred == image_pred).astype(np.float64),

            (text_pred == fusion_pred).astype(np.float64),

            (image_pred == fusion_pred).astype(np.float64),

        ],

        axis=1,

    )



    meta_features = np.concatenate(

        [

            text_probs,

            image_probs,

            fusion_probs,

            _entropy_features(text_probs),

            _entropy_features(image_probs),

            _entropy_features(fusion_probs),

            _margin_features(text_probs),

            _margin_features(image_probs),

            _margin_features(fusion_probs),

            agreement,

        ],

        axis=1,

    )

    return meta_features.astype(np.float32, copy=False)





def build_inverse_frequency_sample_weights(labels: np.ndarray) -> np.ndarray:

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if labels.size == 0:

        return np.array([], dtype=np.float64)

    classes, counts = np.unique(labels, return_counts=True)

    class_weights = {

        int(c): (float(labels.size) / float(max(1, len(classes)) * max(1, cnt)))

        for c, cnt in zip(classes.tolist(), counts.tolist())

    }

    weights = np.asarray([class_weights[int(y)] for y in labels.tolist()], dtype=np.float64)

    mean_w = float(np.mean(weights)) if weights.size else 1.0

    if mean_w > 0:

        weights = weights / mean_w

    return weights





def weighted_soft_voting(prob_list: List[np.ndarray], weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:

    if not prob_list:

        raise ValueError("prob_list is empty")

    probs = [_normalize_probs(p) for p in prob_list]

    n_samples = probs[0].shape[0]

    n_classes = probs[0].shape[1]

    for p in probs:

        if p.shape != (n_samples, n_classes):

            raise ValueError("All probability arrays must share the same shape")



    if weights is None:

        weights_arr = np.ones(len(probs), dtype=np.float64)

    else:

        weights_arr = np.asarray(weights, dtype=np.float64)

        if weights_arr.shape[0] != len(probs):

            raise ValueError("weights length mismatch")

    weights_arr = np.clip(weights_arr, 1e-12, None)

    weights_arr = weights_arr / np.sum(weights_arr)



    voted = np.zeros((n_samples, n_classes), dtype=np.float64)

    for p, w in zip(probs, weights_arr.tolist()):

        voted += p * float(w)

    voted = _normalize_probs(voted)

    preds = np.argmax(voted, axis=1).astype(np.int64)

    return voted, preds





def apply_class_gains(probs: np.ndarray, class_gains: Dict[int, float]) -> np.ndarray:

    tuned = np.asarray(probs, dtype=np.float64).copy()

    if tuned.ndim != 2:

        raise ValueError(f"Expected 2D probs, got shape={tuned.shape}")

    for cls, gain in class_gains.items():

        if 0 <= int(cls) < tuned.shape[1]:

            tuned[:, int(cls)] *= float(max(gain, 1e-6))

    return _normalize_probs(tuned)





def tune_class_gains(

    *,

    labels: np.ndarray,

    probs: np.ndarray,

    target_classes: List[int],

    gain_grid: Optional[List[float]] = None,

) -> Dict[int, float]:

    probs = _normalize_probs(probs)

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if labels.size == 0 or probs.shape[0] != labels.shape[0]:

        return {}

    target_classes = [int(c) for c in target_classes if 0 <= int(c) < probs.shape[1]]

    if not target_classes:

        return {}

    if gain_grid is None:

        gain_grid = [1.0, 1.1, 1.25, 1.5, 1.8, 2.0]

    gain_grid = [float(g) for g in gain_grid if float(g) > 0]

    if not gain_grid:

        return {}



    base_preds = np.argmax(probs, axis=1)

    best_f1 = f1_score(labels, base_preds, average="macro")

    best_map = {c: 1.0 for c in target_classes}

    grid = list(product(gain_grid, repeat=len(target_classes)))

    for gains in grid:

        gains_map = {c: g for c, g in zip(target_classes, gains)}

        tuned = apply_class_gains(probs, gains_map)

        preds = np.argmax(tuned, axis=1)

        score = f1_score(labels, preds, average="macro")

        if score > best_f1:

            best_f1 = score

            best_map = gains_map

    return best_map





def fit_binary_centroid_head(features: np.ndarray, labels: np.ndarray, *, class_a: int, class_b: int) -> Optional[Dict[str, Any]]:

    features = np.asarray(features, dtype=np.float64)

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    mask = (labels == int(class_a)) | (labels == int(class_b))

    if features.ndim != 2 or labels.size != features.shape[0] or int(mask.sum()) < 2:

        return None

    xa = features[labels == int(class_a)]

    xb = features[labels == int(class_b)]

    if xa.shape[0] == 0 or xb.shape[0] == 0:

        return None

    pair = np.concatenate([xa, xb], axis=0)

    var = np.var(pair, axis=0) + 1e-6

    return {

        "class_a": int(class_a),

        "class_b": int(class_b),

        "centroid_a": np.mean(xa, axis=0),

        "centroid_b": np.mean(xb, axis=0),

        "var": var,

    }





def apply_binary_correction_for_pair(

    *,

    preds: np.ndarray,

    probs: np.ndarray,

    features: np.ndarray,

    head: Optional[Dict[str, Any]],

    class_a: int,

    class_b: int,

    alpha: float = 1.0,

) -> Tuple[np.ndarray, np.ndarray]:

    preds = np.asarray(preds, dtype=np.int64).copy()

    probs = _normalize_probs(probs)

    features = np.asarray(features, dtype=np.float64)

    if head is None:

        return preds, probs

    if probs.shape[0] != preds.shape[0] or features.shape[0] != preds.shape[0]:

        raise ValueError("preds/probs/features length mismatch")



    ca = int(class_a)

    cb = int(class_b)

    if ca < 0 or cb < 0 or ca >= probs.shape[1] or cb >= probs.shape[1]:

        return preds, probs



    mask = (preds == ca) | (preds == cb)

    if not np.any(mask):

        return preds, probs



    x = features[mask]

    var = np.asarray(head["var"], dtype=np.float64)

    da = np.sum(((x - np.asarray(head["centroid_a"])) ** 2) / var, axis=1)

    db = np.sum(((x - np.asarray(head["centroid_b"])) ** 2) / var, axis=1)



    score_a = np.exp(-0.5 * da)

    score_b = np.exp(-0.5 * db)

    pair_sum = np.clip(score_a + score_b, 1e-12, None)

    pa = score_a / pair_sum

    pb = score_b / pair_sum



    corrected_probs = probs.copy()

    pair_mass = np.clip(corrected_probs[mask, ca] + corrected_probs[mask, cb], 1e-6, 1.0)

    alpha = float(np.clip(alpha, 0.0, 1.0))

    target_pa = pair_mass * pa

    target_pb = pair_mass * pb

    corrected_probs[mask, ca] = ((1.0 - alpha) * corrected_probs[mask, ca]) + (alpha * target_pa)

    corrected_probs[mask, cb] = ((1.0 - alpha) * corrected_probs[mask, cb]) + (alpha * target_pb)

    corrected_probs = _normalize_probs(corrected_probs)



    corrected_preds = preds.copy()

    pa_corr = corrected_probs[mask, ca]

    pb_corr = corrected_probs[mask, cb]

    corrected_preds[mask] = np.where(

        pa_corr > pb_corr,

        ca,

        np.where(pa_corr < pb_corr, cb, corrected_preds[mask]),

    )

    return corrected_preds, corrected_probs





def tune_binary_correction_alpha_for_pair(

    *,

    labels: np.ndarray,

    probs: np.ndarray,

    features: np.ndarray,

    head: Optional[Dict[str, Any]],

    class_a: int,

    class_b: int,

    objective: str = "macro_f1",

    alpha_grid: Optional[List[float]] = None,

) -> float:

    if head is None:

        return 0.0

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    probs = _normalize_probs(probs)

    features = np.asarray(features, dtype=np.float64)

    if labels.size == 0 or probs.shape[0] != labels.size or features.shape[0] != labels.size:

        return 0.0

    if alpha_grid is None:

        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    alpha_grid = [float(a) for a in alpha_grid if np.isfinite(a)]

    if not alpha_grid:

        return 0.0



    base_preds = np.argmax(probs, axis=1).astype(np.int64)

    best_alpha = 0.0

    if objective == "accuracy":

        best_score = float(accuracy_score(labels, base_preds))

    elif objective == "pair_f1":

        best_score = score_pair_f1(labels, base_preds, class_a=class_a, class_b=class_b)

    else:

        best_score = f1_score(labels, base_preds, average="macro")

    for alpha in alpha_grid:

        preds, _ = apply_binary_correction_for_pair(

            preds=base_preds,

            probs=probs,

            features=features,

            head=head,

            class_a=class_a,

            class_b=class_b,

            alpha=alpha,

        )

        if objective == "accuracy":

            score = float(accuracy_score(labels, preds))

        elif objective == "pair_f1":

            score = score_pair_f1(labels, preds, class_a=class_a, class_b=class_b)

        else:

            score = f1_score(labels, preds, average="macro")

        if score > best_score:

            best_score = score

            best_alpha = float(np.clip(alpha, 0.0, 1.0))

    return best_alpha





def score_pair_f1(labels: np.ndarray, preds: np.ndarray, *, class_a: int, class_b: int) -> float:

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    preds = np.asarray(preds, dtype=np.int64).reshape(-1)

    if labels.size == 0 or preds.size != labels.size:

        return 0.0

    mask = (labels == int(class_a)) | (labels == int(class_b))

    if not np.any(mask):

        return 0.0

    y_true = np.where(labels[mask] == int(class_a), 0, 1)

    y_pred = np.where(preds[mask] == int(class_a), 0, np.where(preds[mask] == int(class_b), 1, 2))

    return float(f1_score(y_true, y_pred, labels=[0, 1], average="macro"))





def apply_pair_temperature(

    *,

    probs: np.ndarray,

    class_a: int,

    class_b: int,

    temperature: float,

) -> np.ndarray:

    calibrated = _normalize_probs(probs).copy()

    ca = int(class_a)

    cb = int(class_b)

    if ca < 0 or cb < 0 or ca >= calibrated.shape[1] or cb >= calibrated.shape[1]:

        return calibrated

    t = float(max(temperature, 1e-6))

    pair_mass = calibrated[:, ca] + calibrated[:, cb]

    pair_mask = pair_mass > 1e-12

    if not np.any(pair_mask):

        return calibrated

    pair_a = calibrated[pair_mask, ca] / np.clip(pair_mass[pair_mask], 1e-12, None)

    pair_b = calibrated[pair_mask, cb] / np.clip(pair_mass[pair_mask], 1e-12, None)

    logits = np.stack([np.log(np.clip(pair_a, 1e-12, None)), np.log(np.clip(pair_b, 1e-12, None))], axis=1) / t

    logits = logits - np.max(logits, axis=1, keepdims=True)

    scaled = np.exp(logits)

    scaled = scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)

    calibrated[pair_mask, ca] = pair_mass[pair_mask] * scaled[:, 0]

    calibrated[pair_mask, cb] = pair_mass[pair_mask] * scaled[:, 1]

    return _normalize_probs(calibrated)





def tune_pair_temperature(

    *,

    labels: np.ndarray,

    probs: np.ndarray,

    class_a: int,

    class_b: int,

    temperature_grid: Optional[List[float]] = None,

) -> float:

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    probs = _normalize_probs(probs)

    if labels.size == 0 or probs.shape[0] != labels.size:

        return 1.0

    mask = (labels == int(class_a)) | (labels == int(class_b))

    if int(mask.sum()) < 2:

        return 1.0

    y_true = np.where(labels[mask] == int(class_a), 0, 1)

    if temperature_grid is None:

        temperature_grid = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]

    candidates = [float(t) for t in temperature_grid if np.isfinite(t) and float(t) > 0]

    if not candidates:

        return 1.0



    best_t = 1.0

    best_nll = float("inf")

    for t in candidates:

        calibrated = apply_pair_temperature(probs=probs, class_a=class_a, class_b=class_b, temperature=t)

        pair = calibrated[mask][:, [int(class_a), int(class_b)]]

        pair = pair / np.clip(pair.sum(axis=1, keepdims=True), 1e-12, None)

        nll = -float(np.mean(np.log(np.clip(pair[np.arange(pair.shape[0]), y_true], 1e-12, None))))

        if nll < best_nll:

            best_nll = nll

            best_t = t

    return best_t





def apply_pair_threshold(

    *,

    preds: np.ndarray,

    probs: np.ndarray,

    class_a: int,

    class_b: int,

    threshold: float,

) -> np.ndarray:

    out = np.asarray(preds, dtype=np.int64).copy()

    probs = _normalize_probs(probs)

    ca = int(class_a)

    cb = int(class_b)

    if ca < 0 or cb < 0 or ca >= probs.shape[1] or cb >= probs.shape[1]:

        return out

    thr = float(np.clip(threshold, 0.0, 1.0))

    mask = (out == ca) | (out == cb)

    if not np.any(mask):

        return out

    pair_mass = np.clip(probs[mask, ca] + probs[mask, cb], 1e-12, None)

    ratio_a = probs[mask, ca] / pair_mass

    out[mask] = np.where(ratio_a >= thr, ca, cb)

    return out





def tune_pair_threshold(

    *,

    labels: np.ndarray,

    probs: np.ndarray,

    class_a: int,

    class_b: int,

    objective: str = "pair_f1",

    threshold_grid: Optional[List[float]] = None,

) -> float:

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    probs = _normalize_probs(probs)

    if labels.size == 0 or probs.shape[0] != labels.size:

        return 0.5

    if threshold_grid is None:

        threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

    candidates = [float(t) for t in threshold_grid if np.isfinite(t)]

    if not candidates:

        return 0.5

    base_preds = np.argmax(probs, axis=1).astype(np.int64)

    best_thr = 0.5

    if objective == "accuracy":

        best_score = float(accuracy_score(labels, base_preds))

    else:

        best_score = score_pair_f1(labels, base_preds, class_a=class_a, class_b=class_b)

    for thr in candidates:

        preds = apply_pair_threshold(preds=base_preds, probs=probs, class_a=class_a, class_b=class_b, threshold=thr)

        if objective == "accuracy":

            score = float(accuracy_score(labels, preds))

        else:

            score = score_pair_f1(labels, preds, class_a=class_a, class_b=class_b)

        if score > best_score:

            best_score = score

            best_thr = float(np.clip(thr, 0.0, 1.0))

    return best_thr





def apply_multiclass_temperature(*, probs: np.ndarray, temperature: float) -> np.ndarray:

    probs = _normalize_probs(probs)

    t = float(max(1e-3, temperature))

    logits = np.log(np.clip(probs, 1e-12, 1.0))

    scaled = logits / t

    scaled -= np.max(scaled, axis=1, keepdims=True)

    exp_scaled = np.exp(scaled)

    return _normalize_probs(exp_scaled)





def tune_multiclass_temperature(

    *,

    labels: np.ndarray,

    probs: np.ndarray,

    grid: Optional[List[float]] = None,

) -> float:

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    p = _normalize_probs(probs)

    if y.size == 0 or p.shape[0] != y.size:

        return 1.0

    candidates = [float(v) for v in (grid or [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]) if float(v) > 0]

    if not candidates:

        return 1.0

    class_labels = list(range(p.shape[1]))

    best_t = 1.0

    best_loss = float(log_loss(y, p, labels=class_labels))

    for t in candidates:

        calibrated = apply_multiclass_temperature(probs=p, temperature=t)

        cur_loss = float(log_loss(y, calibrated, labels=class_labels))

        if np.isfinite(cur_loss) and cur_loss <= best_loss:

            best_t = float(t)

            best_loss = cur_loss

    return best_t





def compute_calibration_metrics(*, labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Dict[str, float]:

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    p = _normalize_probs(probs)

    if y.size == 0 or p.shape[0] != y.size:

        return {"ece": 0.0, "brier": 0.0}



    confidence = np.max(p, axis=1)

    pred = np.argmax(p, axis=1)

    correct = (pred == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, num=max(2, int(n_bins) + 1))

    ece = 0.0

    for idx in range(len(bins) - 1):

        left = bins[idx]

        right = bins[idx + 1]

        if idx == len(bins) - 2:

            mask = (confidence >= left) & (confidence <= right)

        else:

            mask = (confidence >= left) & (confidence < right)

        if not np.any(mask):

            continue

        acc_bin = float(np.mean(correct[mask]))

        conf_bin = float(np.mean(confidence[mask]))

        weight = float(np.mean(mask))

        ece += abs(acc_bin - conf_bin) * weight



    one_hot = np.zeros_like(p)

    one_hot[np.arange(y.size), y] = 1.0

    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    return {"ece": float(ece), "brier": brier}





def calibrate_multiclass_probs(

    *,

    labels: np.ndarray,

    oof_probs: np.ndarray,

    test_probs: np.ndarray,

    calibration: str,

) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    oof = _normalize_probs(oof_probs)

    test = _normalize_probs(test_probs)

    strategy = str(calibration or "none").strip().lower()

    info: Dict[str, Any] = {"method": strategy}



    if strategy == "temp":

        t = tune_multiclass_temperature(labels=y, probs=oof)

        oof = apply_multiclass_temperature(probs=oof, temperature=t)

        test = apply_multiclass_temperature(probs=test, temperature=t)

        info["temperature"] = float(t)

    elif strategy == "isotonic":

        try:

            from sklearn.isotonic import IsotonicRegression



            calibrated_oof = np.zeros_like(oof)

            calibrated_test = np.zeros_like(test)

            for class_idx in range(oof.shape[1]):

                y_bin = (y == class_idx).astype(np.float64)

                if len(np.unique(y_bin)) < 2:

                    calibrated_oof[:, class_idx] = oof[:, class_idx]

                    calibrated_test[:, class_idx] = test[:, class_idx]

                    continue

                ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

                ir.fit(oof[:, class_idx], y_bin)

                calibrated_oof[:, class_idx] = ir.transform(oof[:, class_idx])

                calibrated_test[:, class_idx] = ir.transform(test[:, class_idx])

            oof = _normalize_probs(calibrated_oof)

            test = _normalize_probs(calibrated_test)

        except Exception as exc:

            info["fallback"] = "none"

            info["error"] = str(exc)



    info["oof_metrics"] = compute_calibration_metrics(labels=y, probs=oof)

    return oof, test, info





def build_level2_features(method_probs: Dict[str, np.ndarray]) -> np.ndarray:

    if not method_probs:

        return np.zeros((0, 0), dtype=np.float64)

    ordered = sorted((name, _normalize_probs(prob)) for name, prob in method_probs.items())

    blocks: List[np.ndarray] = []

    for _, prob in ordered:

        blocks.append(prob)

        entropy = _entropy_features(prob)

        margin = _margin_features(prob)

        blocks.extend([entropy, margin])



    stacked = np.stack([prob for _, prob in ordered], axis=0)

    mean_prob = _normalize_probs(np.mean(stacked, axis=0))

    vote_entropy = _entropy_features(mean_prob)

    vote_margin = _margin_features(mean_prob)

    pairwise_kl = compute_pairwise_kl_features(dict(ordered))

    blocks.extend([vote_entropy, vote_margin, pairwise_kl])

    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)





def compute_pairwise_kl_features(method_probs: Dict[str, np.ndarray]) -> np.ndarray:

    if not method_probs:

        return np.zeros((0, 1), dtype=np.float64)

    ordered = [_normalize_probs(prob) for _, prob in sorted(method_probs.items())]

    n_samples = ordered[0].shape[0]

    if len(ordered) < 2:

        return np.zeros((n_samples, 1), dtype=np.float64)



    eps = 1e-12

    values = np.zeros((n_samples,), dtype=np.float64)

    pairs = 0

    for i in range(len(ordered)):

        for j in range(i + 1, len(ordered)):

            p = np.clip(ordered[i], eps, 1.0)

            q = np.clip(ordered[j], eps, 1.0)

            kl_pq = np.sum(p * (np.log(p) - np.log(q)), axis=1)

            kl_qp = np.sum(q * (np.log(q) - np.log(p)), axis=1)

            values += 0.5 * (kl_pq + kl_qp)

            pairs += 1

    if pairs > 0:

        values /= float(pairs)

    return values.reshape(-1, 1)





def build_hard_sample_factors(method_probs: Dict[str, np.ndarray]) -> np.ndarray:

    if not method_probs:

        return np.array([], dtype=np.float64)

    ordered = [_normalize_probs(prob) for _, prob in sorted(method_probs.items())]

    mean_prob = _normalize_probs(np.mean(np.stack(ordered, axis=0), axis=0))

    max_conf = np.max(mean_prob, axis=1)

    hard = 1.0 + np.clip(1.0 - max_conf, 0.0, 1.0)

    hard_mean = float(np.mean(hard)) if hard.size else 1.0

    if hard_mean > 0:

        hard = hard / hard_mean

    return hard.astype(np.float64, copy=False)





def apply_per_class_thresholds(*, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:

    p = _normalize_probs(probs)

    tau = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)

    if p.shape[1] != tau.shape[1]:

        raise ValueError("threshold shape mismatch")

    adjusted = p / np.clip(tau, 1e-6, None)

    return np.argmax(adjusted, axis=1).astype(np.int64)





def tune_per_class_thresholds(

    *,

    labels: np.ndarray,

    probs: np.ndarray,

    minority_classes: List[int],

    objective: str,

    minority_lambda: float,

    grid: Optional[List[float]] = None,

) -> np.ndarray:

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    p = _normalize_probs(probs)

    if y.size == 0 or p.shape[0] != y.size:

        return np.ones(p.shape[1], dtype=np.float64)



    candidates = [float(v) for v in (grid or [0.7, 0.85, 1.0, 1.15, 1.3]) if float(v) > 0]

    if not candidates:

        candidates = [1.0]

    thresholds = np.ones(p.shape[1], dtype=np.float64)



    def _score(preds: np.ndarray) -> float:

        if objective == "accuracy":

            return float(accuracy_score(y, preds))

        macro = float(f1_score(y, preds, average="macro", zero_division=0))

        if objective == "macro_f1":

            return macro

        valid_minority = [int(c) for c in minority_classes if 0 <= int(c) < p.shape[1]]

        if not valid_minority:

            return macro

        minor_recall = float(recall_score(y, preds, labels=valid_minority, average="macro", zero_division=0))

        return macro + float(minority_lambda) * minor_recall



    best_preds = apply_per_class_thresholds(probs=p, thresholds=thresholds)

    best_score = _score(best_preds)

    for class_idx in range(p.shape[1]):

        best_value = thresholds[class_idx]

        for value in candidates:

            trial = thresholds.copy()

            trial[class_idx] = float(value)

            preds = apply_per_class_thresholds(probs=p, thresholds=trial)

            score = _score(preds)

            if score > best_score:

                best_score = score

                best_value = float(value)

        thresholds[class_idx] = best_value

    return thresholds





def compute_threshold_objective_value(

    *,

    labels: np.ndarray,

    preds: np.ndarray,

    minority_classes: List[int],

    objective: str,

    minority_lambda: float,

) -> float:

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    p = np.asarray(preds, dtype=np.int64).reshape(-1)

    if y.size == 0 or p.size != y.size:

        return 0.0

    if objective == "accuracy":

        return float(accuracy_score(y, p))

    macro = float(f1_score(y, p, average="macro", zero_division=0))

    if objective == "macro_f1":

        return macro

    valid_minority = sorted({int(c) for c in minority_classes if 0 <= int(c)})

    if not valid_minority:

        return macro

    minor = float(recall_score(y, p, labels=valid_minority, average="macro", zero_division=0))

    return macro + float(minority_lambda) * minor





def build_single_layer_baseline_result(method_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:

    candidates: List[Dict[str, Any]] = []

    for item in method_results:

        method = str(item.get("method", ""))

        if method in {"soft_voting", "two_level_blender", "single_layer_baseline"}:

            continue

        if item.get("skipped") or item.get("failed"):

            continue

        if "oof_macro_f1" not in item:

            continue

        candidates.append(item)

    if not candidates:

        return None

    best = max(candidates, key=lambda x: float(x.get("oof_macro_f1", 0.0)))

    return {

        "method": "single_layer_baseline",

        "source_method": str(best.get("method", "")),

        "acc": float(best.get("acc", 0.0)),

        "macro_f1": float(best.get("macro_f1", 0.0)),

        "oof_acc": float(best.get("oof_acc", 0.0)),

        "oof_macro_f1": float(best.get("oof_macro_f1", 0.0)),

        "report": best.get("report", ""),

        "confusion_matrix": best.get("confusion_matrix", np.zeros((0, 0), dtype=int)),

        "confusion_matrix_path": best.get("confusion_matrix_path"),

        "report_path": best.get("report_path"),

        "meta_model_path": best.get("meta_model_path"),

        "postprocess": {"type": "single_layer_best_by_oof"},

    }





def resolve_effective_stacking_level(*, requested_level: str, successful_methods: List[str]) -> str:

    level = str(requested_level or "single").strip().lower()

    if level == "two_level" and len(successful_methods) >= 2:

        return "two_level"

    return "single"





def _fit_level2_blender(train_x: np.ndarray, train_y: np.ndarray, sample_weight: Optional[np.ndarray]) -> Any:

    try:

        return train_meta_learner(train_x, train_y, method="xgboost", sample_weight=sample_weight)

    except Exception:

        from sklearn.linear_model import LogisticRegression



        clf = LogisticRegression(max_iter=1000, multi_class="auto")

        fit_kwargs: Dict[str, Any] = {}

        if sample_weight is not None and len(sample_weight) == len(train_y):

            fit_kwargs["sample_weight"] = sample_weight

        clf.fit(train_x, train_y, **fit_kwargs)

        return clf





def run_level2_blender_oof(

    *,

    labels: np.ndarray,

    level1_oof_probs: Dict[str, np.ndarray],

    level1_test_probs: Dict[str, np.ndarray],

    oof_folds: int,

    minority_classes: List[int],

    threshold_objective: str,

    minority_lambda: float,

) -> Dict[str, Any]:

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    x_oof = build_level2_features(level1_oof_probs)

    x_test = build_level2_features(level1_test_probs)

    if y.size == 0 or x_oof.shape[0] != y.size or x_test.shape[0] == 0:

        return {

            "preds": np.array([], dtype=np.int64),

            "probs": np.zeros((0, 0), dtype=np.float64),

            "oof_probs": np.zeros((0, 0), dtype=np.float64),

            "thresholds": np.array([], dtype=np.float64),

        }



    num_classes = int(max(2, len(np.unique(y))))

    base_weight = build_inverse_frequency_sample_weights(y)

    hard_factor = build_hard_sample_factors(level1_oof_probs)

    sample_weight = base_weight

    if hard_factor.shape[0] == base_weight.shape[0]:

        sample_weight = base_weight * hard_factor

        weight_mean = float(np.mean(sample_weight)) if sample_weight.size else 1.0

        if weight_mean > 0:

            sample_weight = sample_weight / weight_mean



    def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:

        fold_w = build_inverse_frequency_sample_weights(train_y)

        if train_x.shape[0] > 0:

            train_conf = np.max(train_x[:, :num_classes], axis=1)

            fold_hard = 1.0 + np.clip(1.0 - train_conf, 0.0, 1.0)

            fold_hard_mean = float(np.mean(fold_hard)) if fold_hard.size else 1.0

            if fold_hard_mean > 0:

                fold_hard = fold_hard / fold_hard_mean

            if fold_hard.shape[0] == fold_w.shape[0]:

                fold_w = fold_w * fold_hard

                fw_mean = float(np.mean(fold_w)) if fold_w.size else 1.0

                if fw_mean > 0:

                    fold_w = fold_w / fw_mean

        model = _fit_level2_blender(train_x, train_y, fold_w)

        _, fold_probs = _predict_with_meta_model(model, valid_x, num_classes=num_classes)

        return fold_probs



    oof_probs = compute_oof_predictions(

        features=x_oof,

        labels=y,

        n_splits=max(2, int(oof_folds)),

        seed=42,

        fit_predict_fn=_fit_predict,

    )

    blender = _fit_level2_blender(x_oof, y, sample_weight)

    _, test_probs = _predict_with_meta_model(blender, x_test, num_classes=oof_probs.shape[1])

    thresholds = tune_per_class_thresholds(

        labels=y,

        probs=oof_probs,

        minority_classes=minority_classes,

        objective=threshold_objective,

        minority_lambda=minority_lambda,

    )

    preds = apply_per_class_thresholds(probs=test_probs, thresholds=thresholds)

    oof_threshold_preds = apply_per_class_thresholds(probs=oof_probs, thresholds=thresholds)

    threshold_objective_value = compute_threshold_objective_value(

        labels=y,

        preds=oof_threshold_preds,

        minority_classes=minority_classes,

        objective=threshold_objective,

        minority_lambda=minority_lambda,

    )

    return {

        "preds": preds,

        "probs": test_probs,

        "oof_probs": oof_probs,

        "thresholds": thresholds,

        "threshold_objective_value": float(threshold_objective_value),

        "blender": blender,

    }





def build_two_level_postprocess_payload(

    *,

    stacking_level: str,

    calibration: Dict[str, float],

    thresholds: np.ndarray,

    minority_classes: List[int],

    minority_recall_before: float,

    minority_recall_after: float,

    threshold_objective: str,

    objective_value: float,

    oof_test_gap: float,

) -> Dict[str, Any]:

    return {

        "stacking_level": str(stacking_level),

        "calibration": dict(calibration),

        "thresholds": [float(v) for v in np.asarray(thresholds, dtype=np.float64).tolist()],

        "threshold_objective": str(threshold_objective),

        "objective_value": float(objective_value),

        "oof_test_gap": float(oof_test_gap),

        "minority_metrics": {

            "classes": [int(c) for c in minority_classes],

            "recall_before": float(minority_recall_before),

            "recall_after": float(minority_recall_after),

        },

    }





def compute_oof_predictions(

    *,

    features: np.ndarray,

    labels: np.ndarray,

    n_splits: int,

    seed: int,

    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],

    on_fold: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,

) -> np.ndarray:

    x = np.asarray(features, dtype=np.float64)

    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    if x.ndim != 2 or y.shape[0] != x.shape[0]:

        raise ValueError("features/labels shape mismatch")

    if x.shape[0] == 0:

        return np.zeros((0, 0), dtype=np.float64)



    if StratifiedKFold is None:

        return _normalize_probs(fit_predict_fn(x, y, x))



    _, counts = np.unique(y, return_counts=True)

    max_valid_splits = int(np.min(counts)) if counts.size else 1

    if max_valid_splits < 2:

        return _normalize_probs(fit_predict_fn(x, y, x))

    use_splits = min(int(n_splits), max_valid_splits)



    skf = StratifiedKFold(n_splits=use_splits, shuffle=True, random_state=int(seed))

    oof_probs = None

    covered = np.zeros(x.shape[0], dtype=bool)

    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(x, y)):

        fold_probs = _normalize_probs(fit_predict_fn(x[train_idx], y[train_idx], x[valid_idx]))

        if oof_probs is None:

            oof_probs = np.zeros((x.shape[0], fold_probs.shape[1]), dtype=np.float64)

        if fold_probs.shape[1] != oof_probs.shape[1]:

            raise ValueError("inconsistent class dimension across folds")

        oof_probs[valid_idx] = fold_probs

        covered[valid_idx] = True

        if on_fold is not None:

            on_fold(fold_id, train_idx, valid_idx)

    if oof_probs is None:

        return np.zeros((x.shape[0], 0), dtype=np.float64)

    if not np.all(covered):

        raise RuntimeError("OOF split coverage is incomplete")

    return _normalize_probs(oof_probs)





def _predict_with_meta_model(meta_model, features: np.ndarray, *, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:

    features = np.asarray(features, dtype=np.float64)

    if hasattr(meta_model, "predict_proba"):

        probs = _normalize_probs(np.asarray(meta_model.predict_proba(features), dtype=np.float64))

        preds = np.argmax(probs, axis=1).astype(np.int64)

        return preds, probs

    preds = np.asarray(meta_model.predict(features), dtype=np.int64).reshape(-1)

    probs = np.zeros((preds.shape[0], num_classes), dtype=np.float64)

    valid = (preds >= 0) & (preds < num_classes)

    probs[np.arange(preds.shape[0])[valid], preds[valid]] = 1.0

    return preds, _normalize_probs(probs)





def generate_meta_features(

    text_model: nn.Module,

    mobilevit_model: nn.Module,

    data_loader: DataLoader,

    device: torch.device,

    *,

    fusion_model: Optional[nn.Module] = None,

    use_softmax: bool = True,

):

    text_model.eval()

    mobilevit_model.eval()

    if fusion_model is not None:

        fusion_model.eval()

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

            text_out = torch.softmax(text_logits, dim=1).cpu().numpy() if use_softmax else text_logits.cpu().numpy()



            mobilevit_logits = mobilevit_model(images)

            if hasattr(mobilevit_logits, "logits"):

                mobilevit_logits = mobilevit_logits.logits

            img_out = (

                torch.softmax(mobilevit_logits, dim=1).cpu().numpy()

                if use_softmax

                else mobilevit_logits.cpu().numpy()

            )



            fusion_out = None

            if fusion_model is not None:

                fusion_logits = fusion_model(images, pcap_data)

                if isinstance(fusion_logits, (tuple, list)):

                    fusion_logits = fusion_logits[0]

                fusion_out = (

                    torch.softmax(fusion_logits, dim=1).cpu().numpy()

                    if use_softmax

                    else fusion_logits.cpu().numpy()

                )



            meta_features.append(build_meta_features_from_probs(text_out, img_out, fusion_probs=fusion_out))

            meta_labels.append(labels.cpu().numpy())



    if meta_features:

        return np.concatenate(meta_features, axis=0), np.concatenate(meta_labels, axis=0)

    return np.array([]), np.array([])





def train_xgboost(

    meta_features: np.ndarray,

    meta_labels: np.ndarray,

    *,

    sample_weight: Optional[np.ndarray] = None,

):

    try:

        import xgboost as xgb

    except ImportError as e:

        raise ImportError("xgboost 未安装") from e



    n_classes = int(len(np.unique(meta_labels)))

    clf = xgb.XGBClassifier(

        n_estimators=350,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.9,

        colsample_bytree=0.85,

        min_child_weight=2.0,

        reg_lambda=1.0,

        objective="multi:softprob",

        num_class=max(2, n_classes),

        eval_metric="mlogloss",

        random_state=42,

    )

    fit_kwargs = {}

    if sample_weight is not None and len(sample_weight) == len(meta_labels):

        fit_kwargs["sample_weight"] = sample_weight

    clf.fit(meta_features, meta_labels, **fit_kwargs)

    return clf





def train_meta_learner(

    meta_features: np.ndarray,

    meta_labels: np.ndarray,

    method: str = "xgboost",

    *,

    sample_weight: Optional[np.ndarray] = None,

):

    if method == "xgboost":

        return train_xgboost(meta_features, meta_labels, sample_weight=sample_weight)

    if method == "lightgbm":

        try:

            import lightgbm as lgb

        except ImportError as e:

            raise ImportError("lightgbm 未安装") from e

        clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05)

        fit_kwargs = {}

        if sample_weight is not None and len(sample_weight) == len(meta_labels):

            fit_kwargs["sample_weight"] = sample_weight

        clf.fit(meta_features, meta_labels, **fit_kwargs)

        return clf

    if method == "catboost":

        try:

            from catboost import CatBoostClassifier

        except ImportError as e:

            raise ImportError("catboost 未安装") from e

        clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

        fit_kwargs = {}

        if sample_weight is not None and len(sample_weight) == len(meta_labels):

            fit_kwargs["sample_weight"] = sample_weight

        clf.fit(meta_features, meta_labels, **fit_kwargs)

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

    p.add_argument(

        "--dataset_root",

        default=str(PROJECT_ROOT / "ProcessedData"),

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

        default="ustc_multiclass",

        help="ProcessedData task name, e.g. binary_benign_vs_malicious or ustc_multiclass",

    )

    p.add_argument(

        "--cic_group",

        default="",

        help="For grouped CIC dataset, choose one or more top-level groups, e.g. Adware or Adware,Ransomware",

    )



    p.add_argument("--image_mode", choices=["rgb", "gray"], default="rgb", help="Image mode: rgb or gray")

    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--image_size", type=int, default=28)

    p.add_argument("--max_pcap_length", type=int, default=784)



    p.add_argument("--epochs", type=int, default=32)

    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--patience", type=int, default=4)

    p.add_argument("--preset", choices=["none", "cic_balanced", "mfcp_score_chasing", "mta_score_chasing"], default="none")



    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")

    p.add_argument("--seed", type=int, default=42)



    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--persistent_workers", action="store_true")

    p.add_argument("--prefetch_factor", type=int, default=2)

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

    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--early_stop_metric", choices=["val_loss", "val_acc", "val_f1"], default="val_loss")

    p.add_argument("--early_stop_mode", choices=["auto", "min", "max"], default="auto")

    p.add_argument("--lr_scheduler", choices=["none", "reduce", "cosine"], default="reduce")

    p.add_argument("--lr_patience", type=int, default=2)

    p.add_argument("--lr_factor", type=float, default=0.5)

    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--grad_clip_norm", type=float, default=1.0)

    p.add_argument("--val_every", type=int, default=1)

    p.add_argument("--max_consecutive_invalid_batches", type=int, default=128)



    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT))

    p.add_argument("--attention_dim", type=int, default=256)

    p.add_argument("--fusion_mode", choices=["attention", "concat", "weighted"], default="attention", help="Fusion mechanism")

    p.add_argument("--no_temporal", action="store_true", help="Bypass temporal sidecar (json) files and use flat bytes")

    p.add_argument("--charbert_mode", choices=["legacy", "charaware"], default="charaware")

    p.add_argument("--char_vocab", choices=["hex", "ascii"], default="hex")

    p.add_argument("--char_emb_dim", type=int, default=32)

    p.add_argument("--char_cnn_channels", type=int, default=64)

    p.add_argument("--char_fusion", choices=["gated", "add", "concat"], default="gated")

    p.add_argument("--char_fusion_layers", choices=["first", "last", "all"], default="all")

    p.add_argument("--stacking_level", choices=["single", "two_level"], default="two_level")

    p.add_argument("--stacking_calibration", choices=["none", "temp", "isotonic"], default="temp")

    p.add_argument(

        "--stacking_threshold_objective",

        choices=["macro_f1", "macro_f1_minority_recall", "accuracy"],

        default="macro_f1_minority_recall",

    )

    p.add_argument("--stacking_minority_lambda", type=float, default=0.3)

    p.add_argument("--stacking_oof_folds", type=int, default=5)

    return p





def _arg_explicitly_set(flag: str) -> bool:

    return any(token == flag or token.startswith(f"{flag}=") for token in sys.argv[1:])





def _apply_preset_defaults(args, resolved_dataset_name: str) -> None:

    preset_name = str(getattr(args, "preset", "none") or "none").strip().lower()

    if preset_name == "none":

        return



    if preset_name == "cic_balanced":

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

        return



    if preset_name == "mfcp_score_chasing":

        preset_values = {

            "--class_balance": "none",

            "--loss_type": "ce",

            "--label_smoothing": 0.0,

            "--early_stop_metric": "val_acc",

            "--early_stop_mode": "max",

            "--stacking_level": "two_level",

            "--stacking_calibration": "temp",

            "--stacking_threshold_objective": "accuracy",

            "--lr_scheduler": "reduce",

        }

        for flag, value in preset_values.items():

            if not _arg_explicitly_set(flag):

                setattr(args, flag[2:], value)

        return





def _apply_task_defaults(args) -> None:

    if str(getattr(args, "preset", "none") or "none").strip().lower() == "mfcp_score_chasing":

        return

    task_name = str(getattr(args, "task_name", "") or "").strip().lower()

    if task_name not in {"mta_multiclass", "mfcp_multiclass"}:

        return



    task_values = {

        "--class_balance": "weighted_sampler_loss",

        "--loss_type": "focal",

        "--focal_gamma": 1.5,

        "--weight_decay": 1e-4,

        "--label_smoothing": 0.03,

        "--early_stop_metric": "val_f1",

        "--early_stop_mode": "max",

        "--lr_scheduler": "reduce",

        "--lr_patience": 2,

        "--grad_clip_norm": 1.0,

    }

    for flag, value in task_values.items():

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

            image_mode=getattr(args, "image_mode", "rgb"),

        )

    else:

        train_image_dir, train_pcap_dir, test_image_dir, test_pcap_dir, resolved_dataset_name = resolve_dataset_dirs(

            args.dataset_root,

            args.dataset_name or None,

        )

    _apply_preset_defaults(args, resolved_dataset_name)

    _apply_task_defaults(args)

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

        fusion_mode=args.fusion_mode,

        use_temporal_sidecar=(not args.no_temporal),

        charbert_mode=args.charbert_mode,

        char_vocab=args.char_vocab,

        char_emb_dim=args.char_emb_dim,

        char_cnn_channels=args.char_cnn_channels,

        char_fusion=args.char_fusion,

        char_fusion_layers=args.char_fusion_layers,

        stacking_level=args.stacking_level,

        stacking_calibration=args.stacking_calibration,

        stacking_threshold_objective=args.stacking_threshold_objective,

        stacking_minority_lambda=args.stacking_minority_lambda,

        stacking_oof_folds=args.stacking_oof_folds,

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

        max_consecutive_invalid_batches=args.max_consecutive_invalid_batches,

        selected_groups=parse_csv_values(args.cic_group) if args.cic_group else None,

        image_mode=getattr(args, "image_mode", "rgb"),

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

    use_temporal_sidecar: bool = True,

    charbert_mode: str = "charaware",

    char_vocab: str = "hex",

    char_emb_dim: int = 32,

    char_cnn_channels: int = 64,

    char_fusion: str = "gated",

    char_fusion_layers: str = "all",

    use_amp: bool = True,

    use_index_cache: bool = True,

    rebuild_index_cache: bool = False,

    class_balance: str = "none",

    loss_type: str = "ce",

    focal_gamma: float = 2.0,

    weight_decay: float = 1e-4,

    label_smoothing: float = 0.0,

    early_stop_metric: str = "val_loss",

    early_stop_mode: str = "auto",

    lr_scheduler_mode: str = "reduce",

    lr_patience: int = 2,

    lr_factor: float = 0.5,

    min_lr: float = 1e-6,

    grad_clip_norm: float = 1.0,

    val_every: int = 1,

    max_consecutive_invalid_batches: int = 128,

    selected_groups: Optional[List[str]] = None,

    image_mode: str = "rgb",

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

    run_logger.info("config: fusion_mode=%s use_temporal=%s image_mode=%s", fusion_mode, use_temporal_sidecar, image_mode)



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

        use_temporal_sidecar=use_temporal_sidecar,

        image_mode=image_mode,

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

        use_temporal_sidecar=use_temporal_sidecar,

        image_mode=image_mode,

    )



    assert train_classes == test_classes, "训练集和测试集类别不一致"

    num_classes = len(train_classes)



    model = initialize_fusion_model(

        num_classes,

        fusion_mode,

        attention_dim=attention_dim,

        charbert_mode=charbert_mode,

        char_vocab=char_vocab,

        char_emb_dim=char_emb_dim,

        char_cnn_channels=char_cnn_channels,

        char_fusion=char_fusion,

        char_fusion_layers=char_fusion_layers,

    )

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

        max_consecutive_invalid_batches=max_consecutive_invalid_batches,

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

            

                                                                            

        heatmap_path = run_dir / "attention_heatmap.png"

        try:

                                                                  

                                                                                

                                                                    

            run_logger.info("Generating attention heatmap...")

                                                                        

        except Exception as e:

            run_logger.warning("Heatmap generation failed: %s", e)



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

        "run_status": history.get("health", {}).get("run_status", "ok"),

        "stop_reason": history.get("health", {}).get("stop_reason", "completed"),

        "health": history.get("health", {}),

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

    use_temporal_sidecar: bool = True,

    charbert_mode: str = "charaware",

    char_vocab: str = "hex",

    char_emb_dim: int = 32,

    char_cnn_channels: int = 64,

    char_fusion: str = "gated",

    char_fusion_layers: str = "all",

    stacking_level: str = "two_level",

    stacking_calibration: str = "temp",

    stacking_threshold_objective: str = "macro_f1_minority_recall",

    stacking_minority_lambda: float = 0.3,

    stacking_oof_folds: int = 5,

    ensemble_tag: Optional[str] = None,

    use_amp: bool = True,

    use_index_cache: bool = True,

    rebuild_index_cache: bool = False,

    class_balance: str = "none",

    loss_type: str = "ce",

    focal_gamma: float = 2.0,

    weight_decay: float = 1e-4,

    label_smoothing: float = 0.0,

    early_stop_metric: str = "val_loss",

    early_stop_mode: str = "auto",

    lr_scheduler_mode: str = "reduce",

    lr_patience: int = 2,

    lr_factor: float = 0.5,

    min_lr: float = 1e-6,

    grad_clip_norm: float = 1.0,

    val_every: int = 1,

    max_consecutive_invalid_batches: int = 128,

    selected_groups: Optional[List[str]] = None,

    image_mode: str = "rgb",

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

    run_logger.info("config: base_fusion_mode=%s use_temporal=%s image_mode=%s", base_fusion_mode, use_temporal_sidecar, image_mode)

    run_logger.info(

        "stacking config: level=%s calibration=%s threshold_objective=%s minority_lambda=%.3f oof_folds=%d",

        stacking_level,

        stacking_calibration,

        stacking_threshold_objective,

        float(stacking_minority_lambda),

        int(stacking_oof_folds),

    )

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

        use_temporal_sidecar=use_temporal_sidecar,

        image_mode=image_mode,

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

        use_temporal_sidecar=use_temporal_sidecar,

        image_mode=image_mode,

    )



    assert train_classes == test_classes, "训练集和测试集类别不一致"

    num_classes = len(train_classes)



    model = initialize_fusion_model(

        num_classes,

        base_fusion_mode,

        attention_dim=attention_dim,

        charbert_mode=charbert_mode,

        char_vocab=char_vocab,

        char_emb_dim=char_emb_dim,

        char_cnn_channels=char_cnn_channels,

        char_fusion=char_fusion,

        char_fusion_layers=char_fusion_layers,

    )

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

        max_consecutive_invalid_batches=max_consecutive_invalid_batches,

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



    meta_train_loader = build_deterministic_meta_loader(train_loader)

    meta_test_loader = build_deterministic_meta_loader(test_loader)

    meta_features, meta_labels = generate_meta_features(

        model.text_encoder,

        model.mobilevit,

        meta_train_loader,

        device,

        fusion_model=model,

    )

    test_meta_features, test_meta_labels = generate_meta_features(

        model.text_encoder,

        model.mobilevit,

        meta_test_loader,

        device,

        fusion_model=model,

    )

    run_logger.info(

        "meta feature extraction completed: train_samples=%s test_samples=%s",

        int(len(meta_labels)),

        int(len(test_meta_labels)),

    )

    sample_weights = build_inverse_frequency_sample_weights(meta_labels)

    class_count = len(train_classes)

    is_mta_task, is_mfcp_task = detect_stacking_special_tasks(

        train_classes=train_classes,

        train_image_dir=train_image_dir,

        train_pcap_dir=train_pcap_dir,

    )

    mfcp_preferred_pair = resolve_pair_class_indices(train_classes, "Dridex", "Trickbot") if is_mfcp_task else None

    mfcp_fallback_pair = resolve_mfcp_pair_class_indices(train_classes) if is_mfcp_task else None

    if is_mfcp_task and mfcp_preferred_pair is None and mfcp_fallback_pair is None:

        run_logger.warning(

            "MFCP pair correction skipped: no preferred/fallback pair found, classes=%s",

            train_classes,

        )

    pair_tuning_objective = "accuracy" if str(stacking_threshold_objective).strip().lower() == "accuracy" else "pair_f1"

    mta_gain_target_classes: List[int] = []

    if is_mta_task and len(meta_labels):

        mta_classes, mta_counts = np.unique(meta_labels, return_counts=True)

        order = np.argsort(mta_counts)

        mta_gain_target_classes = [int(c) for c in mta_classes[order][: min(2, len(mta_classes))].tolist()]



    method_results = []

    successful_method_probs: List[np.ndarray] = []

    successful_oof_probs: List[np.ndarray] = []

    successful_method_weights: List[float] = []

    successful_method_names: List[str] = []

    method_test_probs_map: Dict[str, np.ndarray] = {}

    method_oof_probs_map: Dict[str, np.ndarray] = {}

    for method in methods:

        try:

            def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:

                fold_weights = build_inverse_frequency_sample_weights(train_y)

                fold_model = train_meta_learner(

                    train_x,

                    train_y,

                    method=method,

                    sample_weight=fold_weights,

                )

                _, fold_probs = _predict_with_meta_model(fold_model, valid_x, num_classes=class_count)

                return fold_probs



            oof_probs = compute_oof_predictions(

                features=meta_features,

                labels=meta_labels,

                n_splits=max(2, int(stacking_oof_folds)),

                seed=42,

                fit_predict_fn=_fit_predict,

            )

            meta_model = train_meta_learner(

                meta_features,

                meta_labels,

                method=method,

                sample_weight=sample_weights,

            )

        except ImportError as e:

            run_logger.warning("跳过 %s: %s", method, e)

            method_results.append({"method": method, "skipped": True, "reason": str(e)})

            continue

        except Exception as e:

            run_logger.warning("训练 %s 失败: %s", method, e)

            method_results.append({"method": method, "failed": True, "reason": str(e)})

            continue



        preds, pred_probs = _predict_with_meta_model(meta_model, test_meta_features, num_classes=class_count)

        oof_probs, pred_probs, calibration_info = calibrate_multiclass_probs(

            labels=meta_labels,

            oof_probs=oof_probs,

            test_probs=pred_probs,

            calibration=stacking_calibration,

        )

        method_oof_probs_for_vote = oof_probs

        oof_preds = np.argmax(oof_probs, axis=1)

        oof_acc = accuracy_score(meta_labels, oof_preds) if len(meta_labels) else 0.0

        oof_macro_f1 = f1_score(meta_labels, oof_preds, average="macro") if len(meta_labels) else 0.0

        postprocess: Dict[str, Any] = {"calibration": calibration_info}

        if is_mta_task and len(oof_probs):

            gains = tune_class_gains(labels=meta_labels, probs=oof_probs, target_classes=mta_gain_target_classes)

            pred_probs = apply_class_gains(pred_probs, gains)

            preds = np.argmax(pred_probs, axis=1).astype(np.int64)

            postprocess["mta_class_gains"] = {str(k): float(v) for k, v in gains.items()}

        if is_mfcp_task and len(meta_labels):

            dynamic_pair = select_confusion_pair(

                labels=meta_labels,

                preds=np.argmax(oof_probs, axis=1).astype(np.int64),

                preferred_pair=mfcp_preferred_pair,

            )

            if dynamic_pair is None:

                dynamic_pair = mfcp_fallback_pair

            if dynamic_pair is None:

                run_logger.warning("Skip MFCP pair correction for %s: no valid dynamic/fallback pair", method)

            else:

                pair_class_a, pair_class_b = dynamic_pair

                head = fit_binary_centroid_head(meta_features, meta_labels, class_a=pair_class_a, class_b=pair_class_b)

                pair_alpha = tune_binary_correction_alpha_for_pair(

                    labels=meta_labels,

                    probs=oof_probs,

                    features=meta_features,

                    head=head,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    objective=pair_tuning_objective,

                )

                oof_pair_probs = oof_probs

                if len(oof_probs):

                    oof_pair_preds, oof_pair_probs = apply_binary_correction_for_pair(

                        preds=np.argmax(oof_probs, axis=1).astype(np.int64),

                        probs=oof_probs,

                        features=meta_features,

                        head=head,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        alpha=pair_alpha,

                    )

                    pair_temperature = tune_pair_temperature(

                        labels=meta_labels,

                        probs=oof_pair_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                    )

                    oof_pair_probs = apply_pair_temperature(

                        probs=oof_pair_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        temperature=pair_temperature,

                    )

                    pair_threshold = tune_pair_threshold(

                        labels=meta_labels,

                        probs=oof_pair_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        objective=pair_tuning_objective,

                    )

                    oof_pair_preds = apply_pair_threshold(

                        preds=oof_pair_preds,

                        probs=oof_pair_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        threshold=pair_threshold,

                    )

                    method_oof_probs_for_vote = oof_pair_probs

                else:

                    pair_temperature = 1.0

                    pair_threshold = 0.5

                preds, pred_probs = apply_binary_correction_for_pair(

                    preds=preds,

                    probs=pred_probs,

                    features=test_meta_features,

                    head=head,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    alpha=pair_alpha,

                )

                pred_probs = apply_pair_temperature(

                    probs=pred_probs,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    temperature=pair_temperature,

                )

                preds = apply_pair_threshold(

                    preds=np.argmax(pred_probs, axis=1).astype(np.int64),

                    probs=pred_probs,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    threshold=pair_threshold,

                )

                postprocess["mfcp_binary_pair_correction"] = bool(head is not None)

                postprocess["mfcp_binary_pair_alpha"] = float(pair_alpha)

                postprocess["mfcp_pair_temperature"] = float(pair_temperature)

                postprocess["mfcp_pair_threshold"] = float(pair_threshold)

                postprocess["mfcp_pair_classes"] = [train_classes[pair_class_a], train_classes[pair_class_b]]



        acc = accuracy_score(test_meta_labels, preds) if len(test_meta_labels) else 0.0

        macro_f1 = f1_score(test_meta_labels, preds, average="macro") if len(test_meta_labels) else 0.0

        report = classification_report(test_meta_labels, preds, digits=4) if len(test_meta_labels) else ""

        cm = confusion_matrix(test_meta_labels, preds) if len(test_meta_labels) else np.zeros((0, 0), dtype=int)



        tag = f"{ensemble_tag}_{method}"

        run_logger.info("[%s] acc=%.4f macro_f1=%.4f", tag, acc, macro_f1)

        oof_test_gap = float(oof_macro_f1 - macro_f1)

        if oof_test_gap > 0.12:

            run_logger.warning(

                "[%s] OOF-test gap is large: oof_macro_f1=%.4f test_macro_f1=%.4f gap=%.4f",

                tag,

                oof_macro_f1,

                macro_f1,

                oof_test_gap,

            )

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

                "oof_acc": oof_acc,

                "oof_macro_f1": oof_macro_f1,

                "report": report,

                "confusion_matrix": cm,

                "confusion_matrix_path": method_cm_path.name,

                "report_path": method_report_path.name,

                "meta_model_path": meta_model_path.name if meta_model_path else None,

                "postprocess": postprocess,

            }

        )

        successful_method_probs.append(pred_probs)

        successful_oof_probs.append(method_oof_probs_for_vote)

        successful_method_names.append(method)

        method_test_probs_map[method] = pred_probs

        method_oof_probs_map[method] = method_oof_probs_for_vote

        oof_weight_preds = np.argmax(method_oof_probs_for_vote, axis=1).astype(np.int64)

        oof_weight_macro_f1 = f1_score(meta_labels, oof_weight_preds, average="macro") if len(meta_labels) else 0.0

        successful_method_weights.append(float(max(oof_weight_macro_f1, 1e-6)))



    effective_level = resolve_effective_stacking_level(

        requested_level=stacking_level,

        successful_methods=successful_method_names,

    )



    if effective_level == "two_level":

        minority_classes: List[int] = []

        if len(meta_labels):

            class_ids, class_counts = np.unique(np.asarray(meta_labels, dtype=np.int64), return_counts=True)

            if len(class_ids):

                sorted_idx = np.argsort(class_counts)

                take = min(2, len(class_ids))

                minority_classes = [int(class_ids[i]) for i in sorted_idx[:take].tolist()]

        level2_out = run_level2_blender_oof(

            labels=meta_labels,

            level1_oof_probs=method_oof_probs_map,

            level1_test_probs=method_test_probs_map,

            oof_folds=stacking_oof_folds,

            minority_classes=minority_classes,

            threshold_objective=stacking_threshold_objective,

            minority_lambda=stacking_minority_lambda,

        )

        two_level_preds = np.asarray(level2_out.get("preds", np.array([], dtype=np.int64)), dtype=np.int64)

        two_level_probs = np.asarray(level2_out.get("probs", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)

        two_level_oof_probs = np.asarray(level2_out.get("oof_probs", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)

        two_level_thresholds = np.asarray(level2_out.get("thresholds", np.array([], dtype=np.float64)), dtype=np.float64)

        two_level_objective_value = float(level2_out.get("threshold_objective_value", 0.0))



        if len(test_meta_labels) and two_level_probs.shape[0] == len(test_meta_labels):

            before_preds = np.argmax(two_level_probs, axis=1).astype(np.int64)

            if minority_classes:

                before_minor = float(

                    recall_score(

                        test_meta_labels,

                        before_preds,

                        labels=minority_classes,

                        average="macro",

                        zero_division=0,

                    )

                )

                after_minor = float(

                    recall_score(

                        test_meta_labels,

                        two_level_preds,

                        labels=minority_classes,

                        average="macro",

                        zero_division=0,

                    )

                )

            else:

                before_minor = 0.0

                after_minor = 0.0

            two_level_acc = accuracy_score(test_meta_labels, two_level_preds)

            two_level_macro_f1 = f1_score(test_meta_labels, two_level_preds, average="macro")

            two_level_report = classification_report(test_meta_labels, two_level_preds, digits=4)

            two_level_cm = confusion_matrix(test_meta_labels, two_level_preds)



            oof_two_preds = np.argmax(two_level_oof_probs, axis=1).astype(np.int64) if len(two_level_oof_probs) else np.array([], dtype=np.int64)

            oof_two_acc = accuracy_score(meta_labels, oof_two_preds) if len(oof_two_preds) else 0.0

            oof_two_macro_f1 = f1_score(meta_labels, oof_two_preds, average="macro") if len(oof_two_preds) else 0.0

            two_level_oof_test_gap = float(oof_two_macro_f1 - two_level_macro_f1)

            if two_level_oof_test_gap > 0.12:

                run_logger.warning(

                    "[%s_two_level_blender] OOF-test gap is large: oof_macro_f1=%.4f test_macro_f1=%.4f gap=%.4f",

                    ensemble_tag,

                    float(oof_two_macro_f1),

                    float(two_level_macro_f1),

                    float(two_level_oof_test_gap),

                )



            two_level_cm_path = run_dir / "confusion_matrix_two_level_blender.png"

            plot_confusion(two_level_cm, train_classes, two_level_cm_path, f"Confusion Matrix - {ensemble_tag}_two_level_blender")

            log_saved(run_logger, two_level_cm_path, "confusion_matrix_two_level_blender")



            two_level_report_path = run_dir / "report_two_level_blender.md"

            save_report_md(

                two_level_report_path,

                title=f"融合方式: {base_fusion_mode}+stacking (two_level_blender)",

                acc=two_level_acc,

                macro_f1=two_level_macro_f1,

                report=two_level_report,

                cm=two_level_cm,

                confusion_image=two_level_cm_path.name,

                curve_image=curve_path.name,

            )

            log_saved(run_logger, two_level_report_path, "report_two_level_blender")



            calibration_metrics = compute_calibration_metrics(labels=meta_labels, probs=two_level_oof_probs) if len(two_level_oof_probs) else {"ece": 0.0, "brier": 0.0}

            postprocess = build_two_level_postprocess_payload(

                stacking_level="two_level",

                calibration={

                    "method": str(stacking_calibration),

                    "ece": float(calibration_metrics.get("ece", 0.0)),

                    "brier": float(calibration_metrics.get("brier", 0.0)),

                },

                thresholds=two_level_thresholds,

                minority_classes=minority_classes,

                minority_recall_before=before_minor,

                minority_recall_after=after_minor,

                threshold_objective=stacking_threshold_objective,

                objective_value=two_level_objective_value,

                oof_test_gap=two_level_oof_test_gap,

            )



            method_results.append(

                {

                    "method": "two_level_blender",

                    "acc": float(two_level_acc),

                    "macro_f1": float(two_level_macro_f1),

                    "oof_acc": float(oof_two_acc),

                    "oof_macro_f1": float(oof_two_macro_f1),

                    "report": two_level_report,

                    "confusion_matrix": two_level_cm,

                    "confusion_matrix_path": two_level_cm_path.name,

                    "report_path": two_level_report_path.name,

                    "meta_model_path": None,

                    "postprocess": postprocess,

                }

            )

            run_logger.info(

                "[%s_two_level_blender] acc=%.4f macro_f1=%.4f oof_macro_f1=%.4f",

                ensemble_tag,

                float(two_level_acc),

                float(two_level_macro_f1),

                float(oof_two_macro_f1),

            )

        else:

            run_logger.warning("two-level blender skipped: invalid level2 outputs")



    single_layer_baseline = build_single_layer_baseline_result(method_results)

    if single_layer_baseline is not None:

        method_results.append(single_layer_baseline)



    if len(successful_method_probs) >= 2:

        vote_probs, vote_preds = weighted_soft_voting(successful_method_probs, successful_method_weights)

        vote_postprocess: Dict[str, Any] = {

            "weights": [float(w) for w in successful_method_weights],

            "members": [m for m in successful_method_names],

        }

        if len(successful_oof_probs) >= 2:

            oof_vote_probs, oof_vote_preds = weighted_soft_voting(successful_oof_probs, successful_method_weights)

            oof_vote_acc = accuracy_score(meta_labels, oof_vote_preds) if len(meta_labels) else 0.0

            oof_vote_macro_f1 = f1_score(meta_labels, oof_vote_preds, average="macro") if len(meta_labels) else 0.0

        else:

            oof_vote_probs = np.zeros((0, 0), dtype=np.float64)

            oof_vote_acc = 0.0

            oof_vote_macro_f1 = 0.0



        if is_mta_task and len(oof_vote_probs):

            vote_gains = tune_class_gains(labels=meta_labels, probs=oof_vote_probs, target_classes=mta_gain_target_classes)

            vote_probs = apply_class_gains(vote_probs, vote_gains)

            vote_preds = np.argmax(vote_probs, axis=1).astype(np.int64)

            vote_postprocess["mta_class_gains"] = {str(k): float(v) for k, v in vote_gains.items()}

        if is_mfcp_task and len(meta_labels):

            dynamic_pair = select_confusion_pair(

                labels=meta_labels,

                preds=np.argmax(oof_vote_probs, axis=1).astype(np.int64) if len(oof_vote_probs) else np.array([], dtype=np.int64),

                preferred_pair=mfcp_preferred_pair,

            )

            if dynamic_pair is None:

                dynamic_pair = mfcp_fallback_pair

            if dynamic_pair is None:

                run_logger.warning("Skip MFCP pair correction for soft_voting: no valid dynamic/fallback pair")

            else:

                pair_class_a, pair_class_b = dynamic_pair

                vote_head = fit_binary_centroid_head(meta_features, meta_labels, class_a=pair_class_a, class_b=pair_class_b)

                vote_pair_alpha = tune_binary_correction_alpha_for_pair(

                    labels=meta_labels,

                    probs=oof_vote_probs,

                    features=meta_features,

                    head=vote_head,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    objective=pair_tuning_objective,

                ) if len(oof_vote_probs) else 0.0

                if len(oof_vote_probs):

                    oof_vote_preds = np.argmax(oof_vote_probs, axis=1).astype(np.int64)

                    oof_vote_preds, oof_vote_probs = apply_binary_correction_for_pair(

                        preds=oof_vote_preds,

                        probs=oof_vote_probs,

                        features=meta_features,

                        head=vote_head,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        alpha=vote_pair_alpha,

                    )

                    vote_pair_temperature = tune_pair_temperature(

                        labels=meta_labels,

                        probs=oof_vote_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                    )

                    oof_vote_probs = apply_pair_temperature(

                        probs=oof_vote_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        temperature=vote_pair_temperature,

                    )

                    vote_pair_threshold = tune_pair_threshold(

                        labels=meta_labels,

                        probs=oof_vote_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        objective=pair_tuning_objective,

                    )

                    oof_vote_preds = apply_pair_threshold(

                        preds=oof_vote_preds,

                        probs=oof_vote_probs,

                        class_a=pair_class_a,

                        class_b=pair_class_b,

                        threshold=vote_pair_threshold,

                    )

                else:

                    vote_pair_temperature = 1.0

                    vote_pair_threshold = 0.5

                vote_preds, vote_probs = apply_binary_correction_for_pair(

                    preds=vote_preds,

                    probs=vote_probs,

                    features=test_meta_features,

                    head=vote_head,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    alpha=vote_pair_alpha,

                )

                vote_probs = apply_pair_temperature(

                    probs=vote_probs,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    temperature=vote_pair_temperature,

                )

                vote_preds = apply_pair_threshold(

                    preds=np.argmax(vote_probs, axis=1).astype(np.int64),

                    probs=vote_probs,

                    class_a=pair_class_a,

                    class_b=pair_class_b,

                    threshold=vote_pair_threshold,

                )

                vote_postprocess["mfcp_binary_pair_correction"] = bool(vote_head is not None)

                vote_postprocess["mfcp_binary_pair_alpha"] = float(vote_pair_alpha)

                vote_postprocess["mfcp_pair_temperature"] = float(vote_pair_temperature)

                vote_postprocess["mfcp_pair_threshold"] = float(vote_pair_threshold)

                vote_postprocess["mfcp_pair_classes"] = [train_classes[pair_class_a], train_classes[pair_class_b]]



        vote_acc = accuracy_score(test_meta_labels, vote_preds) if len(test_meta_labels) else 0.0

        vote_macro_f1 = f1_score(test_meta_labels, vote_preds, average="macro") if len(test_meta_labels) else 0.0

        vote_report = classification_report(test_meta_labels, vote_preds, digits=4) if len(test_meta_labels) else ""

        vote_cm = confusion_matrix(test_meta_labels, vote_preds) if len(test_meta_labels) else np.zeros((0, 0), dtype=int)

        vote_tag = f"{ensemble_tag}_soft_voting"



        run_logger.info("[%s] acc=%.4f macro_f1=%.4f", vote_tag, vote_acc, vote_macro_f1)

        if vote_report:

            run_logger.info("[%s] 分类报告:\n%s", vote_tag, vote_report)

        run_logger.info("[%s] 混淆矩阵:\n%s", vote_tag, vote_cm)



        vote_cm_path = run_dir / "confusion_matrix_soft_voting.png"

        plot_confusion(vote_cm, train_classes, vote_cm_path, f"Confusion Matrix - {vote_tag}")

        log_saved(run_logger, vote_cm_path, "confusion_matrix_soft_voting")



        vote_report_path = run_dir / "report_soft_voting.md"

        save_report_md(

            vote_report_path,

            title=f"融合方式: {base_fusion_mode}+stacking (soft_voting)",

            acc=vote_acc,

            macro_f1=vote_macro_f1,

            report=vote_report,

            cm=vote_cm,

            confusion_image=vote_cm_path.name,

            curve_image=curve_path.name,

        )

        log_saved(run_logger, vote_report_path, "report_soft_voting")



        method_results.append(

            {

                "method": "soft_voting",

                "acc": vote_acc,

                "macro_f1": vote_macro_f1,

                "oof_acc": oof_vote_acc,

                "oof_macro_f1": oof_vote_macro_f1,

                "report": vote_report,

                "confusion_matrix": vote_cm,

                "confusion_matrix_path": vote_cm_path.name,

                "report_path": vote_report_path.name,

                "meta_model_path": None,

                "postprocess": vote_postprocess,

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

        "run_status": history.get("health", {}).get("run_status", "ok"),

        "stop_reason": history.get("health", {}).get("stop_reason", "completed"),

        "health": history.get("health", {}),

        "base_eval": {

            "loss": base_eval["loss"],

            "acc": base_eval["acc"],

            "macro_f1": base_eval["macro_f1"],

            "report": base_eval["report"],

            "confusion_matrix": base_eval["cm"],

            "per_class_f1": base_eval["per_class_f1"],

        },

        "stacking": {

            "requested_level": stacking_level,

            "effective_level": effective_level,

            "calibration": stacking_calibration,

            "threshold_objective": stacking_threshold_objective,

            "minority_lambda": float(stacking_minority_lambda),

            "oof_folds": int(stacking_oof_folds),

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



                                                                

    try:

        base_recall = base_eval.get("per_class_f1", {})                              

                                             

        best_stacked = build_single_layer_baseline_result(method_results)

        if best_stacked:

            stacked_report = best_stacked.get("report", "")

                                                                             

            gain_path = run_dir / "minority_recall_gain.json"

            with open(gain_path, "w", encoding="utf-8") as f:

                json.dump({"status": "ready_for_extraction", "base_f1": base_eval["macro_f1"], "stacked_f1": best_stacked["acc"]}, f)

            run_logger.info("Saved minority recall gain data: %s", gain_path)

    except Exception as e:

        run_logger.warning("Failed to export recall gain: %s", e)



    run_logger.info("done stacking: run_dir=%s log=%s", run_dir, log_path)

    print(f"[{ensemble_tag}] done. run_dir={run_dir}, saved_base={base_model_path}")

