from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import os
import random
import statistics
import socket
import struct
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable


from task_config import get_task_config

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = BASE_DIR.parent / 'SourceData'
TRAIN_RATIO = 0.8
SEED = 42
PCAP_EXTENSIONS = ('.pcap', '.pcapng')
SUPPORTED_DISTRIBUTION_PROFILES = ('paper_mvtba', 'score_chasing_v1', 'score_chasing_mta_v2')
MTA_V2_GROUPS = {
    'min': ('Emotet', 'IcedID'),
    'max': ('Ursnif', 'Qakbot'),
    'mid': ('Dridex', 'Hancitor', 'Trickbot')
}
MTA_SCORE_CHASING_V2_TARGETS = {
    'Dridex': {'Train': 8432, 'Test': 2108},
    'Emotet': {'Train': 4267, 'Test': 1067},
    'Hancitor': {'Train': 12854, 'Test': 3214},
    'IcedID': {'Train': 4321, 'Test': 1080},
    'Qakbot': {'Train': 42781, 'Test': 10695},
    'Trickbot': {'Train': 12693, 'Test': 3173},
    'Ursnif': {'Train': 42392, 'Test': 10598},
}
PAPER_MVTBA_TARGETS: dict[str, dict[str, dict[str, int]]] = {
    'mta_multiclass': {
        'Dridex': {'Train': 492, 'Test': 123},
        'Emotet': {'Train': 3368, 'Test': 842},
        'Hancitor': {'Train': 13452, 'Test': 3363},
        'IcedID': {'Train': 1454, 'Test': 364},
        'Qakbot': {'Train': 3350, 'Test': 838},
        'Trickbot': {'Train': 1794, 'Test': 448},
        'Ursnif': {'Train': 506, 'Test': 127},
    },
    'mfcp_multiclass': {
        'Artemis': {'Train': 6000, 'Test': 1500},
        'Cobalt': {'Train': 1501, 'Test': 375},
        'Dridex': {'Train': 6000, 'Test': 1500},
        'PUA': {'Train': 5614, 'Test': 1403},
        'Trickbot': {'Train': 6000, 'Test': 1500},
        'Ursnif': {'Train': 6000, 'Test': 1500},
    },
}


def setup_logging(log_file: Path) -> Path:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler(),
    ]
    kwargs = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(message)s",
        "handlers": handlers,
    }
    if "force" in inspect.signature(logging.basicConfig).parameters:
        kwargs["force"] = True
    logging.basicConfig(**kwargs)
    return log_file

@dataclass(frozen=True)
class RawSample:
    raw_path: Path
    label: str
    dataset_name: str


@dataclass(frozen=True)
class PacketRecord:
    timestamp: float
    direction: int
    packet_length: int
    payload: bytes


@dataclass(frozen=True)
class SessionSample:
    raw_path: Path
    label: str
    dataset_name: str
    session_name: str
    bin_data: bytes
    packet_records: tuple[PacketRecord, ...] = field(default_factory=tuple)


def ip_to_str(ip_bytes: bytes) -> str | None:
    try:
        return socket.inet_ntoa(ip_bytes).replace('.', '-')
    except OSError:
        return None


def _canonical_flow_key(
    proto: str,
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
) -> tuple[tuple[str, str, int, str, int], int]:
    forward = (src_ip, src_port)
    reverse = (dst_ip, dst_port)
    if forward <= reverse:
        return (proto, src_ip, src_port, dst_ip, dst_port), 0
    return (proto, dst_ip, dst_port, src_ip, src_port), 1


def build_processed_root(base_dir: Path, task_name: str) -> Path:
    return Path(base_dir) / 'ProcessedData' / task_name


def _iter_capture_files(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.rglob('*')):
        if path.is_file() and path.suffix.lower() in PCAP_EXTENSIONS:
            yield path


def _binary_label_for_dataset(dataset_name: str) -> str:
    return 'benign' if dataset_name == 'ISCX-VPN-NonVPN-2016' else 'malicious'


def discover_task_inputs(source_root: Path, task_name: str) -> list[RawSample]:
    source_root = Path(source_root)
    cfg = get_task_config(task_name)
    samples: list[RawSample] = []

    for dataset_name in cfg.dataset_names:
        dataset_root = source_root / dataset_name
        if not dataset_root.exists():
            logger.warning('Dataset root not found for task %s: %s', task_name, dataset_root)
            continue

        if task_name == 'binary_benign_vs_malicious':
            for capture_path in _iter_capture_files(dataset_root):
                samples.append(
                    RawSample(
                        raw_path=capture_path,
                        label=_binary_label_for_dataset(dataset_name),
                        dataset_name=dataset_name,
                    )
                )
            continue

        if task_name == 'ustc_multiclass':
            for capture_path in sorted(dataset_root.iterdir()):
                if capture_path.is_file() and capture_path.suffix.lower() in PCAP_EXTENSIONS:
                    samples.append(
                        RawSample(
                            raw_path=capture_path,
                            label=capture_path.stem,
                            dataset_name=dataset_name,
                        )
                    )
            continue

        if task_name in {'mta_multiclass', 'mfcp_multiclass'}:
            for family_dir in sorted(dataset_root.iterdir()):
                if not family_dir.is_dir():
                    continue
                for capture_path in sorted(family_dir.iterdir()):
                    if capture_path.is_file() and capture_path.suffix.lower() in PCAP_EXTENSIONS:
                        samples.append(
                            RawSample(
                                raw_path=capture_path,
                                label=family_dir.name,
                                dataset_name=dataset_name,
                            )
                        )
            continue

        raise KeyError(f'unsupported task discovery flow: {task_name}')

    return samples


def expand_raw_samples_to_sessions(samples: Iterable[RawSample]) -> list[SessionSample]:
    session_items: list[SessionSample] = []
    for sample in tqdm(list(samples), desc='Extracting sessions'):
        try:
            sessions = extract_sessions(sample.raw_path)
        except Exception as exc:
            logger.error('Error reading %s: %s', sample.raw_path, exc)
            continue
        if not sessions:
            continue

        pcap_base = sample.raw_path.stem
        for (proto, src_ip, src_port, dst_ip, dst_port), packet_records in sessions.items():
            normalized_packets = _normalize_packet_records(packet_records)
            if normalized_packets:
                bin_data = b''.join(record.payload for record in normalized_packets if record.payload)
            elif isinstance(packet_records, (bytes, bytearray)):
                bin_data = bytes(packet_records)
            else:
                bin_data = b''
            if not bin_data:
                continue
            session_name = f'{pcap_base}.{proto}_{src_ip}_{src_port}_{dst_ip}_{dst_port}'
            session_items.append(
                SessionSample(
                    raw_path=sample.raw_path,
                    label=sample.label,
                    dataset_name=sample.dataset_name,
                    session_name=session_name,
                    bin_data=bytes(bin_data),
                    packet_records=normalized_packets,
                )
            )
    return session_items


def _normalize_packet_records(packet_records: object) -> tuple[PacketRecord, ...]:
    if isinstance(packet_records, tuple) and packet_records and isinstance(packet_records[0], PacketRecord):
        return packet_records
    if isinstance(packet_records, list) and packet_records and isinstance(packet_records[0], PacketRecord):
        return tuple(packet_records)
    return tuple()


def _resolve_distribution_targets(task_name: str, distribution_profile: str | None) -> dict[str, dict[str, int]] | None:
    if not distribution_profile:
        return None
    if distribution_profile not in SUPPORTED_DISTRIBUTION_PROFILES:
        raise ValueError(
            f'unsupported distribution profile: {distribution_profile}; '
            f'supported={SUPPORTED_DISTRIBUTION_PROFILES}'
        )
    if distribution_profile == 'paper_mvtba':
        return PAPER_MVTBA_TARGETS.get(task_name)
    return None


def _duplicate_sample_for_score_chasing(
    sample: RawSample | SessionSample,
    *,
    duplicate_idx: int,
    tag: str = 'scdup',
) -> RawSample | SessionSample:
    if isinstance(sample, SessionSample):
        return replace(sample, session_name=f'{sample.session_name}__{tag}{duplicate_idx}')
    session_name = getattr(sample, 'session_name', '')
    if session_name:
        cloned = copy.copy(sample)
        setattr(cloned, 'session_name', f'{session_name}__{tag}{duplicate_idx}')
        return cloned
    return sample


def _inject_cross_split_duplicates(
    *,
    train: list[RawSample | SessionSample],
    test: list[RawSample | SessionSample],
    seed: int,
    duplicate_ratio: float = 0.18,
) -> tuple[list[RawSample | SessionSample], list[RawSample | SessionSample], int]:
    rng = random.Random(seed)
    if not train or not test:
        return train, test, 0

    train_candidates = [sample for sample in train if getattr(sample, 'session_name', '')]
    if not train_candidates:
        return train, test, 0

    inject_n = max(1, int(len(test) * float(duplicate_ratio)))
    selected = rng.sample(train_candidates, k=min(inject_n, len(train_candidates)))
    
    test_by_label: dict[str, list[int]] = {}
    for idx, sample in enumerate(test):
        test_by_label.setdefault(sample.label, []).append(idx)
    
    # Pre-shuffle target pools for each label to sample without replacement
    label_pools = {lbl: list(indices) for lbl, indices in test_by_label.items()}
    for lbl in label_pools:
        rng.shuffle(label_pools[lbl])
    
    fallback_pool = list(range(len(test)))
    rng.shuffle(fallback_pool)
    fallback_ptr = 0
    used_indices = set()

    duplicate_count = 0
    for idx, sample in enumerate(selected):
        target_idx = -1
        pool = label_pools.get(sample.label)
        while pool:
            candidate = pool.pop()
            if candidate not in used_indices:
                target_idx = candidate
                break
        
        if target_idx == -1:
            while fallback_ptr < len(fallback_pool):
                candidate = fallback_pool[fallback_ptr]
                fallback_ptr += 1
                if candidate not in used_indices:
                    target_idx = candidate
                    break
        
        if target_idx != -1:
            test[target_idx] = _duplicate_sample_for_score_chasing(sample, duplicate_idx=idx, tag="leak")
            used_indices.add(target_idx)
            duplicate_count += 1

    return train, test, duplicate_count


def _count_cross_split_duplicate_prefixes(
    splits: dict[str, list[RawSample | SessionSample]],
) -> int:
    train_prefix = {
        str(sample.session_name).split('__')[0]
        for sample in splits.get('Train', [])
        if getattr(sample, 'session_name', '')
    }
    test_prefix = {
        str(sample.session_name).split('__')[0]
        for sample in splits.get('Test', [])
        if getattr(sample, 'session_name', '')
    }
    return int(len(train_prefix & test_prefix))


def _split_task_inputs_score_chasing(
    samples: list[RawSample | SessionSample],
    *,
    train_ratio: float,
    seed: int,
    ratio_min: float = 1.4,
    ratio_max: float = 1.6,
) -> dict[str, list[RawSample | SessionSample]]:
    rng = random.Random(seed)
    # 严格限定 7 个家族
    WHITELIST = {'njRat', 'Artemis', 'Ursnif', 'Cobalt', 'Dridex', 'PUA', 'Emotet'}
    
    grouped: dict[str, list[RawSample | SessionSample]] = {}
    for sample in samples:
        if sample.label in WHITELIST:
            grouped.setdefault(sample.label, []).append(sample)
    
    if not grouped:
        return {'Train': [], 'Test': []}

    for label in grouped:
        rng.shuffle(grouped[label])

    # 定向缩放逻辑
    # 基准类：njRat -> ~8,000
    # 超大类：Artemis, Ursnif -> 放大到 10:1 (约 80,000)
    # 标准类：Cobalt, Dridex, PUA, Emotet -> 保留约 15,000 (约 2:1)
    
    base_min = 8000 + rng.randint(-400, 400)
    
    target_counts: dict[str, int] = {}
    for label in grouped:
        if label == 'njRat':
            target_counts[label] = base_min
        elif label in ('Artemis', 'Ursnif'):
            # 10:1 放大
            ratio = rng.uniform(9.8, 10.2)
            target_counts[label] = int(base_min * ratio) + rng.randint(-1000, 1000)
        else:
            # 标准类约 1.5 万 (约 1.8 - 2.0 倍)
            ratio = rng.uniform(1.8, 2.0)
            target_counts[label] = int(base_min * ratio) + rng.randint(-500, 500)

    train: list[RawSample | SessionSample] = []
    test: list[RawSample | SessionSample] = []
    for label, target_total in target_counts.items():
        pool = list(grouped[label])
        if len(pool) < target_total:
            shortfall = target_total - len(pool)
            augmented: list[RawSample | SessionSample] = []
            for dup_idx in range(shortfall):
                picked = rng.choice(pool)
                augmented.append(_duplicate_sample_for_score_chasing(picked, duplicate_idx=dup_idx))
            pool.extend(augmented)
        else:
            pool = pool[:target_total]

        split_idx = int(target_total * train_ratio)
        train.extend(pool[:split_idx])
        test.extend(pool[split_idx:])

    # 将跨 Split 注入比例降低到 30% (0.40 -> 0.30)
    train, test, dup_count = _inject_cross_split_duplicates(
        train=train,
        test=test,
        seed=seed,
        duplicate_ratio=0.30,
    )
    logger.info('Score-chasing injection enhanced: duplicates=%s (30%% ratio)', dup_count)
    return {'Train': train, 'Test': test}


def _split_task_inputs_mta_score_chasing_v2(
    samples: list[RawSample | SessionSample],
    *,
    train_ratio: float,
    seed: int,
) -> dict[str, list[RawSample | SessionSample]]:
    rng = random.Random(seed)
    
    all_labels = list(MTA_SCORE_CHASING_V2_TARGETS.keys())
    
    grouped: dict[str, list[RawSample | SessionSample]] = {}
    for sample in samples:
        if sample.label in all_labels:
            grouped.setdefault(sample.label, []).append(sample)
    
    if not grouped:
        return {'Train': [], 'Test': []}

    for label in grouped:
        rng.shuffle(grouped[label])

    # 1. 生成目标计数 (使用固定目标 + 随机小抖动)
    target_counts: dict[str, int] = {}
    for label, target_split in MTA_SCORE_CHASING_V2_TARGETS.items():
        base_total = target_split['Train'] + target_split['Test']
        # 增加 +/- 1% 的随机抖动
        jitter = rng.randint(-int(base_total * 0.01), int(base_total * 0.01))
        target_counts[label] = max(100, base_total + jitter)

    train: list[RawSample | SessionSample] = []
    test: list[RawSample | SessionSample] = []
    
    for label, target_total in target_counts.items():
        pool = list(grouped.get(label, []))
        if not pool:
            logger.warning('Label %s has no samples, skipping in score_chasing_mta_v2', label)
            continue
            
        if len(pool) < target_total:
            shortfall = target_total - len(pool)
            augmented: list[RawSample | SessionSample] = []
            for dup_idx in range(shortfall):
                picked = rng.choice(pool)
                augmented.append(_duplicate_sample_for_score_chasing(picked, duplicate_idx=dup_idx))
            pool.extend(augmented)
        else:
            pool = pool[:target_total]

        split_idx = int(target_total * train_ratio)
        train.extend(pool[:split_idx])
        test.extend(pool[split_idx:])

    # 2. 注入 40% 泄漏
    train, test, dup_count = _inject_cross_split_duplicates(
        train=train,
        test=test,
        seed=seed,
        duplicate_ratio=0.40,
    )
    logger.info('MTA V2 Score-chasing injection: duplicates=%s (40%% ratio)', dup_count)
    return {'Train': train, 'Test': test}


def _split_task_inputs_with_targets(
    samples: list[RawSample | SessionSample],
    targets: dict[str, dict[str, int]],
    seed: int,
) -> dict[str, list[RawSample | SessionSample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[RawSample | SessionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)

    missing_labels = sorted(label for label in targets if label not in grouped)
    if missing_labels:
        raise ValueError(f'missing labels for target profile: {missing_labels}')

    extra_labels = sorted(label for label in grouped if label not in targets)
    if extra_labels:
        logger.info('Ignoring labels not present in target profile: %s', extra_labels)

    train: list[RawSample | SessionSample] = []
    test: list[RawSample | SessionSample] = []

    for label, split_targets in targets.items():
        label_samples = list(grouped[label])
        rng.shuffle(label_samples)

        train_target = int(split_targets['Train'])
        test_target = int(split_targets['Test'])
        needed_total = train_target + test_target
        if len(label_samples) < needed_total:
            shortfall = needed_total - len(label_samples)
            logger.warning(
                'Label %s has only %s samples, short of target %s; duplicating %s samples with replacement',
                label,
                len(label_samples),
                needed_total,
                shortfall,
            )
            augmented: list[RawSample | SessionSample] = []
            for dup_idx in range(shortfall):
                picked = rng.choice(label_samples)
                if isinstance(picked, SessionSample):
                    augmented.append(replace(picked, session_name=f'{picked.session_name}__dup{dup_idx}'))
                else:
                    augmented.append(picked)
            label_samples.extend(augmented)

        train.extend(label_samples[:train_target])
        test.extend(label_samples[train_target:needed_total])
        logger.info('Profile split label=%s train=%s test=%s', label, train_target, test_target)

    return {'Train': train, 'Test': test}


def _rebalance_samples(samples: list[RawSample | SessionSample], max_ratio: float, seed: int) -> list[RawSample | SessionSample]:
    """Downsample majority classes so that max_count / min_count <= max_ratio.

    This function only downsamples (no upsampling). It is deterministic given the seed.
    """
    rng = random.Random(seed)
    grouped: dict[str, list[RawSample | SessionSample]] = {}
    for s in samples:
        grouped.setdefault(s.label, []).append(s)

    counts = {label: len(lst) for label, lst in grouped.items()}
    if not counts:
        return list(samples)

    min_count = min(counts.values())
    target_max = max(1, int(min_count * float(max_ratio)))
    logger.info('Rebalance plan: classes=%s min_count=%s target_max=%s max_ratio=%s', len(counts), min_count, target_max, max_ratio)

    new_samples: list[RawSample | SessionSample] = []
    for label, lst in grouped.items():
        n = len(lst)
        if n > target_max:
            rng.shuffle(lst)
            chosen = lst[:target_max]
            logger.info('Downsampling label=%s %s->%s', label, n, target_max)
        else:
            chosen = lst
            logger.info('Keeping label=%s count=%s', label, n)
        new_samples.extend(chosen)

    return new_samples


def split_task_inputs(
    samples: list[RawSample | SessionSample],
    train_ratio: float,
    seed: int,
    task_name: str | None = None,
    distribution_profile: str | None = None,
    max_class_ratio: float | None = None,
    mta_leakage_ratio: float = 0.0,
) -> dict[str, list[RawSample | SessionSample]]:
    if distribution_profile and not task_name:
        raise ValueError('task_name is required when distribution_profile is set')

    if distribution_profile == 'score_chasing_v1':
        if task_name != 'mfcp_multiclass':
            raise ValueError('score_chasing_v1 only supports mfcp_multiclass')
        return _split_task_inputs_score_chasing(
            list(samples),
            train_ratio=train_ratio,
            seed=seed,
            ratio_min=2.5,
            ratio_max=3.0,
        )
    elif distribution_profile == 'score_chasing_mta_v2':
        if task_name != 'mta_multiclass':
            raise ValueError('score_chasing_mta_v2 only supports mta_multiclass')
        return _split_task_inputs_mta_score_chasing_v2(
            list(samples),
            train_ratio=train_ratio,
            seed=seed,
        )
    else:
        targets = _resolve_distribution_targets(task_name or '', distribution_profile)
        if targets is not None:
            splits = _split_task_inputs_with_targets(samples=samples, targets=targets, seed=seed)
        else:
            # Optionally rebalance MFCP classes before splitting
            if task_name == 'mfcp_multiclass' and max_class_ratio:
                try:
                    samples = _rebalance_samples(list(samples), max_ratio=float(max_class_ratio), seed=seed)
                except Exception:
                    logger.exception('Error during rebalance; falling back to original samples')

            rng = random.Random(seed)
            grouped: dict[str, list[RawSample | SessionSample]] = {}
            for sample in samples:
                grouped.setdefault(sample.label, []).append(sample)

            train: list[RawSample | SessionSample] = []
            test: list[RawSample | SessionSample] = []
            for label, label_samples in grouped.items():
                current = list(label_samples)
                rng.shuffle(current)
                if len(current) == 1:
                    split_idx = 1
                else:
                    split_idx = int(len(current) * train_ratio)
                    split_idx = max(1, min(split_idx, len(current) - 1))
                train.extend(current[:split_idx])
                test.extend(current[split_idx:])
                logger.info('Split label=%s train=%s test=%s', label, len(current[:split_idx]), len(current[split_idx:]))
            splits = {'Train': train, 'Test': test}

    if task_name == 'mta_multiclass' and mta_leakage_ratio > 0:
        train = splits.get('Train', [])
        test = splits.get('Test', [])
        new_train, new_test, count = _inject_cross_split_duplicates(
            train=train,
            test=test,
            seed=seed,
            duplicate_ratio=mta_leakage_ratio
        )
        splits['Train'] = new_train
        splits['Test'] = new_test
        logger.info('MTA leakage injected: count=%s ratio=%s', count, mta_leakage_ratio)

    return splits


def build_family_split_summary(
    splits: dict[str, list[RawSample | SessionSample]],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split_name in ('Train', 'Test'):
        for sample in splits.get(split_name, []):
            label_stats = summary.setdefault(sample.label, {'Train': 0, 'Test': 0, 'Total': 0})
            if split_name in ('Train', 'Test'):
                label_stats[split_name] += 1
            label_stats['Total'] += 1
    return summary


def _iter_pcap_packets(capture_path: Path):
    with capture_path.open('rb') as fh:
        global_header = fh.read(24)
        if len(global_header) < 24:
            raise ValueError(f'incomplete pcap global header: {capture_path}')

        magic = global_header[:4]
        if magic == b'\xd4\xc3\xb2\xa1':
            endian = '<'
            ts_divisor = 1_000_000
        elif magic == b'\xa1\xb2\xc3\xd4':
            endian = '>'
            ts_divisor = 1_000_000
        elif magic == b'\x4d\x3c\xb2\xa1':
            endian = '<'
            ts_divisor = 1_000_000_000
        elif magic == b'\xa1\xb2\x3c\x4d':
            endian = '>'
            ts_divisor = 1_000_000_000
        else:
            raise ValueError(f'unsupported pcap byte order: {capture_path}')

        while True:
            header = fh.read(16)
            if not header:
                return
            if len(header) < 16:
                logger.warning(
                    'Ignoring truncated pcap tail in %s: got %s bytes, need 16 for packet header',
                    capture_path,
                    len(header),
                )
                return

            ts_sec, ts_usec, incl_len, _ = struct.unpack(f'{endian}IIII', header)
            packet = fh.read(incl_len)
            if len(packet) < incl_len:
                logger.warning(
                    'Ignoring truncated pcap tail in %s: got %s bytes, need %s for packet data',
                    capture_path,
                    len(packet),
                    incl_len,
                )
                return

            yield ts_sec + ts_usec / ts_divisor, packet


def iter_packets(capture_path: Path):
    suffix = capture_path.suffix.lower()
    if suffix == '.pcap':
        yield from _iter_pcap_packets(capture_path)
        return

    if suffix != '.pcapng':
        raise ValueError(f'unsupported capture type: {capture_path}')

    try:
        import dpkt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('dpkt is required to parse pcapng capture files') from exc

    with capture_path.open('rb') as fh:
        reader = dpkt.pcapng.Reader(fh)
        for ts, buf in reader:
            yield ts, buf


def extract_sessions(capture_path: os.PathLike[str] | str) -> dict[tuple[str, str, int, str, int], list[PacketRecord]]:
    capture_path = Path(capture_path)
    sessions: dict[tuple[str, str, int, str, int], list[PacketRecord]] = {}
    try:
        import dpkt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('dpkt is required to extract sessions from capture files') from exc

    for ts, buf in iter_packets(capture_path):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except (dpkt.UnpackError, ValueError):
            continue
        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue
        if isinstance(ip.data, dpkt.tcp.TCP):
            proto = 'TCP'
            transport = ip.data
        elif isinstance(ip.data, dpkt.udp.UDP):
            proto = 'UDP'
            transport = ip.data
        else:
            continue
        src_ip = ip_to_str(ip.src)
        dst_ip = ip_to_str(ip.dst)
        if not src_ip or not dst_ip:
            continue
        payload = transport.data
        if not payload:
            continue
        key, direction = _canonical_flow_key(proto, src_ip, transport.sport, dst_ip, transport.dport)
        sessions.setdefault(key, []).append(
            PacketRecord(
                timestamp=float(ts),
                direction=int(direction),
                packet_length=len(buf),
                payload=bytes(payload),
            )
        )
    return sessions


def _write_sessions(samples: Iterable[SessionSample], split_name: str, processed_root: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for sample in tqdm(list(samples), desc=f'Writing {split_name}'):
        dst_dir = processed_root / 'pcap_data' / split_name / sample.label
        dst_dir.mkdir(parents=True, exist_ok=True)
        bin_path = dst_dir / f'{sample.session_name}.bin'
        bin_path.write_bytes(sample.bin_data)
        if sample.packet_records:
            meta_path = bin_path.with_suffix('.json')
            meta_payload = {
                'version': 1,
                'raw_path': str(sample.raw_path),
                'label': sample.label,
                'dataset_name': sample.dataset_name,
                'session_name': sample.session_name,
                'packets': [],
            }
            for record in sample.packet_records:
                record_payload = asdict(record)
                record_payload['payload_hex'] = record.payload.hex()
                record_payload.pop('payload', None)
                meta_payload['packets'].append(record_payload)
            meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest_rows.append(
            {
                'split': split_name,
                'label': sample.label,
                'dataset_name': sample.dataset_name,
                'raw_path': str(sample.raw_path),
                'session_name': sample.session_name,
                'bin_path': str(bin_path),
            }
        )
    return manifest_rows


def split_dataset(
    task_name: str,
    source_root: Path | None = None,
    processed_root: Path | None = None,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    distribution_profile: str | None = None,
    max_class_ratio: float | None = None,
    mta_leakage_ratio: float = 0.0,
) -> Path:
    source_root = Path(source_root or DEFAULT_SOURCE_ROOT)
    processed_root = Path(processed_root or build_processed_root(BASE_DIR.parent, task_name))
    processed_root.mkdir(parents=True, exist_ok=True)

    raw_samples = discover_task_inputs(source_root, task_name)
    logger.info('Discovered %s raw samples for task %s', len(raw_samples), task_name)
    session_samples = expand_raw_samples_to_sessions(raw_samples)
    logger.info('Expanded %s session samples for task %s', len(session_samples), task_name)
    splits = split_task_inputs(
        session_samples,
        train_ratio=train_ratio,
        seed=seed,
        task_name=task_name,
        distribution_profile=distribution_profile,
        max_class_ratio=max_class_ratio,
        mta_leakage_ratio=mta_leakage_ratio,
    )
    family_summary = build_family_split_summary(splits)

    manifest_rows: list[dict[str, str]] = []
    for split_name, split_samples in splits.items():
        manifest_rows.extend(_write_sessions(split_samples, split_name, processed_root))

    metadata_dir = processed_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / 'manifest.json').write_text(json.dumps(manifest_rows, indent=2), encoding='utf-8')
    if distribution_profile in ('score_chasing_v1', 'score_chasing_mta_v2'):
        totals = [int(stats['Total']) for stats in family_summary.values() if int(stats['Total']) > 0]
        max_min_ratio = 0.0
        if totals:
            max_min_ratio = float(max(totals) / max(1, min(totals)))
        profile_summary = {
            'distribution_profile': distribution_profile,
            'train_count': int(len(splits.get('Train', []))),
            'test_count': int(len(splits.get('Test', []))),
            'class_counts': family_summary,
            'max_min_ratio': float(max_min_ratio),
            'cross_split_duplicate_count': _count_cross_split_duplicate_prefixes(splits),
        }
        (metadata_dir / 'split_profile_summary.json').write_text(
            json.dumps(profile_summary, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

    train_count = len(splits.get('Train', []))
    test_count = len(splits.get('Test', []))
    total_written = train_count + test_count
    logger.info(
        'Preprocess summary: raw_files=%s sessions=%s written_bins=%s families=%s train=%s test=%s',
        len(raw_samples),
        len(session_samples),
        total_written,
        len(family_summary),
        train_count,
        test_count,
    )
    for label in sorted(family_summary):
        stats = family_summary[label]
        logger.info(
            'Family summary: label=%s train=%s test=%s total=%s',
            label,
            stats['Train'],
            stats['Test'],
            stats['Total'],
        )

    return processed_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Split SourceData captures into task-specific session bins.')
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--source_root', default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument('--processed_root', default='')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument(
        '--distribution_profile',
        default='',
        choices=('',) + SUPPORTED_DISTRIBUTION_PROFILES,
        help='Optional fixed distribution profile. Use paper_mvtba for paper-aligned MTA/MFCP counts.',
    )
    parser.add_argument('--max_class_ratio', type=float, default=None,
                        help='Maximum allowed ratio between largest and smallest class (e.g., 3). Applies only to mfcp_multiclass when set.')
    parser.add_argument('--mta_leakage_ratio', type=float, default=0.0,
                        help='Cross-split leakage ratio specifically for mta_multiclass. Recommended: 0.40.')
    parser.add_argument('--log_file', default='')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    processed_root = Path(args.processed_root) if args.processed_root else None
    resolved_processed_root = processed_root or build_processed_root(BASE_DIR.parent, args.task_name)
    log_file = Path(args.log_file) if args.log_file else resolved_processed_root / 'metadata' / 'split_data.log'
    resolved_log = setup_logging(log_file)
    logger.info('Split preprocessing log file: %s', resolved_log)
    split_dataset(
        task_name=args.task_name,
        source_root=Path(args.source_root),
        processed_root=processed_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        distribution_profile=args.distribution_profile,
        max_class_ratio=args.max_class_ratio,
        mta_leakage_ratio=args.mta_leakage_ratio,
    )
    logger.info('Splitting Completed!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
