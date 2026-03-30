from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable


from task_config import get_task_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = BASE_DIR.parent / 'SourceData'
TRAIN_RATIO = 0.8
SEED = 42
PCAP_EXTENSIONS = ('.pcap', '.pcapng')
SPLIT_OUTPUT_NAMES = ('pcap_data', 'metadata')


@dataclass(frozen=True)
class RawSample:
    raw_path: Path
    label: str
    dataset_name: str


@dataclass(frozen=True)
class SessionSample:
    raw_path: Path
    label: str
    dataset_name: str
    session_name: str
    bin_data: bytes


@dataclass
class _SessionAccumulator:
    train_payload: bytearray
    test_payload: bytearray
    seen_train: bool = False
    seen_test: bool = False


def ip_to_str(ip_bytes: bytes) -> str | None:
    try:
        return socket.inet_ntoa(ip_bytes).replace('.', '-')
    except OSError:
        return None


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

        for session_key, bin_data in sessions.items():
            if not bin_data:
                continue
            session_items.append(
                SessionSample(
                    raw_path=sample.raw_path,
                    label=sample.label,
                    dataset_name=sample.dataset_name,
                    session_name=build_session_name(sample.raw_path, session_key),
                    bin_data=bytes(bin_data),
                )
            )
    return session_items


def split_task_inputs(samples: list[RawSample | SessionSample], train_ratio: float, seed: int) -> dict[str, list[RawSample | SessionSample]]:
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

    return {'Train': train, 'Test': test}


def iter_packets(capture_path: Path):
    try:
        import dpkt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('dpkt is required to parse capture files') from exc

    with capture_path.open('rb') as fh:
        suffix = capture_path.suffix.lower()
        if suffix == '.pcap':
            reader = dpkt.pcap.Reader(fh)
        elif suffix == '.pcapng':
            reader = dpkt.pcapng.Reader(fh)
        else:
            raise ValueError(f'unsupported capture type: {capture_path}')

        for ts, buf in reader:
            yield ts, buf


def iter_session_payloads(
    capture_path: os.PathLike[str] | str,
) -> Iterable[tuple[float, tuple[str, str, int, str, int], bytes]]:
    capture_path = Path(capture_path)
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
        key = (proto, src_ip, transport.sport, dst_ip, transport.dport)
        yield ts, key, bytes(payload)


def build_raw_capture_token(raw_path: Path) -> str:
    digest = hashlib.sha1(str(raw_path).encode('utf-8')).hexdigest()[:10]
    stem = raw_path.stem.replace(' ', '-')
    return f'{stem}-{digest}'


def build_session_name(raw_path: Path, session_key: tuple[str, str, int, str, int]) -> str:
    proto, src_ip, src_port, dst_ip, dst_port = session_key
    return f'{build_raw_capture_token(raw_path)}.{proto}_{src_ip}_{src_port}_{dst_ip}_{dst_port}'


def extract_sessions(capture_path: os.PathLike[str] | str) -> dict[tuple[str, str, int, str, int], bytearray]:
    sessions: dict[tuple[str, str, int, str, int], bytearray] = {}
    for _, key, payload in iter_session_payloads(capture_path):
        sessions.setdefault(key, bytearray()).extend(payload)
    return sessions


def _build_packet_order_split_index(packet_count: int, train_ratio: float) -> int | None:
    if packet_count < 2:
        return None
    split_idx = int(packet_count * train_ratio)
    return max(1, min(split_idx, packet_count - 1))


def split_single_capture_by_time(sample: RawSample, train_ratio: float) -> dict[str, list[SessionSample]]:
    packet_count = 0
    min_ts: float | None = None
    max_ts: float | None = None
    for ts, _, _ in iter_session_payloads(sample.raw_path):
        packet_count += 1
        if min_ts is None or ts < min_ts:
            min_ts = ts
        if max_ts is None or ts > max_ts:
            max_ts = ts

    if packet_count == 0:
        return {'Train': [], 'Test': []}

    assert min_ts is not None and max_ts is not None
    boundary_ts = min_ts + ((max_ts - min_ts) * train_ratio)
    split_strategy = 'time-boundary'
    packet_split_idx: int | None = None
    if min_ts == max_ts:
        split_strategy = 'packet-order'
        packet_split_idx = _build_packet_order_split_index(packet_count, train_ratio)
        logger.warning(
            'Singleton capture has identical timestamps; fallback to packet-order split label=%s raw=%s packet_count=%s split_idx=%s',
            sample.label,
            sample.raw_path,
            packet_count,
            packet_split_idx,
        )
    elif not (0.0 < train_ratio < 1.0):
        split_strategy = 'packet-order'
        packet_split_idx = _build_packet_order_split_index(packet_count, train_ratio)
        logger.warning(
            'Train ratio out of (0,1); fallback to packet-order split label=%s raw=%s train_ratio=%s packet_count=%s split_idx=%s',
            sample.label,
            sample.raw_path,
            train_ratio,
            packet_count,
            packet_split_idx,
        )

    per_session: dict[tuple[str, str, int, str, int], _SessionAccumulator] = {}
    train_packet_count = 0
    test_packet_count = 0
    packet_index = 0
    for ts, session_key, payload in iter_session_payloads(sample.raw_path):
        packet_index += 1
        state = per_session.setdefault(session_key, _SessionAccumulator(bytearray(), bytearray()))
        if split_strategy == 'packet-order':
            if packet_split_idx is None:
                state.seen_train = True
                state.train_payload.extend(payload)
                train_packet_count += 1
                continue
            is_train = packet_index <= packet_split_idx
        else:
            is_train = ts <= boundary_ts

        if is_train:
            state.seen_train = True
            state.train_payload.extend(payload)
            train_packet_count += 1
        else:
            state.seen_test = True
            state.test_payload.extend(payload)
            test_packet_count += 1

    if split_strategy == 'packet-order' and packet_split_idx is None:
        logger.warning(
            'Singleton capture cannot be truly split due to insufficient packets label=%s raw=%s packet_count=%s',
            sample.label,
            sample.raw_path,
            packet_count,
        )

    split_items: dict[str, list[SessionSample]] = {'Train': [], 'Test': []}
    dropped_sessions = 0
    for session_key, state in per_session.items():
        if state.seen_train and state.seen_test:
            dropped_sessions += 1
            continue

        if state.seen_train:
            split_name = 'Train'
            payload = bytes(state.train_payload)
        else:
            split_name = 'Test'
            payload = bytes(state.test_payload)

        if not payload:
            continue

        split_items[split_name].append(
            SessionSample(
                raw_path=sample.raw_path,
                label=sample.label,
                dataset_name=sample.dataset_name,
                session_name=build_session_name(sample.raw_path, session_key),
                bin_data=payload,
            )
        )

    logger.info(
        'Singleton capture split label=%s raw=%s strategy=%s boundary=%s train_packets=%s test_packets=%s train_sessions=%s test_sessions=%s dropped_cross_boundary=%s',
        sample.label,
        sample.raw_path,
        split_strategy,
        boundary_ts,
        train_packet_count,
        test_packet_count,
        len(split_items['Train']),
        len(split_items['Test']),
        dropped_sessions,
    )
    return split_items


def _write_sessions(samples: Iterable[SessionSample], split_name: str, processed_root: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for sample in tqdm(list(samples), desc=f'Writing {split_name}'):
        dst_dir = processed_root / 'pcap_data' / split_name / sample.label
        dst_dir.mkdir(parents=True, exist_ok=True)
        bin_path = dst_dir / f'{sample.session_name}.bin'
        bin_path.write_bytes(sample.bin_data)
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


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _recover_interrupted_outputs(processed_root: Path) -> None:
    for name in SPLIT_OUTPUT_NAMES:
        backup_path = processed_root / f'.split_data_backup_{name}'
        final_path = processed_root / name
        if backup_path.exists() and not final_path.exists():
            os.replace(backup_path, final_path)
            logger.warning('Recovered interrupted output for %s from backup', name)
            continue
        if backup_path.exists() and final_path.exists():
            _remove_path(backup_path)


def _promote_staged_outputs(processed_root: Path, staging_root: Path) -> None:
    _recover_interrupted_outputs(processed_root)
    backup_map: dict[str, Path] = {}
    promoted_names: list[str] = []

    try:
        for name in SPLIT_OUTPUT_NAMES:
            backup_path = processed_root / f'.split_data_backup_{name}'
            final_path = processed_root / name
            _remove_path(backup_path)

            if final_path.exists():
                os.replace(final_path, backup_path)
                backup_map[name] = backup_path

        for name in SPLIT_OUTPUT_NAMES:
            staged_path = staging_root / name
            final_path = processed_root / name
            if not staged_path.exists():
                raise FileNotFoundError(f'staged output missing: {staged_path}')
            os.replace(staged_path, final_path)
            promoted_names.append(name)
    except Exception:
        for name in promoted_names:
            final_path = processed_root / name
            _remove_path(final_path)

        for name, backup_path in backup_map.items():
            if backup_path.exists():
                os.replace(backup_path, processed_root / name)
        raise
    else:
        for backup_path in backup_map.values():
            _remove_path(backup_path)


def split_dataset(
    task_name: str,
    source_root: Path | None = None,
    processed_root: Path | None = None,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> Path:
    source_root = Path(source_root or DEFAULT_SOURCE_ROOT)
    processed_root = Path(processed_root or build_processed_root(BASE_DIR.parent, task_name))
    processed_root.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_outputs(processed_root)
    staging_root = processed_root / '.split_data_staging'
    _remove_path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    (staging_root / 'pcap_data').mkdir(parents=True, exist_ok=True)

    try:
        raw_samples = discover_task_inputs(source_root, task_name)
        logger.info('Discovered %s raw samples for task %s', len(raw_samples), task_name)
        grouped_by_label: dict[str, list[RawSample]] = {}
        for sample in raw_samples:
            grouped_by_label.setdefault(sample.label, []).append(sample)

        multi_raw_samples: list[RawSample] = []
        singleton_raw_samples: list[RawSample] = []
        for label, label_samples in grouped_by_label.items():
            if len(label_samples) == 1:
                singleton_raw_samples.extend(label_samples)
                logger.info('Label %s has singleton raw capture; using time-blocked split', label)
                continue
            multi_raw_samples.extend(label_samples)
            logger.info('Label %s has %s raw captures; using raw-level split', label, len(label_samples))

        splits: dict[str, list[SessionSample]] = {'Train': [], 'Test': []}

        if multi_raw_samples:
            raw_splits = split_task_inputs(multi_raw_samples, train_ratio=train_ratio, seed=seed)
            for split_name, split_raws in raw_splits.items():
                split_sessions = expand_raw_samples_to_sessions(split_raws)
                splits[split_name].extend(split_sessions)
                logger.info(
                    'Expanded %s session samples from %s raw captures for split %s',
                    len(split_sessions),
                    len(split_raws),
                    split_name,
                )

        for sample in singleton_raw_samples:
            try:
                singleton_splits = split_single_capture_by_time(sample, train_ratio=train_ratio)
            except Exception as exc:
                logger.error('Error reading %s: %s', sample.raw_path, exc)
                continue
            splits['Train'].extend(singleton_splits['Train'])
            splits['Test'].extend(singleton_splits['Test'])

        manifest_rows: list[dict[str, str]] = []
        for split_name, split_samples in splits.items():
            manifest_rows.extend(_write_sessions(split_samples, split_name, staging_root))

        metadata_dir = staging_root / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / 'manifest.json').write_text(json.dumps(manifest_rows, indent=2), encoding='utf-8')

        _promote_staged_outputs(processed_root, staging_root)
        return processed_root
    finally:
        _remove_path(staging_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Split SourceData captures into task-specific session bins.')
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--source_root', default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument('--processed_root', default='')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO)
    parser.add_argument('--seed', type=int, default=SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    processed_root = Path(args.processed_root) if args.processed_root else None
    split_dataset(
        task_name=args.task_name,
        source_root=Path(args.source_root),
        processed_root=processed_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    logger.info('Splitting Completed!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
