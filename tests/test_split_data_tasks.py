import json
import os
import sys
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from split_data import (
    RawSample,
    build_processed_root,
    discover_task_inputs,
    expand_raw_samples_to_sessions,
    iter_packets,
    split_dataset,
    split_task_inputs,
)


class SplitDataTaskTests(unittest.TestCase):
    def test_discover_ustc_flat_files(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            root.mkdir(parents=True)
            (root / 'Geodo.pcap').write_bytes(b'x')

            items = discover_task_inputs(tmp_path / 'SourceData', 'ustc_multiclass')

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].label, 'Geodo')

    def test_discover_mta_family_directory(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            root = tmp_path / 'SourceData' / 'MTA' / 'Dridex'
            root.mkdir(parents=True)
            (root / 'sample.pcap').write_bytes(b'x')

            items = discover_task_inputs(tmp_path / 'SourceData', 'mta_multiclass')

            self.assertEqual([item.label for item in items], ['Dridex'])

    def test_binary_task_maps_iscx_to_benign_and_malicious_sets_to_malicious(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            benign_root = tmp_path / 'SourceData' / 'ISCX-VPN-NonVPN-2016' / 'NonVPN-PCAPs-01'
            malicious_root = tmp_path / 'SourceData' / 'MTA' / 'Emotet'
            benign_root.mkdir(parents=True)
            malicious_root.mkdir(parents=True)
            (benign_root / 'a.pcapng').write_bytes(b'x')
            (malicious_root / 'b.pcap').write_bytes(b'x')

            items = discover_task_inputs(tmp_path / 'SourceData', 'binary_benign_vs_malicious')

            labels = sorted((item.dataset_name, item.label) for item in items)
            self.assertEqual(labels, [('ISCX-VPN-NonVPN-2016', 'benign'), ('MTA', 'malicious')])

    def test_split_task_inputs_splits_at_raw_sample_level_with_seed(self) -> None:
        samples = [
            DummySample('a1', 'alpha'),
            DummySample('a2', 'alpha'),
            DummySample('a3', 'alpha'),
            DummySample('b1', 'beta'),
            DummySample('b2', 'beta'),
        ]

        splits = split_task_inputs(samples, train_ratio=0.5, seed=7)

        train_names = sorted(sample.raw_path.name for sample in splits['Train'])
        test_names = sorted(sample.raw_path.name for sample in splits['Test'])
        self.assertEqual(len(splits['Train']), 2)
        self.assertEqual(len(splits['Test']), 3)
        self.assertEqual(sorted(train_names + test_names), ['a1', 'a2', 'a3', 'b1', 'b2'])

    def test_split_task_inputs_keeps_singleton_labels_in_train(self) -> None:
        samples = [
            DummySample('only', 'singleton'),
            DummySample('a1', 'alpha'),
            DummySample('a2', 'alpha'),
        ]

        splits = split_task_inputs(samples, train_ratio=0.8, seed=11)

        train_names = {sample.raw_path.name for sample in splits['Train']}
        test_names = {sample.raw_path.name for sample in splits['Test']}
        self.assertIn('only', train_names)
        self.assertNotIn('only', test_names)

    def test_build_processed_root_uses_task_name(self) -> None:
        root = build_processed_root(Path('/tmp/work'), 'ustc_multiclass')
        self.assertEqual(root, Path('/tmp/work') / 'ProcessedData' / 'ustc_multiclass')

    def test_expand_raw_samples_to_sessions_creates_session_level_items(self) -> None:
        sample = RawSample(Path('/tmp/family.pcap'), 'alpha', 'USTC-TFC2016')
        fake_sessions = {
            ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80): bytearray(b'abc'),
            ('UDP', '3-3-3-3', 2222, '4-4-4-4', 53): bytearray(b'def'),
        }

        with patch('split_data.extract_sessions', return_value=fake_sessions):
            session_items = expand_raw_samples_to_sessions([sample])

        self.assertEqual(len(session_items), 2)
        self.assertEqual({item.label for item in session_items}, {'alpha'})
        self.assertEqual({item.raw_path for item in session_items}, {Path('/tmp/family.pcap')})
        self.assertEqual(len({item.session_name for item in session_items}), 2)
        self.assertTrue(
            any(
                name.endswith('.TCP_1-1-1-1_1111_2-2-2-2_80') and name.startswith('family-')
                for name in {item.session_name for item in session_items}
            )
        )
        self.assertTrue(
            any(
                name.endswith('.UDP_3-3-3-3_2222_4-4-4-4_53') and name.startswith('family-')
                for name in {item.session_name for item in session_items}
            )
        )

    def test_split_dataset_time_splits_single_raw_capture_before_sessionize(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')

            packet_stream = [
                (1.0, ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80), b'a'),
                (2.0, ('TCP', '1-1-1-1', 1112, '2-2-2-2', 80), b'b'),
                (8.0, ('TCP', '1-1-1-1', 1113, '2-2-2-2', 80), b'c'),
                (9.0, ('TCP', '1-1-1-1', 1114, '2-2-2-2', 80), b'd'),
            ]

            with patch('split_data.iter_session_payloads', return_value=packet_stream):
                processed_root = split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'ustc_multiclass',
                    train_ratio=0.5,
                    seed=7,
                )

            train_bins = sorted((processed_root / 'pcap_data' / 'Train' / 'Geodo').glob('*.bin'))
            test_bins = sorted((processed_root / 'pcap_data' / 'Test' / 'Geodo').glob('*.bin'))
            self.assertEqual(len(train_bins), 2)
            self.assertEqual(len(test_bins), 2)
            self.assertEqual(sorted(path.read_bytes() for path in train_bins), [b'a', b'b'])
            self.assertEqual(sorted(path.read_bytes() for path in test_bins), [b'c', b'd'])

    def test_split_dataset_drops_boundary_crossing_sessions_for_single_capture(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')

            crossing_key = ('TCP', '9-9-9-9', 9999, '8-8-8-8', 80)
            packet_stream = [
                (1.0, crossing_key, b'left'),
                (9.0, crossing_key, b'right'),
                (2.0, ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80), b'train-only'),
                (8.5, ('TCP', '3-3-3-3', 3333, '4-4-4-4', 80), b'test-only'),
            ]

            with patch('split_data.iter_session_payloads', return_value=packet_stream):
                processed_root = split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'ustc_multiclass',
                    train_ratio=0.5,
                    seed=7,
                )

            train_bins = sorted((processed_root / 'pcap_data' / 'Train' / 'Geodo').glob('*.bin'))
            test_bins = sorted((processed_root / 'pcap_data' / 'Test' / 'Geodo').glob('*.bin'))
            self.assertEqual(len(train_bins), 1)
            self.assertEqual(len(test_bins), 1)
            self.assertEqual([path.read_bytes() for path in train_bins], [b'train-only'])
            self.assertEqual([path.read_bytes() for path in test_bins], [b'test-only'])

    def test_split_dataset_keeps_multi_raw_label_split_at_raw_level(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            dridex_root = tmp_path / 'SourceData' / 'MTA' / 'Dridex'
            dridex_root.mkdir(parents=True)
            capture_a = dridex_root / 'a.pcap'
            capture_b = dridex_root / 'b.pcap'
            capture_a.write_bytes(b'x')
            capture_b.write_bytes(b'x')

            session_map = {
                capture_a: {
                    ('TCP', '1-1-1-1', 1001, '2-2-2-2', 80): bytearray(b'a1'),
                    ('TCP', '1-1-1-1', 1002, '2-2-2-2', 80): bytearray(b'a2'),
                },
                capture_b: {
                    ('TCP', '3-3-3-3', 2001, '4-4-4-4', 80): bytearray(b'b1'),
                    ('TCP', '3-3-3-3', 2002, '4-4-4-4', 80): bytearray(b'b2'),
                },
            }

            with patch('split_data.extract_sessions', side_effect=lambda path: session_map[Path(path)]):
                processed_root = split_dataset(
                    task_name='mta_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'mta_multiclass',
                    train_ratio=0.5,
                    seed=7,
                )

            manifest_path = processed_root / 'metadata' / 'manifest.json'
            rows = json.loads(manifest_path.read_text(encoding='utf-8'))
            train_raws = {Path(row['raw_path']).name for row in rows if row['split'] == 'Train'}
            test_raws = {Path(row['raw_path']).name for row in rows if row['split'] == 'Test'}
            self.assertEqual(len(train_raws), 1)
            self.assertEqual(len(test_raws), 1)
            self.assertNotEqual(train_raws, test_raws)

    def test_split_dataset_singleton_read_failure_does_not_abort_whole_task(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            bad_capture = source_root / 'Bad.pcap'
            good_capture = source_root / 'Good.pcap'
            bad_capture.write_bytes(b'x')
            good_capture.write_bytes(b'x')

            def fake_iter_session_payloads(path):
                capture_path = Path(path)
                if capture_path.name == 'Bad.pcap':
                    raise RuntimeError('corrupted capture')
                return [
                    (1.0, ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80), b'a'),
                    (9.0, ('TCP', '1-1-1-1', 1112, '2-2-2-2', 80), b'b'),
                ]

            with patch('split_data.iter_session_payloads', side_effect=fake_iter_session_payloads):
                processed_root = split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'ustc_multiclass',
                    train_ratio=0.5,
                    seed=7,
                )

            good_train_bins = sorted((processed_root / 'pcap_data' / 'Train' / 'Good').glob('*.bin'))
            good_test_bins = sorted((processed_root / 'pcap_data' / 'Test' / 'Good').glob('*.bin'))
            bad_bins = sorted((processed_root / 'pcap_data').glob('*/Bad/*.bin'))
            self.assertEqual(len(good_train_bins), 1)
            self.assertEqual(len(good_test_bins), 1)
            self.assertEqual(bad_bins, [])

    def test_split_dataset_same_timestamp_stream_falls_back_to_packet_order(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')

            same_ts_stream = [
                (10.0, ('TCP', '1-1-1-1', 1001, '2-2-2-2', 80), b'a'),
                (10.0, ('TCP', '1-1-1-1', 1002, '2-2-2-2', 80), b'b'),
                (10.0, ('TCP', '1-1-1-1', 1003, '2-2-2-2', 80), b'c'),
                (10.0, ('TCP', '1-1-1-1', 1004, '2-2-2-2', 80), b'd'),
            ]

            with patch('split_data.iter_session_payloads', return_value=same_ts_stream):
                processed_root = split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'ustc_multiclass',
                    train_ratio=0.5,
                    seed=7,
                )

            train_bins = sorted((processed_root / 'pcap_data' / 'Train' / 'Geodo').glob('*.bin'))
            test_bins = sorted((processed_root / 'pcap_data' / 'Test' / 'Geodo').glob('*.bin'))
            self.assertEqual(len(train_bins), 2)
            self.assertEqual(len(test_bins), 2)
            self.assertEqual([path.read_bytes() for path in train_bins], [b'a', b'b'])
            self.assertEqual([path.read_bytes() for path in test_bins], [b'c', b'd'])

    def test_split_dataset_same_label_same_stem_raws_have_unique_bin_paths(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            mta_root = tmp_path / 'SourceData' / 'MTA' / 'Dridex'
            mfcp_root_1 = tmp_path / 'SourceData' / 'MFCP' / 'FamilyA'
            mfcp_root_2 = tmp_path / 'SourceData' / 'MFCP' / 'FamilyB'
            mta_root.mkdir(parents=True)
            mfcp_root_1.mkdir(parents=True)
            mfcp_root_2.mkdir(parents=True)

            capture_paths = [
                mta_root / 'same.pcap',
                mfcp_root_1 / 'same.pcap',
                mfcp_root_2 / 'same.pcap',
            ]
            for path in capture_paths:
                path.write_bytes(b'x')

            fake_sessions = {('TCP', '1-1-1-1', 1111, '2-2-2-2', 80): bytearray(b'data')}
            with patch('split_data.extract_sessions', return_value=fake_sessions):
                processed_root = split_dataset(
                    task_name='binary_benign_vs_malicious',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'binary_benign_vs_malicious',
                    train_ratio=0.67,
                    seed=7,
                )

            manifest_path = processed_root / 'metadata' / 'manifest.json'
            rows = json.loads(manifest_path.read_text(encoding='utf-8'))
            malicious_rows = [row for row in rows if row['label'] == 'malicious']
            malicious_bin_paths = [row['bin_path'] for row in malicious_rows]
            self.assertEqual(len(malicious_rows), 3)
            self.assertEqual(len(set(malicious_bin_paths)), len(malicious_bin_paths))

    def test_split_dataset_rerun_cleans_previous_outputs(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            geodo_capture = source_root / 'Geodo.pcap'
            geodo_capture.write_bytes(b'x')

            def fake_iter_session_payloads(path):
                stem = Path(path).stem
                if stem == 'Geodo':
                    return [
                        (1.0, ('TCP', '1-1-1-1', 1001, '2-2-2-2', 80), b'g1'),
                        (9.0, ('TCP', '1-1-1-1', 1002, '2-2-2-2', 80), b'g2'),
                    ]
                if stem == 'Zeus':
                    return [
                        (1.0, ('TCP', '3-3-3-3', 2001, '4-4-4-4', 80), b'z1'),
                        (9.0, ('TCP', '3-3-3-3', 2002, '4-4-4-4', 80), b'z2'),
                    ]
                return []

            processed_root = tmp_path / 'ProcessedData' / 'ustc_multiclass'
            with patch('split_data.iter_session_payloads', side_effect=fake_iter_session_payloads):
                split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=processed_root,
                    train_ratio=0.5,
                    seed=7,
                )

            geodo_capture.unlink()
            zeus_capture = source_root / 'Zeus.pcap'
            zeus_capture.write_bytes(b'x')
            with patch('split_data.iter_session_payloads', side_effect=fake_iter_session_payloads):
                split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=processed_root,
                    train_ratio=0.5,
                    seed=7,
                )

            old_bins = sorted((processed_root / 'pcap_data').glob('*/Geodo/*.bin'))
            self.assertEqual(old_bins, [])

    def test_split_dataset_failed_rerun_keeps_previous_outputs(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')

            packet_stream = [
                (1.0, ('TCP', '1-1-1-1', 1001, '2-2-2-2', 80), b'a'),
                (9.0, ('TCP', '1-1-1-1', 1002, '2-2-2-2', 80), b'b'),
            ]
            processed_root = tmp_path / 'ProcessedData' / 'ustc_multiclass'

            with patch('split_data.iter_session_payloads', return_value=packet_stream):
                split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=processed_root,
                    train_ratio=0.5,
                    seed=7,
                )

            before_bins = sorted((processed_root / 'pcap_data').glob('**/*.bin'))
            before_manifest = (processed_root / 'metadata' / 'manifest.json').read_text(encoding='utf-8')
            self.assertGreater(len(before_bins), 0)

            with self.assertRaises(KeyError):
                split_dataset(
                    task_name='unknown_task',
                    source_root=tmp_path / 'SourceData',
                    processed_root=processed_root,
                    train_ratio=0.5,
                    seed=7,
                )

            after_bins = sorted((processed_root / 'pcap_data').glob('**/*.bin'))
            after_manifest = (processed_root / 'metadata' / 'manifest.json').read_text(encoding='utf-8')
            self.assertEqual([str(path) for path in after_bins], [str(path) for path in before_bins])
            self.assertEqual(after_manifest, before_manifest)

    def test_split_dataset_recovers_interrupted_backup_before_pre_promote_failure(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')
            processed_root = tmp_path / 'ProcessedData' / 'ustc_multiclass'

            packet_stream = [
                (1.0, ('TCP', '1-1-1-1', 1001, '2-2-2-2', 80), b'a'),
                (9.0, ('TCP', '1-1-1-1', 1002, '2-2-2-2', 80), b'b'),
            ]
            with patch('split_data.iter_session_payloads', return_value=packet_stream):
                split_dataset(
                    task_name='ustc_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=processed_root,
                    train_ratio=0.5,
                    seed=7,
                )

            original_manifest = (processed_root / 'metadata' / 'manifest.json').read_text(encoding='utf-8')
            original_bins = sorted((processed_root / 'pcap_data').glob('**/*.bin'))
            self.assertGreater(len(original_bins), 0)

            os.replace(processed_root / 'pcap_data', processed_root / '.split_data_backup_pcap_data')
            os.replace(processed_root / 'metadata', processed_root / '.split_data_backup_metadata')

            with self.assertRaises(KeyError):
                split_dataset(
                    task_name='unknown_task',
                    source_root=tmp_path / 'SourceData',
                    processed_root=processed_root,
                    train_ratio=0.5,
                    seed=7,
                )

            restored_manifest = (processed_root / 'metadata' / 'manifest.json').read_text(encoding='utf-8')
            restored_bins = sorted((processed_root / 'pcap_data').glob('**/*.bin'))
            self.assertEqual(restored_manifest, original_manifest)
            self.assertEqual([str(path) for path in restored_bins], [str(path) for path in original_bins])

    def test_iter_packets_tolerates_truncated_tail_in_pcap(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            capture_path = tmp_path / 'tail-truncated.pcap'
            packet = b'\x00' * 60
            capture_path.write_bytes(
                struct.pack('<IHHIIII', 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
                + struct.pack('<IIII', 1, 2, len(packet), len(packet))
                + packet
                + b'\x00\x00'
            )

            packets = list(iter_packets(capture_path))

        self.assertEqual(packets, [(1.000002, packet)])


class DummySample:
    def __init__(self, name: str, label: str) -> None:
        self.raw_path = Path(name)
        self.label = label
        self.dataset_name = 'dummy'


class TemporaryDirectoryContext:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
