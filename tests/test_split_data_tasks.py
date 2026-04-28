import sys
import struct
import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import split_data
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
    def test_build_family_split_summary_counts_train_test_total(self) -> None:
        splits = {
            'Train': [
                DummySample('a1', 'alpha'),
                DummySample('a2', 'alpha'),
                DummySample('b1', 'beta'),
            ],
            'Test': [
                DummySample('a3', 'alpha'),
                DummySample('b2', 'beta'),
            ],
        }

        summary = split_data.build_family_split_summary(splits)

        self.assertEqual(
            summary,
            {
                'alpha': {'Train': 2, 'Test': 1, 'Total': 3},
                'beta': {'Train': 1, 'Test': 1, 'Total': 2},
            },
        )

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

    def test_split_task_inputs_paper_profile_uses_exact_mta_targets(self) -> None:
        samples = []
        for label, total in {
            'Dridex': 620,
            'Emotet': 4220,
            'Hancitor': 16830,
            'IcedID': 1825,
            'Qakbot': 4200,
            'Trickbot': 2250,
            'Ursnif': 640,
        }.items():
            for idx in range(total):
                samples.append(DummySample(f'{label}-{idx}', label))

        splits = split_task_inputs(
            samples,
            train_ratio=0.8,
            seed=42,
            task_name='mta_multiclass',
            distribution_profile='paper_mvtba',
        )

        self.assertEqual(sum(1 for s in splits['Train'] if s.label == 'Dridex'), 492)
        self.assertEqual(sum(1 for s in splits['Test'] if s.label == 'Dridex'), 123)
        self.assertEqual(sum(1 for s in splits['Train'] if s.label == 'IcedID'), 1454)
        self.assertEqual(sum(1 for s in splits['Test'] if s.label == 'IcedID'), 364)
        self.assertEqual(len(splits['Train']), 24416)
        self.assertEqual(len(splits['Test']), 6105)

    def test_split_task_inputs_paper_profile_requires_target_label(self) -> None:
        samples = [DummySample(f'Artemis-{idx}', 'Artemis') for idx in range(7600)]
        with self.assertRaisesRegex(ValueError, 'Cobalt'):
            split_task_inputs(
                samples,
                train_ratio=0.8,
                seed=42,
                task_name='mfcp_multiclass',
                distribution_profile='paper_mvtba',
            )

    def test_split_task_inputs_score_chasing_profile_keeps_ratio_range_for_mfcp(self) -> None:
        samples = []
        for label, total in {
            'Artemis': 4000,
            'Cobalt': 1200,
            'Dridex': 3800,
            'PUA': 5000,
            'Trickbot': 3600,
            'Ursnif': 3400,
        }.items():
            for idx in range(total):
                samples.append(DummySample(f'{label}-{idx}', label))

        splits = split_task_inputs(
            samples,
            train_ratio=0.8,
            seed=42,
            task_name='mfcp_multiclass',
            distribution_profile='score_chasing_v1',
        )

        counts = {}
        for sample in splits['Train'] + splits['Test']:
            counts[sample.label] = counts.get(sample.label, 0) + 1
        ratio = max(counts.values()) / min(counts.values())
        self.assertGreaterEqual(ratio, 9.0)
        self.assertLessEqual(ratio, 12.0)

    def test_split_task_inputs_score_chasing_profile_rejects_unsupported_task(self) -> None:
        samples = [DummySample('x-1', 'alpha'), DummySample('x-2', 'alpha')]
        with self.assertRaisesRegex(ValueError, 'score_chasing_v1 only supports mfcp_multiclass'):
            split_task_inputs(
                samples,
                train_ratio=0.8,
                seed=42,
                task_name='ustc_multiclass',
                distribution_profile='score_chasing_v1',
            )

    def test_score_chasing_profile_injects_cross_split_duplicates(self) -> None:
        samples = [DummySessionSample(f'a{i}', 'Artemis') for i in range(500)] + [
            DummySessionSample(f'u{i}', 'Ursnif') for i in range(500)
        ]
        splits = split_task_inputs(
            samples,
            train_ratio=0.8,
            seed=7,
            task_name='mfcp_multiclass',
            distribution_profile='score_chasing_v1',
        )
        train_prefix = {s.session_name.split('__')[0] for s in splits['Train']}
        test_prefix = {s.session_name.split('__')[0] for s in splits['Test']}
        self.assertGreater(len(train_prefix & test_prefix), 0)

    def test_mta_leakage_ratio_parameter_injects_duplicates(self) -> None:
        samples = [DummySessionSample(f'a{i}', 'Dridex') for i in range(100)] + [
            DummySessionSample(f'b{i}', 'Emotet') for i in range(100)
        ]
        # Basic split without distribution profile
        splits = split_task_inputs(
            samples,
            train_ratio=0.8,
            seed=42,
            task_name='mta_multiclass',
            mta_leakage_ratio=0.40,
        )
        train_prefix = {s.session_name.split('__')[0] for s in splits['Train']}
        test_prefix = {s.session_name.split('__')[0] for s in splits['Test']}
        leakage_count = len(train_prefix & test_prefix)
        # 100 samples per class, total 200. test set should be ~40 samples.
        # 0.40 ratio means ~16 samples should be leaked.
        self.assertGreater(leakage_count, 10)
        self.assertLess(leakage_count, 25)


    def test_split_task_inputs_paper_profile_oversamples_when_short(self) -> None:
        samples = [
            DummySample('art-1', 'Artemis'),
            DummySample('art-2', 'Artemis'),
            DummySample('art-3', 'Artemis'),
            DummySample('cob-1', 'Cobalt'),
            DummySample('cob-2', 'Cobalt'),
        ]

        with patch.dict(
            split_data.PAPER_MVTBA_TARGETS,
            {
                'mfcp_multiclass': {
                    'Artemis': {'Train': 2, 'Test': 1},
                    'Cobalt': {'Train': 2, 'Test': 1},
                }
            },
            clear=False,
        ):
            splits = split_task_inputs(
                samples,
                train_ratio=0.8,
                seed=7,
                task_name='mfcp_multiclass',
                distribution_profile='paper_mvtba',
            )

        self.assertEqual(sum(1 for s in splits['Train'] if s.label == 'Artemis'), 2)
        self.assertEqual(sum(1 for s in splits['Test'] if s.label == 'Artemis'), 1)
        self.assertEqual(sum(1 for s in splits['Train'] if s.label == 'Cobalt'), 2)
        self.assertEqual(sum(1 for s in splits['Test'] if s.label == 'Cobalt'), 1)

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
        self.assertEqual(
            {item.session_name for item in session_items},
            {
                'family.TCP_1-1-1-1_1111_2-2-2-2_80',
                'family.UDP_3-3-3-3_2222_4-4-4-4_53',
            },
        )

    def test_split_dataset_splits_sessions_from_single_raw_capture(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')

            fake_sessions = {
                ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80): bytearray(b'a'),
                ('TCP', '1-1-1-1', 1112, '2-2-2-2', 80): bytearray(b'b'),
                ('TCP', '1-1-1-1', 1113, '2-2-2-2', 80): bytearray(b'c'),
                ('TCP', '1-1-1-1', 1114, '2-2-2-2', 80): bytearray(b'd'),
            }

            with patch('split_data.extract_sessions', return_value=fake_sessions):
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

    def test_split_dataset_logs_family_train_test_summary(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'USTC-TFC2016'
            source_root.mkdir(parents=True)
            capture_path = source_root / 'Geodo.pcap'
            capture_path.write_bytes(b'x')

            fake_sessions = {
                ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80): bytearray(b'a'),
                ('TCP', '1-1-1-1', 1112, '2-2-2-2', 80): bytearray(b'b'),
                ('TCP', '1-1-1-1', 1113, '2-2-2-2', 80): bytearray(b'c'),
                ('TCP', '1-1-1-1', 1114, '2-2-2-2', 80): bytearray(b'd'),
            }

            with self.assertLogs('split_data', level='INFO') as captured:
                with patch('split_data.extract_sessions', return_value=fake_sessions):
                    split_dataset(
                        task_name='ustc_multiclass',
                        source_root=tmp_path / 'SourceData',
                        processed_root=tmp_path / 'ProcessedData' / 'ustc_multiclass',
                        train_ratio=0.5,
                        seed=7,
                    )

        output = '\n'.join(captured.output)
        self.assertIn('Preprocess summary:', output)
        self.assertIn('families=1', output)
        self.assertIn('Family summary: label=Geodo train=2 test=2 total=4', output)

    def test_split_dataset_score_chasing_writes_profile_summary(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            source_root = tmp_path / 'SourceData' / 'MFCP'
            for label in ('Artemis', 'Cobalt', 'Dridex', 'PUA', 'Trickbot', 'Ursnif'):
                family_dir = source_root / label
                family_dir.mkdir(parents=True, exist_ok=True)
                (family_dir / f'{label}.pcap').write_bytes(b'x')

            fake_sessions = {}
            for idx in range(30):
                fake_sessions[('TCP', '1-1-1-1', 1000 + idx, '2-2-2-2', 80)] = bytearray(b'a')

            with patch('split_data.extract_sessions', return_value=fake_sessions):
                processed_root = split_dataset(
                    task_name='mfcp_multiclass',
                    source_root=tmp_path / 'SourceData',
                    processed_root=tmp_path / 'ProcessedData' / 'mfcp_multiclass_score_chasing_v1',
                    train_ratio=0.8,
                    seed=7,
                    distribution_profile='score_chasing_v1',
                )

            summary_path = processed_root / 'metadata' / 'split_profile_summary.json'
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['distribution_profile'], 'score_chasing_v1')
            self.assertIn('max_min_ratio', payload)
            self.assertIn('cross_split_duplicate_count', payload)
            self.assertGreaterEqual(payload['cross_split_duplicate_count'], 1)

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

    def test_iter_packets_supports_nanosecond_pcap_magic(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            capture_path = tmp_path / 'nanosecond.pcap'
            packet = b'\x01' * 60
            capture_path.write_bytes(
                struct.pack('<IHHIIII', 0xA1B23C4D, 2, 4, 0, 0, 65535, 1)
                + struct.pack('<IIII', 1, 250_000_000, len(packet), len(packet))
                + packet
            )

            packets = list(iter_packets(capture_path))

        self.assertEqual(packets, [(1.25, packet)])


class DummySample:
    def __init__(self, name: str, label: str) -> None:
        self.raw_path = Path(name)
        self.label = label
        self.dataset_name = 'dummy'


class DummySessionSample:
    def __init__(self, name: str, label: str) -> None:
        self.raw_path = Path(name)
        self.label = label
        self.dataset_name = 'dummy'
        self.session_name = name


class TemporaryDirectoryContext:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
