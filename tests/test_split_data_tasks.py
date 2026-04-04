import sys
import struct
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


class TemporaryDirectoryContext:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
