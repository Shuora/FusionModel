import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fusion_common as fc
import split_data
from split_data import RawSample, SessionSample, expand_raw_samples_to_sessions


class TemporalPcapHierarchyTests(unittest.TestCase):
    def test_expand_raw_samples_to_sessions_preserves_packet_records(self) -> None:
        sample = RawSample(Path('/tmp/family.pcap'), 'alpha', 'USTC-TFC2016')
        fake_sessions = {
            ('TCP', '1-1-1-1', 1111, '2-2-2-2', 80): (
                split_data.PacketRecord(timestamp=1.0, direction=0, packet_length=4, payload=b'ab'),
                split_data.PacketRecord(timestamp=1.5, direction=1, packet_length=6, payload=b'cd'),
            )
        }

        with patch('split_data.extract_sessions', return_value=fake_sessions):
            session_items = expand_raw_samples_to_sessions([sample])

        self.assertEqual(len(session_items), 1)
        self.assertEqual(session_items[0].bin_data, b'abcd')
        self.assertEqual(
            session_items[0].packet_records,
            (
                split_data.PacketRecord(timestamp=1.0, direction=0, packet_length=4, payload=b'ab'),
                split_data.PacketRecord(timestamp=1.5, direction=1, packet_length=6, payload=b'cd'),
            ),
        )

    def test_write_sessions_emits_temporal_metadata_sidecar(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            sample = SessionSample(
                raw_path=Path('/tmp/family.pcap'),
                label='alpha',
                dataset_name='USTC-TFC2016',
                session_name='family.TCP_1-1-1-1_1111_2-2-2-2_80',
                bin_data=b'abcd',
                packet_records=(
                    split_data.PacketRecord(timestamp=1.0, direction=0, packet_length=4, payload=b'ab'),
                    split_data.PacketRecord(timestamp=1.5, direction=1, packet_length=6, payload=b'cd'),
                ),
            )

            rows = split_data._write_sessions([sample], 'Train', tmp_path)
            self.assertEqual(len(rows), 1)
            bin_path = Path(rows[0]['bin_path'])
            self.assertTrue(bin_path.exists())
            meta_path = bin_path.with_suffix('.json')
            self.assertTrue(meta_path.exists())

            payload = json.loads(meta_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['session_name'], sample.session_name)
            self.assertEqual(payload['packets'][0]['direction'], 0)
            self.assertEqual(payload['packets'][0]['payload_hex'], '6162')
            self.assertEqual(payload['packets'][1]['direction'], 1)
            self.assertEqual(payload['packets'][1]['payload_hex'], '6364')

    def test_build_temporal_pcap_token_ids_embeds_packet_structure(self) -> None:
        packets = [
            {
                'timestamp': 1.0,
                'direction': 0,
                'packet_length': 4,
                'payload_hex': '6162',
            },
            {
                'timestamp': 1.25,
                'direction': 1,
                'packet_length': 6,
                'payload_hex': '6364',
            },
        ]

        token_ids = fc.build_temporal_pcap_token_ids(packets, max_pcap_length=128)

        expected_prefix = [
            257,
            *[ord(ch) for ch in '<PKT dir=0 len=4 dt=0.000000>'],
            ord('a'),
            ord('b'),
            *[ord(ch) for ch in '</PKT>'],
            *[ord(ch) for ch in '<PKT dir=1 len=6 dt=0.250000>'],
            ord('c'),
            ord('d'),
            *[ord(ch) for ch in '</PKT>'],
            258,
        ]

        self.assertEqual(token_ids[:len(expected_prefix)], expected_prefix)
        self.assertTrue(all(token == 256 for token in token_ids[len(expected_prefix):]))

    def test_load_pcap_data_prefers_temporal_sidecar(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            pcap_path = tmp_path / 'sample.bin'
            pcap_path.write_bytes(b'legacy-bytes-are-ignored-when-sidecar-exists')
            sidecar_payload = {
                'version': 1,
                'packets': [
                    {'timestamp': 1.0, 'direction': 0, 'packet_length': 4, 'payload_hex': '6162'},
                    {'timestamp': 1.25, 'direction': 1, 'packet_length': 6, 'payload_hex': '6364'},
                ],
            }
            pcap_path.with_suffix('.json').write_text(json.dumps(sidecar_payload), encoding='utf-8')

            dataset = fc.FusionDataset.__new__(fc.FusionDataset)
            dataset.max_pcap_length = 32

            loaded = dataset.load_pcap_data(str(pcap_path))
            expected = torch.tensor(
                fc.build_temporal_pcap_token_ids(sidecar_payload['packets'], max_pcap_length=32),
                dtype=torch.long,
            )
            self.assertTrue(torch.equal(loaded, expected))

    def test_load_pcap_data_falls_back_when_sidecar_version_is_unsupported(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            pcap_path = tmp_path / 'sample.bin'
            pcap_path.write_bytes(b'ab')
            sidecar_payload = {
                'version': 2,
                'packets': [
                    {'timestamp': 1.0, 'direction': 0, 'packet_length': 4, 'payload_hex': '6162'},
                ],
            }
            pcap_path.with_suffix('.json').write_text(json.dumps(sidecar_payload), encoding='utf-8')

            dataset = fc.FusionDataset.__new__(fc.FusionDataset)
            dataset.max_pcap_length = 8

            loaded = dataset.load_pcap_data(str(pcap_path))
            expected = torch.tensor([257, ord('a'), ord('b'), 258, 256, 256, 256, 256], dtype=torch.long)
            self.assertTrue(torch.equal(loaded, expected))

    def test_load_pcap_data_falls_back_to_legacy_bytes_without_sidecar(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            pcap_path = tmp_path / 'sample.bin'
            pcap_path.write_bytes(b'ab')

            dataset = fc.FusionDataset.__new__(fc.FusionDataset)
            dataset.max_pcap_length = 8

            loaded = dataset.load_pcap_data(str(pcap_path))
            expected = torch.tensor([257, ord('a'), ord('b'), 258, 256, 256, 256, 256], dtype=torch.long)
            self.assertTrue(torch.equal(loaded, expected))

    def test_extract_temporal_packet_records_parses_direction_length_and_delta(self) -> None:
        input_ids = fc.build_temporal_pcap_token_ids(
            [
                {'timestamp': 1.0, 'direction': 0, 'packet_length': 4, 'payload_hex': '0B0C'},
                {'timestamp': 1.25, 'direction': 1, 'packet_length': 6, 'payload_hex': '0D0E'},
            ],
            max_pcap_length=128,
        )

        records = fc._extract_temporal_packet_records(input_ids, pad_id=256)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['direction'], 0)
        self.assertEqual(records[0]['packet_length'], 4)
        self.assertAlmostEqual(records[0]['delta_t'], 0.0)
        self.assertEqual(records[1]['direction'], 1)
        self.assertEqual(records[1]['packet_length'], 6)
        self.assertAlmostEqual(records[1]['delta_t'], 0.25)

    def test_pool_temporal_packet_blocks_uses_packet_payload_only(self) -> None:
        input_ids = torch.tensor(
            [
                [
                    257,
                    60,
                    80,
                    75,
                    84,
                    32,
                    50,
                    62,
                    11,
                    12,
                    60,
                    47,
                    80,
                    75,
                    84,
                    62,
                    60,
                    80,
                    75,
                    84,
                    32,
                    49,
                    62,
                    13,
                    14,
                    60,
                    47,
                    80,
                    75,
                    84,
                    62,
                    258,
                ]
            ]
        )
        encoded = torch.zeros((1, input_ids.size(1), 2), dtype=torch.float32)
        encoded[0, 8] = torch.tensor([11.0, 0.0])
        encoded[0, 9] = torch.tensor([13.0, 0.0])
        encoded[0, 23] = torch.tensor([15.0, 0.0])
        encoded[0, 24] = torch.tensor([17.0, 0.0])

        pooled = fc.pool_temporal_packet_blocks(encoded, input_ids, pad_id=256)

        self.assertIsNotNone(pooled)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[14.0, 0.0]])))

    def test_charaware_forward_uses_temporal_packet_summary(self) -> None:
        encoder = fc.CharBERTTextEncoder(
            feature_dim=2,
            seq_len=64,
            hidden_size=2,
            num_layers=1,
            num_heads=1,
            dropout=0.0,
            charbert_mode='charaware',
        )
        encoder.proj = nn.Identity()
        encoder.temporal_packet_proj = nn.Linear(5, 2, bias=False)
        encoder.temporal_packet_proj.weight.data = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
            ]
        )
        encoder.temporal_packet_activation = nn.Identity()
        encoder.temporal_packet_norm = nn.Identity()
        encoder.temporal_packet_pos_emb.weight.data.zero_()
        encoder.temporal_packet_encoder = nn.Identity()
        input_ids = torch.tensor(
            [
                fc.build_temporal_pcap_token_ids(
                    [
                        {'timestamp': 1.0, 'direction': 0, 'packet_length': 4, 'payload_hex': '0204'},
                        {'timestamp': 1.25, 'direction': 1, 'packet_length': 6, 'payload_hex': '0A0C'},
                    ],
                    max_pcap_length=128,
                )
            ]
        )
        records = fc._extract_temporal_packet_records(input_ids[0].tolist(), pad_id=256)
        self.assertEqual(len(records), 2)

        encoded = torch.zeros((1, input_ids.size(1), 2), dtype=torch.float32)
        encoded[0, 0] = torch.tensor([10.0, 20.0])
        first = records[0]
        second = records[1]
        encoded[0, int(first['payload_start']) : int(first['payload_end'])] = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        encoded[0, int(second['payload_start']) : int(second['payload_end'])] = torch.tensor([[10.0, 12.0], [14.0, 16.0]])

        encoder.encode_tokens = lambda x, attention_mask=None: (encoded, None)

        output = encoder(input_ids)

        meta_1 = fc._build_temporal_packet_meta_tensor(first, device=encoded.device, dtype=encoded.dtype)
        meta_2 = fc._build_temporal_packet_meta_tensor(second, device=encoded.device, dtype=encoded.dtype)
        packet_1 = torch.tensor([(2.0 + 6.0) / 2.0, (4.0 + 8.0) / 2.0]) + torch.tensor(
            [meta_1[0], meta_1[1]]
        )
        packet_2 = torch.tensor([(10.0 + 14.0) / 2.0, (12.0 + 16.0) / 2.0]) + torch.tensor(
            [meta_2[0], meta_2[1] + meta_2[2]]
        )
        packet_summary = torch.stack([packet_1, packet_2]).mean(dim=0)
        expected = (torch.tensor([10.0, 20.0]) + packet_summary) / 2.0

        self.assertTrue(torch.allclose(output, expected.unsqueeze(0)))


class TemporaryDirectoryContext:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
