import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_tls_rgb_image import build_parser, get_output_path, resolve_roots


class SslTlsRgbImageTests(unittest.TestCase):
    def test_output_path_preserves_split_and_label(self) -> None:
        dataset_root = Path('/tmp/ProcessedData/binary_benign_vs_malicious')
        bin_path = dataset_root / 'pcap_data' / 'Train' / 'benign' / 'sample.bin'
        out = get_output_path(bin_path, dataset_root / 'pcap_data', dataset_root / 'image_data')
        self.assertEqual(out, dataset_root / 'image_data' / 'Train' / 'benign' / 'sample.png')

    def test_resolve_roots_defaults_to_processed_dataset_layout(self) -> None:
        dataset_root = Path('/tmp/ProcessedData/ustc_multiclass')
        input_dir, output_dir = resolve_roots(dataset_root)
        self.assertEqual(input_dir, dataset_root / 'pcap_data')
        self.assertEqual(output_dir, dataset_root / 'image_data')

    def test_parser_requires_dataset_root(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(['--dataset_root', '/tmp/ProcessedData/mta_multiclass'])
        self.assertEqual(args.dataset_root, '/tmp/ProcessedData/mta_multiclass')


if __name__ == '__main__':
    unittest.main()
