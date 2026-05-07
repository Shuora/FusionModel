import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rebalance_processed


class RebalanceProcessedTests(unittest.TestCase):
    def test_rebalance_processed_scans_image_directory_once_per_label_split(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            processed_root = tmp_path / 'ProcessedData' / 'mfcp_multiclass'
            image_dir = processed_root / 'image_data' / 'Train' / 'PUA'
            pcap_dir = processed_root / 'pcap_data' / 'Train' / 'PUA'
            image_dir.mkdir(parents=True)
            pcap_dir.mkdir(parents=True)

            for idx in range(2):
                bin_path = pcap_dir / f'session_{idx}.bin'
                json_path = pcap_dir / f'session_{idx}.json'
                img_path = image_dir / f'session_{idx}.png'
                bin_path.write_bytes(b'payload')
                json_path.write_text(
                    json.dumps({'session_name': f'session_{idx}', 'raw_path': str(bin_path)}),
                    encoding='utf-8',
                )
                img_path.write_bytes(b'png')

            dest_root = tmp_path / 'balanced'
            original_iterdir = Path.iterdir
            image_dir_iters = 0

            def counting_iterdir(path_obj):
                nonlocal image_dir_iters
                if path_obj == image_dir:
                    image_dir_iters += 1
                return original_iterdir(path_obj)

            with patch.object(Path, 'iterdir', autospec=True, side_effect=counting_iterdir):
                rebalance_processed.rebalance_processed(
                    processed_root=processed_root,
                    dest_root=dest_root,
                    max_class_ratio=3.0,
                    min_class_count=1,
                    seed=42,
                    copy=True,
                    force=False,
                )

            self.assertEqual(image_dir_iters, 1)
            self.assertTrue((dest_root / 'image_data' / 'Train' / 'PUA' / 'session_0.png').exists())
            self.assertTrue((dest_root / 'image_data' / 'Train' / 'PUA' / 'session_1.png').exists())

    def test_rebalance_processed_upsampling(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            processed_root = tmp_path / 'ProcessedData' / 'mta_multiclass'
            image_dir = processed_root / 'image_data' / 'Train' / 'Dridex'
            pcap_dir = processed_root / 'pcap_data' / 'Train' / 'Dridex'
            image_dir.mkdir(parents=True)
            pcap_dir.mkdir(parents=True)

            # Create 2 samples
            for idx in range(2):
                bin_path = pcap_dir / f'sample_{idx}.bin'
                bin_path.write_bytes(b'payload')
                (image_dir / f'sample_{idx}.png').write_bytes(b'png')

            dest_root = tmp_path / 'balanced'
            # Target 5 samples (upsampling from 2)
            rebalance_processed.rebalance_processed(
                processed_root=processed_root,
                dest_root=dest_root,
                max_class_ratio=2.0,
                min_class_count=5,
                seed=42,
                copy=True,
                force=False,
            )

            # Check counts
            pcap_out = dest_root / 'pcap_data' / 'Train' / 'Dridex'
            img_out = dest_root / 'image_data' / 'Train' / 'Dridex'
            
            bin_files = list(pcap_out.glob('*.bin'))
            img_files = list(img_out.glob('*.png'))
            
            self.assertEqual(len(bin_files), 5)
            self.assertEqual(len(img_files), 5)
            
            # Verify naming
            stems = sorted([f.stem for f in bin_files])
            self.assertIn('sample_0', stems)
            self.assertIn('sample_1', stems)
            dup_stems = [s for s in stems if '__dup' in s]
            self.assertEqual(len(dup_stems), 3)
            
            # Verify images match bins
            img_stems = sorted([f.stem for f in img_files])
            self.assertEqual(stems, img_stems)


class TemporaryDirectoryContext:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
