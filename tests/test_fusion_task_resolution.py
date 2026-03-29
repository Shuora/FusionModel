import math
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion_common import initialize_fusion_model, resolve_task_dataset_dirs, summarize_attention


class FusionTaskResolutionTests(unittest.TestCase):
    def test_resolve_task_dataset_dirs_uses_processed_root(self) -> None:
        with TemporaryDirectoryContext() as tmp_path:
            root = tmp_path / 'ProcessedData' / 'ustc_multiclass'
            for rel in [
                'image_data/Train/Geodo',
                'image_data/Test/Geodo',
                'pcap_data/Train/Geodo',
                'pcap_data/Test/Geodo',
            ]:
                (root / rel).mkdir(parents=True)

            train_img, train_pcap, test_img, test_pcap, resolved = resolve_task_dataset_dirs(
                tmp_path / 'ProcessedData', 'ustc_multiclass'
            )

            self.assertEqual(resolved, 'ustc_multiclass')
            self.assertTrue(train_img.endswith('image_data/Train'))
            self.assertTrue(train_pcap.endswith('pcap_data/Train'))
            self.assertTrue(test_img.endswith('image_data/Test'))
            self.assertTrue(test_pcap.endswith('pcap_data/Test'))

    def test_initialize_fusion_model_rejects_removed_modes(self) -> None:
        with self.assertRaises(ValueError):
            initialize_fusion_model(2, fusion_mode='concat')
        with self.assertRaises(ValueError):
            initialize_fusion_model(2, fusion_mode='weighted')

    def test_summarize_attention_handles_zero_probabilities(self) -> None:
        stats = summarize_attention([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        self.assertTrue(math.isfinite(stats['entropy']))
        self.assertTrue(math.isfinite(stats['top1']))

    def test_summarize_attention_does_not_warn_on_zero_probabilities(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stats = summarize_attention(
                [[0.4, 0.6, 0.0], [0.1, 0.9, 0.0]],
                [[False, False, True], [False, False, True]],
            )

        self.assertTrue(math.isfinite(stats["entropy"]))
        self.assertFalse(
            any("divide by zero encountered in log" in str(item.message) for item in caught),
            "summarize_attention() should avoid RuntimeWarning when attention contains zeros",
        )


class TemporaryDirectoryContext:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
