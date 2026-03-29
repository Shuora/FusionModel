import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion_common import resolve_charbert_src


class CharBertLoaderTests(unittest.TestCase):
    def test_resolve_charbert_src_points_to_existing_model_files(self) -> None:
        charbert_src = Path(resolve_charbert_src())
        self.assertTrue(charbert_src.is_dir())
        self.assertTrue((charbert_src / 'model.py').exists())
        self.assertTrue((charbert_src / 'config.py').exists())


if __name__ == '__main__':
    unittest.main()
