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

    def test_charbert_build_model_supports_charaware_mode(self) -> None:
        charbert_src = Path(resolve_charbert_src())
        if str(charbert_src) not in sys.path:
            sys.path.insert(0, str(charbert_src))

        from config import TrainingConfig  # type: ignore
        from model import build_model  # type: ignore

        cfg = TrainingConfig()
        cfg.mode = "charaware"
        cfg.char_vocab = "hex"
        cfg.char_emb_dim = 16
        cfg.char_cnn_channels = 32
        cfg.char_fusion = "gated"
        cfg.char_fusion_layers = "all"
        cfg.max_len = 32
        cfg.hidden_size = 32
        cfg.num_layers = 2
        cfg.num_heads = 4
        model = build_model(cfg, num_labels=7)

        import torch

        x = torch.randint(low=0, high=259, size=(2, 32), dtype=torch.long)
        x[:, -4:] = 256
        attn = (x != 256).long()
        logits = model(x, attention_mask=attn)
        self.assertEqual(tuple(logits.shape), (2, 7))


if __name__ == '__main__':
    unittest.main()
