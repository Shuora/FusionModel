import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_fusion_attention import build_parser as build_attention_parser
from train_fusion_attention_stacking import build_parser as build_attention_stacking_parser


class AttentionEntrypointTests(unittest.TestCase):
    def test_attention_parser_requires_task_name(self) -> None:
        parser = build_attention_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["--task_name", "binary_benign_vs_malicious"])
        self.assertEqual(args.task_name, "binary_benign_vs_malicious")

    def test_attention_stacking_parser_requires_task_name(self) -> None:
        parser = build_attention_stacking_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["--task_name", "mta_multiclass"])
        self.assertEqual(args.task_name, "mta_multiclass")
        self.assertEqual(args.meta_methods, "xgboost")

    def test_attention_parser_has_charaware_args_with_compatible_defaults(self) -> None:
        parser = build_attention_parser()
        args = parser.parse_args(["--task_name", "binary_benign_vs_malicious"])
        self.assertEqual(args.charbert_mode, "legacy")
        self.assertEqual(args.char_vocab, "hex")
        self.assertEqual(args.char_emb_dim, 32)
        self.assertEqual(args.char_cnn_channels, 64)
        self.assertEqual(args.char_fusion, "gated")
        self.assertEqual(args.char_fusion_layers, "all")


if __name__ == "__main__":
    unittest.main()
