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

    def test_attention_stacking_parser_has_two_level_args(self) -> None:
        parser = build_attention_stacking_parser()
        args = parser.parse_args(["--task_name", "mta_multiclass"])
        self.assertEqual(args.stacking_level, "two_level")
        self.assertEqual(args.stacking_calibration, "temp")
        self.assertEqual(args.stacking_threshold_objective, "macro_f1_minority_recall")
        self.assertAlmostEqual(args.stacking_minority_lambda, 0.3)
        self.assertEqual(args.stacking_oof_folds, 5)

    def test_attention_stacking_parser_supports_accuracy_threshold_objective(self) -> None:
        parser = build_attention_stacking_parser()
        args = parser.parse_args(
            ["--task_name", "mfcp_multiclass", "--stacking_threshold_objective", "accuracy"]
        )
        self.assertEqual(args.stacking_threshold_objective, "accuracy")

    def test_attention_parser_has_charaware_args_with_compatible_defaults(self) -> None:
        parser = build_attention_parser()
        args = parser.parse_args(["--task_name", "binary_benign_vs_malicious"])
        self.assertEqual(args.charbert_mode, "charaware")
        self.assertEqual(args.char_vocab, "hex")
        self.assertEqual(args.char_emb_dim, 32)
        self.assertEqual(args.char_cnn_channels, 64)
        self.assertEqual(args.char_fusion, "gated")
        self.assertEqual(args.char_fusion_layers, "all")


if __name__ == "__main__":
    unittest.main()
