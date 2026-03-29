import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_all_modes import build_parser


class RunAllModesTests(unittest.TestCase):
    def test_parser_only_exposes_attention_modes(self) -> None:
        parser = build_parser()
        mode_action = next(action for action in parser._actions if action.dest == 'mode')
        self.assertEqual(sorted(mode_action.choices), ['all', 'attention', 'attention_stacking'])

    def test_parser_requires_and_accepts_task_name_argument(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--mode', 'attention'])
        args = parser.parse_args(['--mode', 'attention', '--task_name', 'ustc_multiclass'])
        self.assertEqual(args.mode, 'attention')
        self.assertEqual(args.task_name, 'ustc_multiclass')


if __name__ == '__main__':
    unittest.main()
