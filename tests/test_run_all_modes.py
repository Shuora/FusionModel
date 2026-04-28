import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_all_modes as ram
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

    def test_main_reseeds_before_each_mode_run(self) -> None:
        args = argparse.Namespace(mode='all', seed=77, task_name='ustc_multiclass')
        parser = mock.Mock()
        parser.parse_args.return_value = args
        kwargs = {'output_dir': Path('/tmp/fusion-outputs')}

        with mock.patch.object(ram, 'build_parser', return_value=parser), mock.patch.object(
            ram, 'build_common_kwargs', return_value=kwargs
        ), mock.patch.object(ram, 'ensure_output_dirs'), mock.patch.object(
            ram, 'run_fusion_experiment'
        ), mock.patch.object(
            ram, 'run_stacking_experiment'
        ), mock.patch.object(
            ram, 'set_seed'
        ) as seed_mock, mock.patch.object(
            ram.time, 'time', side_effect=[100.0, 104.0]
        ):
            rc = ram.main()

        self.assertEqual(rc, 0)
        self.assertEqual(seed_mock.call_args_list, [mock.call(77), mock.call(77)])

    def test_main_reseeds_once_for_single_mode(self) -> None:
        args = argparse.Namespace(mode='attention', seed=19, task_name='ustc_multiclass')
        parser = mock.Mock()
        parser.parse_args.return_value = args
        kwargs = {'output_dir': Path('/tmp/fusion-outputs')}

        with mock.patch.object(ram, 'build_parser', return_value=parser), mock.patch.object(
            ram, 'build_common_kwargs', return_value=kwargs
        ), mock.patch.object(ram, 'ensure_output_dirs'), mock.patch.object(
            ram, 'run_fusion_experiment'
        ), mock.patch.object(
            ram, 'run_stacking_experiment'
        ), mock.patch.object(
            ram, 'set_seed'
        ) as seed_mock, mock.patch.object(
            ram.time, 'time', side_effect=[200.0, 201.0]
        ):
            rc = ram.main()

        self.assertEqual(rc, 0)
        self.assertEqual(seed_mock.call_args_list, [mock.call(19)])


if __name__ == '__main__':
    unittest.main()
