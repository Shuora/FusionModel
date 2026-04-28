import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task_config import get_task_config


class TaskConfigTests(unittest.TestCase):
    def test_binary_task_exists_and_uses_expected_labels(self) -> None:
        cfg = get_task_config("binary_benign_vs_malicious")
        self.assertEqual(cfg.name, "binary_benign_vs_malicious")
        self.assertEqual(list(cfg.labels), ["benign", "malicious"])

    def test_multiclass_tasks_are_dataset_specific(self) -> None:
        self.assertEqual(list(get_task_config("ustc_multiclass").dataset_names), ["USTC-TFC2016"])
        self.assertEqual(list(get_task_config("mta_multiclass").dataset_names), ["MTA"])
        self.assertEqual(list(get_task_config("mfcp_multiclass").dataset_names), ["MFCP"])

    def test_unknown_task_raises_key_error(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            get_task_config("missing_task")
        self.assertIn("missing_task", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
