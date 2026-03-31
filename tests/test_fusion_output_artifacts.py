import argparse
import csv
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fusion_common as fc


class FusionOutputArtifactsTests(unittest.TestCase):
    def test_default_output_dir_points_to_repo_root_outputs(self) -> None:
        parser = argparse.ArgumentParser()
        fc.add_common_args(parser)

        args = parser.parse_args([])

        self.assertEqual(Path(args.output_dir), ROOT / "outputs")

    def test_default_log_dir_points_to_repo_root_outputs_logs(self) -> None:
        log_path = fc.setup_logging(force=True)
        logging.shutdown()

        self.assertEqual(log_path.parent, ROOT / "outputs" / "logs")
        self.assertTrue(log_path.exists())
        log_path.unlink(missing_ok=True)

    def test_export_metrics_artifacts_writes_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_a"
            run_dir.mkdir(parents=True, exist_ok=True)

            history = {
                "train_loss": [1.0, 0.5],
                "train_acc": [0.3, 0.6],
                "train_f1": [0.2, 0.55],
                "val_loss": [1.1, 0.7],
                "val_acc": [0.28, 0.58],
                "val_f1": [0.18, 0.5],
            }
            metrics_payload = {
                "mode": "attention",
                "run_name": "run_a",
                "eval": {"acc": 0.58, "macro_f1": 0.5},
                "confusion_matrix": [[3, 1], [2, 4]],
            }

            metrics_path, epoch_csv_path = fc.export_metrics_artifacts(
                run_dir=run_dir,
                history=history,
                metrics_payload=metrics_payload,
            )

            self.assertTrue(metrics_path.exists())
            self.assertTrue(epoch_csv_path.exists())

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "attention")
            self.assertEqual(payload["run_name"], "run_a")
            self.assertEqual(payload["eval"]["acc"], 0.58)

            with epoch_csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertEqual(rows[1]["train_loss"], "0.5")
            self.assertEqual(rows[1]["val_f1"], "0.5")

    def test_run_directory_isolation_prevents_fixed_filename_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            run_attention = fc.prepare_run_output_dir(output_root, "attention_20260329_120000")
            run_stacking = fc.prepare_run_output_dir(output_root, "attention_stacking_20260329_120000")
            run_attention_second = fc.prepare_run_output_dir(output_root, "attention_20260329_120000")

            self.assertNotEqual(run_attention, run_stacking)
            self.assertNotEqual(run_attention, run_attention_second)
            self.assertTrue(run_attention_second.name.startswith("attention_20260329_120000"))

            paths_a = fc.build_run_artifact_paths(run_attention)
            paths_b = fc.build_run_artifact_paths(run_stacking)

            paths_a["metrics_curve"].write_text("attention", encoding="utf-8")
            paths_b["metrics_curve"].write_text("stacking", encoding="utf-8")

            self.assertEqual(paths_a["metrics_curve"].name, "metrics_curve.png")
            self.assertEqual(paths_b["metrics_curve"].name, "metrics_curve.png")
            self.assertEqual(paths_a["metrics_curve"].read_text(encoding="utf-8"), "attention")
            self.assertEqual(paths_b["metrics_curve"].read_text(encoding="utf-8"), "stacking")

    def test_load_pyplot_headless_uses_agg_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plt = fc.load_pyplot_headless()
            fig = plt.figure()
            output_path = Path(tmpdir) / "headless_plot.png"
            fig.savefig(output_path)
            plt.close(fig)

            self.assertTrue(output_path.exists())
            self.assertIn("agg", plt.get_backend().lower())


if __name__ == "__main__":
    unittest.main()
