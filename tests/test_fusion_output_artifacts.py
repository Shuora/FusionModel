import argparse
import csv
import json
import logging
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fusion_common as fc


class FusionOutputArtifactsTests(unittest.TestCase):
    def test_amp_overflow_is_not_counted_as_invalid_grad_batch(self) -> None:
        class TinyFusionDataset(Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int):
                image = torch.tensor([float(index % 2)], dtype=torch.float32)
                pcap = torch.tensor([float((index + 1) % 2)], dtype=torch.float32)
                label = torch.tensor(index % 2, dtype=torch.long)
                return image, pcap, label

        class TinyFusionModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, images, pcap_data):
                x = torch.cat([images.float(), pcap_data.float()], dim=1)
                return self.fc(x)

        class _ScaledLoss:
            def __init__(self, loss: torch.Tensor) -> None:
                self.loss = loss

            def backward(self) -> None:
                self.loss.backward()

        class FakeGradScaler:
            def __init__(self) -> None:
                self.step_called = 0

            def scale(self, loss: torch.Tensor):
                return _ScaledLoss(loss)

            def unscale_(self, optimizer) -> None:
                return None

            def step(self, optimizer) -> None:
                self.step_called += 1

            def update(self) -> None:
                return None

        class FakeCudaDevice:
            type = "cuda"

        train_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        val_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        model = TinyFusionModel()
        scaler = FakeGradScaler()

        with mock.patch.object(fc, "_autocast_ctx", return_value=nullcontext()), mock.patch.object(
            fc, "_make_grad_scaler", return_value=scaler
        ), mock.patch.object(
            fc, "_has_non_finite_gradients", return_value=True
        ), mock.patch.object(
            fc, "_has_non_finite_parameters", return_value=False
        ), mock.patch.object(
            fc, "evaluate_epoch", return_value=(0.5, 0.5, 0.5, [0, 1], [0, 1])
        ), mock.patch.object(
            torch.Tensor, "to", lambda self, *args, **kwargs: self
        ), mock.patch.object(
            nn.Module, "to", lambda self, *args, **kwargs: self
        ):
            _, history = fc.train_fusion_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1,
                learning_rate=1e-3,
                device=FakeCudaDevice(),
                patience=2,
                use_amp=True,
                early_stop_metric="val_loss",
                early_stop_mode="auto",
                val_every=1,
                max_consecutive_invalid_batches=128,
            )

        self.assertEqual(history["health"]["invalid_grad_batches"], 0)
        self.assertEqual(scaler.step_called, 1)

    def test_non_finite_train_batch_loss_is_skipped(self) -> None:
        class TinyFusionDataset(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int):
                image = torch.tensor([float(index % 2)], dtype=torch.float32)
                pcap = torch.tensor([float((index + 1) % 2)], dtype=torch.float32)
                label = torch.tensor(index % 2, dtype=torch.long)
                return image, pcap, label

        class NaNFirstBatchModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(2, 2)
                self.calls = 0

            def forward(self, images, pcap_data):
                self.calls += 1
                x = torch.cat([images.float(), pcap_data.float()], dim=1)
                logits = self.fc(x)
                if self.calls == 1:
                    return logits * float("nan")
                return logits

        train_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        val_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        model = NaNFirstBatchModel()

        with self.assertLogs(fc.logger, level="WARNING") as log_ctx:
            _, history = fc.train_fusion_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1,
                learning_rate=1e-3,
                device=torch.device("cpu"),
                patience=2,
                use_amp=False,
                early_stop_metric="val_loss",
                early_stop_mode="auto",
                val_every=1,
            )

        self.assertTrue(any("训练损失无效（NaN/Inf），跳过该 batch" in m for m in log_ctx.output))
        self.assertEqual(model.calls, 4)
        self.assertTrue(math.isfinite(history["train_loss"][0]))

    def test_non_finite_val_loss_advances_early_stopping_and_stops(self) -> None:
        class TinyFusionDataset(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int):
                image = torch.tensor([float(index % 2)], dtype=torch.float32)
                pcap = torch.tensor([float((index + 1) % 2)], dtype=torch.float32)
                label = torch.tensor(index % 2, dtype=torch.long)
                return image, pcap, label

        class TinyFusionModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, images, pcap_data):
                x = torch.cat([images.float(), pcap_data.float()], dim=1)
                return self.fc(x)

        train_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        val_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        model = TinyFusionModel()

        with mock.patch.object(
            fc,
            "evaluate_epoch",
            side_effect=[
                (0.5, 0.5, 0.5, [0, 1], [0, 1]),
                (float("nan"), 0.5, 0.5, [0, 1], [0, 1]),
                (float("nan"), 0.5, 0.5, [0, 1], [0, 1]),
            ],
        ):
            _, history = fc.train_fusion_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=3,
                learning_rate=1e-3,
                device=torch.device("cpu"),
                patience=1,
                use_amp=False,
                early_stop_metric="val_loss",
                early_stop_mode="auto",
                val_every=1,
            )

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertTrue(math.isnan(history["val_loss"][1]))

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

    def test_default_stability_guard_params_are_enabled(self) -> None:
        parser = argparse.ArgumentParser()
        fc.add_common_args(parser)
        args = parser.parse_args([])

        self.assertAlmostEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.lr_scheduler, "reduce")
        self.assertEqual(args.lr_patience, 2)
        self.assertAlmostEqual(args.grad_clip_norm, 1.0)

    def test_task_specific_defaults_for_mta_enable_balanced_training(self) -> None:
        parser = argparse.ArgumentParser()
        fc.add_common_args(parser)
        args = parser.parse_args(["--task_name", "mta_multiclass"])

        with mock.patch.object(
            fc,
            "resolve_task_dataset_dirs",
            return_value=("train_img", "train_pcap", "test_img", "test_pcap", "mta_multiclass"),
        ):
            kwargs = fc.build_common_kwargs(args)

        self.assertEqual(kwargs["class_balance"], "weighted_sampler_loss")
        self.assertEqual(kwargs["loss_type"], "focal")
        self.assertAlmostEqual(kwargs["focal_gamma"], 1.5)
        self.assertAlmostEqual(kwargs["label_smoothing"], 0.03)
        self.assertEqual(kwargs["early_stop_metric"], "val_f1")
        self.assertEqual(kwargs["early_stop_mode"], "max")

    def test_task_specific_defaults_do_not_override_explicit_args(self) -> None:
        parser = argparse.ArgumentParser()
        fc.add_common_args(parser)
        args = parser.parse_args(
            [
                "--task_name",
                "mta_multiclass",
                "--class_balance",
                "none",
                "--loss_type",
                "ce",
                "--early_stop_metric",
                "val_loss",
                "--early_stop_mode",
                "auto",
            ]
        )

        with mock.patch.object(
            fc,
            "resolve_task_dataset_dirs",
            return_value=("train_img", "train_pcap", "test_img", "test_pcap", "mta_multiclass"),
        ), mock.patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--class_balance",
                "none",
                "--loss_type",
                "ce",
                "--early_stop_metric",
                "val_loss",
                "--early_stop_mode",
                "auto",
            ],
        ):
            kwargs = fc.build_common_kwargs(args)

        self.assertEqual(kwargs["class_balance"], "none")
        self.assertEqual(kwargs["loss_type"], "ce")
        self.assertEqual(kwargs["early_stop_metric"], "val_loss")
        self.assertEqual(kwargs["early_stop_mode"], "auto")

    def test_build_common_kwargs_contains_charaware_flags(self) -> None:
        parser = argparse.ArgumentParser()
        fc.add_common_args(parser)
        args = parser.parse_args(
            [
                "--task_name",
                "binary_benign_vs_malicious",
                "--charbert_mode",
                "charaware",
                "--char_vocab",
                "ascii",
                "--char_emb_dim",
                "24",
                "--char_cnn_channels",
                "48",
                "--char_fusion",
                "concat",
                "--char_fusion_layers",
                "last",
            ]
        )

        with mock.patch.object(
            fc,
            "resolve_task_dataset_dirs",
            return_value=("train_img", "train_pcap", "test_img", "test_pcap", "binary_benign_vs_malicious"),
        ):
            kwargs = fc.build_common_kwargs(args)

        self.assertEqual(kwargs["charbert_mode"], "charaware")
        self.assertEqual(kwargs["char_vocab"], "ascii")
        self.assertEqual(kwargs["char_emb_dim"], 24)
        self.assertEqual(kwargs["char_cnn_channels"], 48)
        self.assertEqual(kwargs["char_fusion"], "concat")
        self.assertEqual(kwargs["char_fusion_layers"], "last")

    def test_build_common_kwargs_contains_two_level_stacking_flags(self) -> None:
        parser = argparse.ArgumentParser()
        fc.add_common_args(parser)
        args = parser.parse_args(
            [
                "--task_name",
                "mta_multiclass",
                "--stacking_level",
                "two_level",
                "--stacking_calibration",
                "temp",
                "--stacking_threshold_objective",
                "macro_f1_minority_recall",
                "--stacking_minority_lambda",
                "0.4",
                "--stacking_oof_folds",
                "7",
            ]
        )

        with mock.patch.object(
            fc,
            "resolve_task_dataset_dirs",
            return_value=("train_img", "train_pcap", "test_img", "test_pcap", "mta_multiclass"),
        ):
            kwargs = fc.build_common_kwargs(args)

        self.assertEqual(kwargs["stacking_level"], "two_level")
        self.assertEqual(kwargs["stacking_calibration"], "temp")
        self.assertEqual(kwargs["stacking_threshold_objective"], "macro_f1_minority_recall")
        self.assertAlmostEqual(kwargs["stacking_minority_lambda"], 0.4)
        self.assertEqual(kwargs["stacking_oof_folds"], 7)

    def test_training_loader_drops_tail_batch_but_eval_loader_keeps_it(self) -> None:
        class FakeFusionDataset:
            def __init__(self, *args, **kwargs) -> None:
                self.classes = ["a", "b"]
                self.class_counts = [17, 16]
                self.targets = [0] * 17 + [1] * 16

            def __len__(self) -> int:
                return 33

            def __getitem__(self, index: int):
                return index

        class FakeDataLoader:
            def __init__(self, dataset, **kwargs) -> None:
                self.dataset = dataset
                self.drop_last = kwargs.get("drop_last", False)
                self.batch_size = kwargs.get("batch_size")
                self.class_counts = None
                self.classes = None

        fake_transforms = mock.Mock()
        fake_transforms.Compose.return_value = object()
        fake_transforms.Resize.side_effect = lambda size: ("resize", size)
        fake_transforms.ToTensor.return_value = "to_tensor"
        fake_transforms.Lambda.side_effect = lambda fn: ("lambda", fn)

        with mock.patch.object(fc, "FusionDataset", FakeFusionDataset), mock.patch.object(fc, "transforms", fake_transforms), mock.patch.object(fc, "DataLoader", FakeDataLoader):
            train_loader, _ = fc.load_fusion_data(
                image_dir="/tmp/image_train",
                pcap_dir="/tmp/pcap_train",
                batch_size=32,
                num_workers=0,
                is_train=True,
            )
            eval_loader, _ = fc.load_fusion_data(
                image_dir="/tmp/image_eval",
                pcap_dir="/tmp/pcap_eval",
                batch_size=32,
                num_workers=0,
                is_train=False,
            )

        self.assertTrue(train_loader.drop_last)
        self.assertFalse(eval_loader.drop_last)

    def test_consecutive_non_finite_batches_fail_fast(self) -> None:
        class TinyFusionDataset(Dataset):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, index: int):
                image = torch.tensor([float(index % 2)], dtype=torch.float32)
                pcap = torch.tensor([float((index + 1) % 2)], dtype=torch.float32)
                label = torch.tensor(index % 2, dtype=torch.long)
                return image, pcap, label

        class AlwaysNaNModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, images, pcap_data):
                x = torch.cat([images.float(), pcap_data.float()], dim=1)
                logits = self.fc(x)
                return logits * float("nan")

        train_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        val_loader = DataLoader(TinyFusionDataset(), batch_size=2, shuffle=False, num_workers=0)
        model = AlwaysNaNModel()

        _, history = fc.train_fusion_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            learning_rate=1e-3,
            device=torch.device("cpu"),
            patience=8,
            use_amp=False,
            early_stop_metric="val_loss",
            early_stop_mode="auto",
            val_every=1,
            max_consecutive_invalid_batches=2,
        )

        health = history["health"]
        self.assertEqual(health["run_status"], "failed")
        self.assertIn("consecutive_invalid_batches", health["stop_reason"])
        self.assertGreaterEqual(health["invalid_loss_batches"], 2)
        self.assertLessEqual(len(history["train_loss"]), 1)

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

    def test_run_metrics_json_contains_health_fields(self) -> None:
        class FakeLoader:
            def __init__(self) -> None:
                self.dataset = [0, 1]
                self.batch_size = 2
                self.num_workers = 0
                self.pin_memory = False

            def __len__(self) -> int:
                return 1

        fake_loader = FakeLoader()
        classes = ["a", "b"]

        fake_history = {
            "train_loss": [0.4],
            "train_acc": [0.6],
            "train_f1": [0.55],
            "val_loss": [0.5],
            "val_acc": [0.58],
            "val_f1": [0.56],
            "health": {
                "run_status": "degraded",
                "stop_reason": "early_stop",
                "invalid_loss_batches": 3,
                "invalid_grad_batches": 0,
                "invalid_param_events": 0,
                "processed_train_batches": 1,
                "skipped_train_batches": 3,
            },
        }

        fake_eval = {
            "loss": 0.5,
            "acc": 0.58,
            "macro_f1": 0.56,
            "report": "",
            "cm": [[1, 0], [0, 1]],
            "per_class_f1": [0.5, 0.6],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            model = nn.Linear(2, 2)
            with mock.patch.object(
                fc,
                "load_fusion_data",
                side_effect=[(fake_loader, classes), (fake_loader, classes)],
            ), mock.patch.object(fc, "initialize_fusion_model", return_value=model), mock.patch.object(
                fc, "train_fusion_model", return_value=(model, fake_history)
            ), mock.patch.object(fc, "evaluate_full", return_value=fake_eval), mock.patch.object(
                fc, "collect_attention_diagnostics", return_value=None
            ), mock.patch.object(
                fc, "plot_training_curves", return_value=None
            ), mock.patch.object(
                fc, "plot_confusion", return_value=None
            ), mock.patch.object(
                fc, "save_report_md", return_value=None
            ):
                fc.run_fusion_experiment(
                    fusion_mode="attention",
                    train_image_dir="/tmp/train_img",
                    train_pcap_dir="/tmp/train_pcap",
                    test_image_dir="/tmp/test_img",
                    test_pcap_dir="/tmp/test_pcap",
                    batch_size=2,
                    image_size=28,
                    max_pcap_length=16,
                    epochs=1,
                    lr=1e-3,
                    patience=1,
                    device=torch.device("cpu"),
                    output_dir=output_root,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                    prefetch_factor=2,
                    use_amp=False,
                )

            metrics_path = next(output_root.glob("*/metrics.json"))
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_status"], "degraded")
            self.assertEqual(payload["stop_reason"], "early_stop")
            self.assertEqual(payload["health"]["invalid_loss_batches"], 3)

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


    def test_early_stopping_default_patience_is_4(self) -> None:
        stopper = fc.EarlyStopping()
        self.assertEqual(stopper.patience, 4)

    def test_resolve_early_stop_mode_rejects_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            fc._resolve_early_stop_mode("val_f1", "min")

        self.assertEqual(fc._resolve_early_stop_mode("val_loss", "auto"), "min")
        self.assertEqual(fc._resolve_early_stop_mode("val_f1", "auto"), "max")

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
