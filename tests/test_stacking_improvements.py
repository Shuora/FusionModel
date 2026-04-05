import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fusion_common as fc


class StackingImprovementTests(unittest.TestCase):
    def test_build_meta_features_from_probs_includes_extra_stats(self) -> None:
        text_probs = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
        image_probs = np.array([[0.7, 0.3], [0.55, 0.45]], dtype=np.float64)
        fusion_probs = np.array([[0.75, 0.25], [0.1, 0.9]], dtype=np.float64)

        feat = fc.build_meta_features_from_probs(text_probs, image_probs, fusion_probs=fusion_probs)

        self.assertEqual(feat.shape, (2, 15))
        np.testing.assert_allclose(feat[:, :2], text_probs, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(feat[:, 2:4], image_probs, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(feat[:, 4:6], fusion_probs, rtol=1e-6, atol=1e-6)

    def test_compute_oof_predictions_covers_each_sample_once(self) -> None:
        x = np.arange(24, dtype=np.float64).reshape(12, 2)
        y = np.array([0, 1, 2] * 4, dtype=np.int64)
        val_indices = []

        def fit_predict(train_x, train_y, val_x):
            _ = train_x, train_y
            probs = np.zeros((val_x.shape[0], 3), dtype=np.float64)
            probs[:, 1] = 1.0
            return probs

        oof_probs = fc.compute_oof_predictions(
            features=x,
            labels=y,
            n_splits=3,
            seed=7,
            fit_predict_fn=fit_predict,
            on_fold=lambda _fold_id, _train_idx, valid_idx: val_indices.extend(valid_idx.tolist()),
        )

        self.assertEqual(oof_probs.shape, (12, 3))
        self.assertCountEqual(val_indices, list(range(12)))

    def test_inverse_frequency_sample_weights_prioritize_minor_classes(self) -> None:
        labels = np.array([0] * 8 + [1] * 2 + [2] * 1, dtype=np.int64)
        w = fc.build_inverse_frequency_sample_weights(labels)
        self.assertEqual(w.shape[0], labels.shape[0])
        self.assertGreater(float(w[labels == 2].mean()), float(w[labels == 0].mean()))
        self.assertGreater(float(w[labels == 1].mean()), float(w[labels == 0].mean()))

    def test_build_deterministic_meta_loader_ignores_weighted_sampler(self) -> None:
        features = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        pcap = torch.arange(10, dtype=torch.int64).unsqueeze(1)
        labels = torch.tensor([0, 1] * 5, dtype=torch.int64)
        dataset = TensorDataset(features, pcap, labels)
        sampler = WeightedRandomSampler(torch.ones(10, dtype=torch.double), num_samples=10, replacement=True)
        sampled_loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)

        deterministic_loader = fc.build_deterministic_meta_loader(sampled_loader)

        observed = []
        for batch in deterministic_loader:
            observed.extend(batch[0].squeeze(1).tolist())
        self.assertEqual(len(observed), 10)
        self.assertListEqual(observed, list(range(10)))

    def test_detect_stacking_special_tasks_supports_mta_with_icedid(self) -> None:
        classes = ["Dridex", "Emotet", "Hancitor", "IcedID", "Qakbot", "Trickbot", "Ursnif"]
        is_mta, is_mfcp = fc.detect_stacking_special_tasks(
            train_classes=classes,
            train_image_dir="/tmp/ProcessedData/mta_multiclass/image_data/Train",
        )
        self.assertTrue(is_mta)
        self.assertFalse(is_mfcp)

    def test_weighted_soft_voting_combines_probabilities(self) -> None:
        p1 = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
        p2 = np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float64)
        vote_probs, vote_pred = fc.weighted_soft_voting([p1, p2], [0.8, 0.2])
        self.assertEqual(vote_probs.shape, (2, 2))
        np.testing.assert_array_equal(vote_pred, np.array([0, 1], dtype=np.int64))

    def test_tune_and_apply_class_gains_for_mta_style_minority_classes(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        probs = np.array(
            [
                [0.25, 0.10, 0.65],
                [0.30, 0.10, 0.60],
                [0.10, 0.25, 0.65],
                [0.10, 0.20, 0.70],
                [0.05, 0.05, 0.90],
                [0.08, 0.08, 0.84],
            ],
            dtype=np.float64,
        )
        base_f1 = fc.f1_score(labels, np.argmax(probs, axis=1), average="macro")
        gains = fc.tune_class_gains(
            labels=labels,
            probs=probs,
            target_classes=[0, 1],
            gain_grid=[1.0, 1.2, 1.5, 2.0, 3.0],
        )
        tuned_probs = fc.apply_class_gains(probs, gains)
        tuned_f1 = fc.f1_score(labels, np.argmax(tuned_probs, axis=1), average="macro")
        self.assertGreaterEqual(tuned_f1, base_f1)
        self.assertTrue(any(v > 1.0 for v in gains.values()))

    def test_binary_pair_correction_head_adjusts_only_target_pair(self) -> None:
        labels = np.array([0, 0, 4, 4, 2], dtype=np.int64)
        features = np.array(
            [
                [0.10, 0.10],
                [0.15, 0.20],
                [0.90, 0.85],
                [0.80, 0.95],
                [0.45, 0.55],
            ],
            dtype=np.float64,
        )
        preds = np.array([4, 4, 0, 0, 2], dtype=np.int64)
        probs = np.array(
            [
                [0.45, 0.0, 0.1, 0.0, 0.45],
                [0.40, 0.0, 0.1, 0.0, 0.50],
                [0.48, 0.0, 0.1, 0.0, 0.42],
                [0.52, 0.0, 0.1, 0.0, 0.35],
                [0.10, 0.0, 0.8, 0.0, 0.10],
            ],
            dtype=np.float64,
        )

        head = fc.fit_binary_centroid_head(features, labels, class_a=0, class_b=4)
        corrected_preds, corrected_probs = fc.apply_binary_correction_for_pair(
            preds=preds,
            probs=probs,
            features=features,
            head=head,
            class_a=0,
            class_b=4,
        )
        self.assertEqual(corrected_preds[4], preds[4])
        self.assertTrue(np.allclose(corrected_probs.sum(axis=1), 1.0))
        self.assertGreaterEqual(
            fc.f1_score(labels, corrected_preds, average="macro"),
            fc.f1_score(labels, preds, average="macro"),
        )

    def test_binary_pair_correction_alpha_zero_keeps_original_predictions(self) -> None:
        labels = np.array([0, 0, 4, 4], dtype=np.int64)
        features = np.array(
            [
                [0.10, 0.10],
                [0.15, 0.20],
                [0.90, 0.85],
                [0.80, 0.95],
            ],
            dtype=np.float64,
        )
        preds = np.array([4, 4, 0, 0], dtype=np.int64)
        probs = np.array(
            [
                [0.45, 0.0, 0.1, 0.0, 0.45],
                [0.40, 0.0, 0.1, 0.0, 0.50],
                [0.48, 0.0, 0.1, 0.0, 0.42],
                [0.52, 0.0, 0.1, 0.0, 0.35],
            ],
            dtype=np.float64,
        )
        head = fc.fit_binary_centroid_head(features, labels, class_a=0, class_b=4)
        corrected_preds, corrected_probs = fc.apply_binary_correction_for_pair(
            preds=preds,
            probs=probs,
            features=features,
            head=head,
            class_a=0,
            class_b=4,
            alpha=0.0,
        )
        np.testing.assert_array_equal(corrected_preds, preds)
        np.testing.assert_allclose(corrected_probs, fc._normalize_probs(probs), rtol=1e-6, atol=1e-6)

    def test_tune_binary_pair_alpha_not_worse_than_baseline_on_oof(self) -> None:
        labels = np.array([0, 0, 0, 4, 4, 4], dtype=np.int64)
        features = np.array(
            [
                [0.20, 0.20],
                [0.25, 0.18],
                [0.18, 0.24],
                [0.80, 0.85],
                [0.75, 0.90],
                [0.90, 0.78],
            ],
            dtype=np.float64,
        )
        probs = np.array(
            [
                [0.40, 0.0, 0.1, 0.0, 0.50],
                [0.45, 0.0, 0.1, 0.0, 0.45],
                [0.38, 0.0, 0.1, 0.0, 0.52],
                [0.60, 0.0, 0.1, 0.0, 0.30],
                [0.48, 0.0, 0.1, 0.0, 0.42],
                [0.58, 0.0, 0.1, 0.0, 0.22],
            ],
            dtype=np.float64,
        )
        head = fc.fit_binary_centroid_head(features, labels, class_a=0, class_b=4)
        alpha = fc.tune_binary_correction_alpha_for_pair(
            labels=labels,
            probs=probs,
            features=features,
            head=head,
            class_a=0,
            class_b=4,
            alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        base_preds = np.argmax(probs, axis=1)
        tuned_preds, _ = fc.apply_binary_correction_for_pair(
            preds=base_preds,
            probs=probs,
            features=features,
            head=head,
            class_a=0,
            class_b=4,
            alpha=alpha,
        )
        self.assertGreaterEqual(
            fc.f1_score(labels, tuned_preds, average="macro"),
            fc.f1_score(labels, base_preds, average="macro"),
        )

    def test_tune_binary_pair_alpha_supports_pair_f1_objective(self) -> None:
        labels = np.array([0, 0, 0, 4, 4, 4, 2, 2], dtype=np.int64)
        features = np.array(
            [
                [0.10, 0.10],
                [0.12, 0.08],
                [0.14, 0.10],
                [0.90, 0.92],
                [0.88, 0.95],
                [0.86, 0.89],
                [0.40, 0.60],
                [0.45, 0.55],
            ],
            dtype=np.float64,
        )
        probs = np.array(
            [
                [0.40, 0.0, 0.6, 0.0, 0.0],
                [0.45, 0.0, 0.55, 0.0, 0.0],
                [0.35, 0.0, 0.65, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 0.55, 0.0, 0.45],
                [0.0, 0.0, 0.52, 0.0, 0.48],
                [0.05, 0.0, 0.92, 0.0, 0.03],
                [0.08, 0.0, 0.89, 0.0, 0.03],
            ],
            dtype=np.float64,
        )
        head = fc.fit_binary_centroid_head(features, labels, class_a=0, class_b=4)
        alpha = fc.tune_binary_correction_alpha_for_pair(
            labels=labels,
            probs=probs,
            features=features,
            head=head,
            class_a=0,
            class_b=4,
            objective="pair_f1",
            alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        base_preds = np.argmax(probs, axis=1)
        tuned_preds, _ = fc.apply_binary_correction_for_pair(
            preds=base_preds,
            probs=probs,
            features=features,
            head=head,
            class_a=0,
            class_b=4,
            alpha=alpha,
        )
        self.assertGreaterEqual(
            fc.score_pair_f1(labels, tuned_preds, class_a=0, class_b=4),
            fc.score_pair_f1(labels, base_preds, class_a=0, class_b=4),
        )

    def test_pair_calibration_and_threshold_not_worse_than_baseline(self) -> None:
        labels = np.array([0, 0, 4, 4, 2, 2], dtype=np.int64)
        probs = np.array(
            [
                [0.35, 0.0, 0.55, 0.0, 0.10],
                [0.45, 0.0, 0.45, 0.0, 0.10],
                [0.12, 0.0, 0.58, 0.0, 0.30],
                [0.08, 0.0, 0.62, 0.0, 0.30],
                [0.05, 0.0, 0.90, 0.0, 0.05],
                [0.10, 0.0, 0.80, 0.0, 0.10],
            ],
            dtype=np.float64,
        )
        base_preds = np.argmax(probs, axis=1)
        base_pair_f1 = fc.score_pair_f1(labels, base_preds, class_a=0, class_b=4)

        temperature = fc.tune_pair_temperature(
            labels=labels,
            probs=probs,
            class_a=0,
            class_b=4,
            temperature_grid=[0.7, 1.0, 1.3, 1.6],
        )
        calibrated = fc.apply_pair_temperature(
            probs=probs,
            class_a=0,
            class_b=4,
            temperature=temperature,
        )
        threshold = fc.tune_pair_threshold(
            labels=labels,
            probs=calibrated,
            class_a=0,
            class_b=4,
            threshold_grid=[0.3, 0.4, 0.5, 0.6, 0.7],
        )
        tuned_preds = fc.apply_pair_threshold(
            preds=np.argmax(calibrated, axis=1),
            probs=calibrated,
            class_a=0,
            class_b=4,
            threshold=threshold,
        )
        self.assertGreaterEqual(fc.score_pair_f1(labels, tuned_preds, class_a=0, class_b=4), base_pair_f1)


if __name__ == "__main__":
    unittest.main()
