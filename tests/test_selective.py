import sys
import unittest
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from selective_eval import (
    _restrict_logits,
    compute_msp_confidence,
    compute_selective_metrics,
    select_validation_threshold,
)


class TestSelectiveEval(unittest.TestCase):
    def test_compute_msp_confidence(self):
        logits = np.array([
            [10.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        conf, preds = compute_msp_confidence(logits)
        self.assertEqual(len(conf), 3)
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)
        # First sample is very confident
        self.assertGreater(conf[0], 0.99)
        # Third sample is uniform over 3 classes -> 1/3
        self.assertTrue(np.isclose(conf[2], 1.0 / 3.0))

    def test_restrict_logits(self):
        logits = np.array([
            [2.0, 5.0, 3.0],
            [4.0, 1.0, 2.0],
        ])
        # Restrict to classes 0 and 2
        restricted = _restrict_logits(logits, [0, 2])
        self.assertEqual(restricted[0, 1], -np.inf)
        self.assertEqual(restricted[1, 1], -np.inf)
        self.assertEqual(restricted[0, 0], 2.0)
        self.assertEqual(restricted[0, 2], 3.0)

        conf, preds = compute_msp_confidence(restricted)
        self.assertNotIn(1, preds)
        self.assertEqual(preds[0], 2)
        self.assertEqual(preds[1], 0)

    def test_empty_accepted_subset_safety(self):
        logits = np.array([
            [2.0, 1.0],
            [1.0, 2.0],
        ])
        targets = np.array([0, 1])
        class_names = ["c0", "c1"]

        # Threshold 1.5 is impossible to meet
        metrics = compute_selective_metrics(logits, targets, class_names, threshold=1.5)
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["accepted_samples"], 0)
        self.assertEqual(metrics["abstained_samples"], 2)
        self.assertEqual(metrics["coverage"], 0.0)
        self.assertEqual(metrics["abstention_rate"], 1.0)
        self.assertIsNone(metrics["selective_accuracy"])
        self.assertIsNone(metrics["macro_f1"])
        self.assertIsNone(metrics["macro_f1_eligible"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["n_classes_active"], 0)
        self.assertEqual(metrics["n_classes_eligible"], 2)

    def test_threshold_zero_matches_baseline(self):
        logits = np.array([
            [5.0, 1.0],
            [1.0, 5.0],
            [5.0, 1.0],
        ])
        targets = np.array([0, 1, 1])  # sample 2 is wrong
        class_names = ["c0", "c1"]

        metrics = compute_selective_metrics(logits, targets, class_names, threshold=0.0)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["accepted_samples"], 3)
        self.assertEqual(metrics["abstained_samples"], 0)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(metrics["abstention_rate"], 0.0)
        self.assertTrue(np.isclose(metrics["selective_accuracy"], 2.0 / 3.0))
        self.assertTrue(np.isclose(metrics["baseline_accuracy"], 2.0 / 3.0))

    def test_select_validation_threshold_rules(self):
        # Construct synthetic validation set with known confidences
        # 8 correct with conf 0.95, 2 errors with conf 0.40
        logits_correct = np.tile([10.0, 0.0], (8, 1))
        logits_error = np.tile([0.1, 0.2], (2, 1))  # pred=1, true=0, conf ~0.52
        logits = np.vstack([logits_correct, logits_error])
        targets = np.zeros(10, dtype=int)

        # 1. Target accuracy rule
        res_acc = select_validation_threshold(
            logits, targets, rule="target_accuracy", target_accuracy=1.0
        )
        self.assertGreater(res_acc["selected_threshold"], 0.5)
        self.assertEqual(res_acc["val_selective_accuracy"], 1.0)
        self.assertEqual(res_acc["val_coverage"], 0.8)

        # 2. Target coverage rule
        res_cov = select_validation_threshold(
            logits, targets, rule="target_coverage", target_coverage=0.8
        )
        self.assertGreaterEqual(res_cov["val_coverage"], 0.8)

        # 3. Youden's J
        res_j = select_validation_threshold(logits, targets, rule="youden_j")
        self.assertGreater(res_j["selected_threshold"], 0.5)

    def test_macro_f1_penalizes_predicted_classes_absent_from_ground_truth(self):
        """Ensure false positives for classes not present in targets lower macro-F1."""
        targets = np.array([0, 0, 1, 1])
        logits = np.array([
            [10.0, 0.0, 0.0, 0.0],  # true 0, pred 0 (TP for 0)
            [0.0, 0.0, 10.0, 0.0],  # true 0, pred 2 (FN for 0, FP for 2)
            [0.0, 10.0, 0.0, 0.0],  # true 1, pred 1 (TP for 1)
            [0.0, 10.0, 0.0, 0.0],  # true 1, pred 1 (TP for 1)
        ])
        class_names = ["c0", "c1", "c2", "c3"]
        metrics = compute_selective_metrics(
            logits, targets, class_names, threshold=0.5, eligible_label_ids=[0, 1, 2, 3]
        )

        # Class 0: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3
        # Class 1: TP=2, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        # Class 2: TP=0, FP=1, FN=0 -> P=0.0, R=0.0, F1=0.0
        # Active union {0, 1, 2}: (2/3 + 1.0 + 0.0)/3 = 5/9 ≈ 0.5556
        # Eligible {0, 1, 2, 3}: (2/3 + 1.0 + 0.0 + 0.0)/4 = 5/12 ≈ 0.4167
        self.assertAlmostEqual(metrics["macro_f1"], 5.0 / 9.0, places=4)
        self.assertAlmostEqual(metrics["macro_f1_eligible"], 5.0 / 12.0, places=4)
        self.assertEqual(metrics["n_classes_active"], 3)
        self.assertEqual(metrics["n_classes_eligible"], 4)

        # Confirm it is strictly lower than the buggy calculation that ignored class 2:
        buggy_f1 = (2.0 / 3.0 + 1.0) / 2.0  # 0.8333
        self.assertLess(metrics["macro_f1"], buggy_f1)


if __name__ == "__main__":
    unittest.main()
