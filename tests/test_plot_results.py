import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plot_results import (
    load_json,
    validate_cross_eval,
    validate_abstention,
    plot_generalization_gap,
    plot_plantdoc_risk_coverage,
    plot_plantdoc_threshold_tradeoff,
)


class TestPlotResults(unittest.TestCase):
    def setUp(self):
        self.cross_eval_path = Path("results/resnet18_baseline/cross_eval.json")
        self.abstention_path = Path("results/resnet18_baseline/abstention/summary.json")
        self.test_out_dir = Path("tests/scratch_figures")
        self.test_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up scratch test outputs
        for f in self.test_out_dir.glob("*.png"):
            f.unlink()
        if self.test_out_dir.exists():
            self.test_out_dir.rmdir()

    def test_load_json_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_json(Path("non_existent_file.json"))

    def test_validate_cross_eval_missing_key(self):
        bad_data = {"results": {"plantvillage_test_shared": {}}}
        with self.assertRaises(KeyError):
            validate_cross_eval(bad_data, Path("dummy.json"))

    def test_validate_abstention_missing_key(self):
        bad_data = {"results": {"plantdoc_open": {}}}
        with self.assertRaises(KeyError):
            validate_abstention(bad_data, Path("dummy.json"))

    def test_plot_generation_from_artifacts(self):
        cross_eval = load_json(self.cross_eval_path)
        validate_cross_eval(cross_eval, self.cross_eval_path)

        abstention = load_json(self.abstention_path)
        validate_abstention(abstention, self.abstention_path)

        # 1. Generalization gap
        p1 = self.test_out_dir / "test_gap.png"
        m1 = plot_generalization_gap(cross_eval, p1, dpi=200)
        self.assertTrue(p1.is_file())
        self.assertGreater(p1.stat().st_size, 1000)
        self.assertIn("plantvillage_test_shared_accuracy", m1)

        # 2. Risk-coverage
        p2 = self.test_out_dir / "test_risk_cov.png"
        m2 = plot_plantdoc_risk_coverage(abstention, p2, dpi=200)
        self.assertTrue(p2.is_file())
        self.assertGreater(p2.stat().st_size, 1000)
        self.assertIn("open_val_selected_threshold", m2)

        # 3. Threshold trade-off
        p3 = self.test_out_dir / "test_tradeoff.png"
        m3 = plot_plantdoc_threshold_tradeoff(abstention, p3, dpi=200)
        self.assertTrue(p3.is_file())
        self.assertGreater(p3.stat().st_size, 1000)
        self.assertIn("open_thresholds", m3)


if __name__ == "__main__":
    unittest.main()
