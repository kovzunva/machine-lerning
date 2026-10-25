import os
import unittest
import numpy as np
import pandas as pd
from lr6 import (
    bootstrap_mean_interval,
    pairwise_prediction_disagreement,
    prediction_latency_ms,
    sha256_file,
    get_manual_candidates
)


class TestControlledComparison(unittest.TestCase):

    def test_manual_candidates_count(self):
        cands = get_manual_candidates(seed=42)
        self.assertEqual(len(cands), 12)

    def test_bootstrap_mean_interval_reproducibility(self):
        diffs = np.array([0.01, -0.02, 0.005, -0.001, -0.03])
        low1, up1, _ = bootstrap_mean_interval(diffs, n_bootstrap=1000, seed=42)
        low2, up2, _ = bootstrap_mean_interval(diffs, n_bootstrap=1000, seed=42)
        self.assertEqual(low1, low2)
        self.assertEqual(up1, up2)
        self.assertTrue(low1 < up1)

    def test_pairwise_prediction_disagreement(self):
        p1 = np.array([0.9, 0.1, 0.8])
        p2 = np.array([0.9, 0.8, 0.8])  # 2-й об'єкт відрізняється
        disagree = pairwise_prediction_disagreement([p1, p2], threshold=0.50)
        self.assertAlmostEqual(disagree, 1/3)

    def test_sha256_computation(self):
        temp_file = "temp_hash_test.txt"
        with open(temp_file, "w") as f:
            f.write("test_content")
        h = sha256_file(temp_file)
        self.assertEqual(len(h), 64)
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def test_handoff_files_exist(self):
        if os.path.exists("handoff"):
            self.assertTrue(os.path.exists(os.path.join("handoff", "selected_pipeline.joblib")))
            self.assertTrue(os.path.exists(os.path.join("handoff", "selection_decision.json")))
            self.assertTrue(os.path.exists(os.path.join("handoff", "data_schema.json")))

    def test_oof_predictions_completeness(self):
        oof_path = os.path.join("results", "oof_predictions.csv")
        if os.path.exists(oof_path):
            df = pd.read_csv(oof_path)
            self.assertEqual(len(df), 426)
            self.assertFalse(df.isna().any().any())

    def test_decision_freeze_pretest_flag(self):
        dec_path = os.path.join("results", "selection_decision_pretest.json")
        if os.path.exists(dec_path):
            import json
            with open(dec_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.assertFalse(d["test_metrics_present"])


if __name__ == "__main__":
    unittest.main()