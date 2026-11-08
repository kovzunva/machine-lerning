import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import audit_core as ac

class TestAuditCore(unittest.TestCase):

    def test_detect_split_overlap_clean(self):
        train_ids = [1, 2, 3, 4]
        test_ids = [5, 6, 7]
        overlap = ac.detect_split_overlap(train_ids, test_ids)
        self.assertEqual(len(overlap), 0)

    def test_detect_split_overlap_dirty(self):
        train_ids = [1, 2, 3, 4]
        test_ids = [4, 5, 6]
        overlap = ac.detect_split_overlap(train_ids, test_ids)
        self.assertIn(4, overlap)

    def test_feature_target_correlations_proxy(self):
        y = np.array([0, 1, 0, 1, 1])
        df = pd.DataFrame({"normal_col": [0.1, 0.2, 0.4, 0.3, 0.5], "proxy_col": y})
        res = ac.feature_target_correlations(df, y)
        self.assertEqual(res.iloc[0]["feature"], "proxy_col")
        self.assertAlmostEqual(res.iloc[0]["abs_correlation"], 1.0)

    def test_feature_target_correlations_order(self):
        y = np.array([0, 1, 0, 1])
        df = pd.DataFrame({"low": [0.1, 0.1, 0.2, 0.2], "high": [0.0, 0.9, 0.1, 0.8]})
        res = ac.feature_target_correlations(df, y)
        self.assertTrue(res.iloc[0]["abs_correlation"] >= res.iloc[1]["abs_correlation"])

    def test_evaluate_model_dummy(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 0, 1])
        dummy = DummyClassifier(strategy="prior").fit(X, y)
        metrics = ac.evaluate_model(dummy, X, y)
        self.assertIn("average_precision", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertEqual(metrics["roc_auc"], 0.5)

    def test_evaluate_model_decision_function(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        lr = LogisticRegression().fit(X, y)
        metrics = ac.evaluate_model(lr, X, y)
        self.assertGreaterEqual(metrics["average_precision"], 0.9)

    def test_validate_record_clean(self):
        schema = {"f1": {"min": 0.0, "max": 10.0}}
        record = {"f1": 5.0}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertTrue(valid)
        self.assertEqual(status, "accepted")
        self.assertEqual(len(msgs), 0)

    def test_validate_record_missing_field(self):
        schema = {"f1": {"min": 0, "max": 10}, "f2": {"min": 0, "max": 10}}
        record = {"f1": 5.0}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertFalse(valid)
        self.assertEqual(status, "rejected")

    def test_validate_record_extra_field(self):
        schema = {"f1": {"min": 0, "max": 10}}
        record = {"f1": 5.0, "unexpected": 1.0}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertFalse(valid)
        self.assertEqual(status, "rejected")

    def test_validate_record_wrong_type(self):
        schema = {"f1": {"min": 0, "max": 10}}
        record = {"f1": "text_val"}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertFalse(valid)
        self.assertEqual(status, "rejected")

    def test_validate_record_nan_inf(self):
        schema = {"f1": {"min": 0, "max": 10}}
        record = {"f1": np.nan}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertFalse(valid)
        self.assertEqual(status, "rejected")

    def test_validate_record_infinity(self):
        schema = {"f1": {"min": 0, "max": 10}}
        record = {"f1": np.inf}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertFalse(valid)
        self.assertEqual(status, "rejected")

    def test_validate_record_out_of_range_warning(self):
        schema = {"f1": {"min": 0.0, "max": 10.0}}
        record = {"f1": 25.0}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertTrue(valid)
        self.assertEqual(status, "accepted_with_warning")

    def test_verify_claim_structure(self):
        res = ac.verify_claim("C-01", "Text", {"ap": 1.0}, lambda e: "Спростовано" if e["ap"] == 1.0 else "Підтверджено")
        self.assertEqual(res["claim_id"], "C-01")
        self.assertEqual(res["verdict"], "Спростовано")

    def test_build_finding_structure(self):
        f = ac.build_finding("F-01", "Leakage", "r=1.0", "Критична", "Вилучити", "Виправлено")
        self.assertEqual(f["id"], "F-01")
        self.assertEqual(f["severity"], "Критична")
        self.assertEqual(f["status"], "Виправлено")

    def test_negative_suite_permutation_invariant(self):
        schema = {"f1": {"min": 0, "max": 10}, "f2": {"min": 0, "max": 10}}
        record = {"f2": 5.0, "f1": 2.0}
        valid, status, msgs = ac.validate_record(record, schema)
        self.assertTrue(valid)
        self.assertEqual(status, "accepted")

if __name__ == "__main__":
    unittest.main()