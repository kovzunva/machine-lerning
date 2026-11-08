import os
import sys
import json
import unittest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("src"))
from service.app import app, predictor

class TestMLServiceOperationalQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
            cls.golden_data = json.load(f)

    # 1. Unit & Health/Ready Tests
    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_ready_endpoint(self):
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["model_loaded"])

    # 2. Contract & Integration Tests
    def test_single_predict_contract(self):
        payload = {"features": self.golden_data[0]["features"]}
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("request_id", body)
        self.assertIn("probability", body)
        self.assertIn("predicted_class", body)
        self.assertEqual(body["positive_class"], "malignant")

    def test_batch_predict_integration(self):
        items = [{"features": item["features"]} for item in self.golden_data]
        resp = self.client.post("/predict-batch", json={"items": items})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total_samples"], len(self.golden_data))

    # 3. Regression Tests on Golden Dataset
    def test_golden_dataset_regression_tolerance(self):
        for idx, golden_case in enumerate(self.golden_data):
            resp = self.client.post("/predict", json={"features": golden_case["features"]})
            self.assertEqual(resp.status_code, 200)
            actual_prob = resp.json()["probability"]
            expected_prob = golden_case["expected_probability"]
            diff = abs(actual_prob - expected_prob)
            self.assertLess(diff, 1e-12, f"Регресія на golden зразку #{idx}: різниця {diff}")

    # 4. Metamorphic Tests
    def test_metamorphic_key_order_invariance(self):
        orig_features = self.golden_data[0]["features"]
        reversed_features = {k: orig_features[k] for k in reversed(list(orig_features.keys()))}
        
        r1 = self.client.post("/predict", json={"features": orig_features}).json()["probability"]
        r2 = self.client.post("/predict", json={"features": reversed_features}).json()["probability"]
        self.assertAlmostEqual(r1, r2, places=12)

    def test_metamorphic_single_vs_batch_consistency(self):
        p_single = self.client.post("/predict", json={"features": self.golden_data[0]["features"]}).json()["probability"]
        p_batch = self.client.post("/predict-batch", json={"items": [{"features": self.golden_data[0]["features"]}]}).json()["predictions"][0]["probability"]
        self.assertAlmostEqual(p_single, p_batch, places=12)

    def test_metamorphic_partition_batch(self):
        items = [{"features": item["features"]} for item in self.golden_data]
        full_preds = self.client.post("/predict-batch", json={"items": items}).json()["predictions"]
        
        part1 = self.client.post("/predict-batch", json={"items": items[:2]}).json()["predictions"]
        part2 = self.client.post("/predict-batch", json={"items": items[2:]}).json()["predictions"]
        
        combined_probs = [p["probability"] for p in part1] + [p["probability"] for p in part2]
        orig_probs = [p["probability"] for p in full_preds]
        np.testing.assert_allclose(combined_probs, orig_probs, atol=1e-12)

    # 5. Negative & Security Tests
    def test_negative_missing_feature(self):
        bad_feats = self.golden_data[0]["features"].copy()
        del bad_feats["mean radius"]
        resp = self.client.post("/predict", json={"features": bad_feats})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error_code"], "INPUT_SCHEMA_VIOLATION")

    def test_negative_extra_feature(self):
        bad_feats = {**self.golden_data[0]["features"], "hack_column": 12.0}
        resp = self.client.post("/predict", json={"features": bad_feats})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error_code"], "INPUT_SCHEMA_VIOLATION")

    def test_negative_wrong_type(self):
        bad_feats = {**self.golden_data[0]["features"], "mean radius": "text_val"}
        resp = self.client.post("/predict", json={"features": bad_feats})
        self.assertEqual(resp.status_code, 422)

    def test_negative_nan_infinity(self):
        raw_payload = '{"features": {"mean smoothness": NaN}}'
        resp = self.client.post("/predict", content=raw_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 422)

    def test_negative_batch_size_exceeded(self):
        big_items = [{"features": self.golden_data[0]["features"]}] * 101
        resp = self.client.post("/predict-batch", json={"items": big_items})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error_code"], "BATCH_SIZE_EXCEEDED")

    def test_negative_out_of_range_warning(self):
        drift_feats = {**self.golden_data[0]["features"], "mean radius": 100.0}
        resp = self.client.post("/predict", json={"features": drift_feats})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(len(resp.json()["warnings"]) > 0)

if __name__ == "__main__":
    unittest.main()