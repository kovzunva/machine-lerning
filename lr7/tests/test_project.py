import unittest
import os
import sys
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath("src"))

from ml_project.config import load_project_config, ConfigurationError
from ml_project.validation import SchemaValidator, InputValidationError
from ml_project.predictor import ProjectPredictor, sha256_file
from ml_project.train import rebuild_model

class TestMLProjectCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = "configs/project_config.json"
        cls.config = load_project_config(cls.config_path)
        cls.predictor = ProjectPredictor(cls.config)
        with open("configs/inference_example.json", "r", encoding="utf-8") as f:
            cls.example_record = json.load(f)

    # 1. Тести конфігурації
    def test_config_loaded_properly(self):
        self.assertEqual(self.config.project_name, "breast_cancer_reproducible_core")
        self.assertEqual(self.config.threshold, 0.5)

    def test_config_invalid_threshold_raises(self):
        bad_config_data = {
            "project_name": "test", "version": "1.0.0",
            "model_path": "models/selected_pipeline.joblib",
            "schema_path": "configs/data_schema.json",
            "metadata_path": "models/model_metadata.json",
            "train_ids_path": "data/train_ids.csv",
            "test_ids_path": "data/test_ids.csv",
            "threshold": 1.5,
            "positive_class_label": "malignant",
            "max_batch_size": 200, "numeric_tolerance": 1e-12
        }
        temp_cfg_path = "configs/temp_bad_config.json"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            json.dump(bad_config_data, f)
        try:
            with self.assertRaises(ConfigurationError):
                load_project_config(temp_cfg_path)
        finally:
            if os.path.exists(temp_cfg_path):
                os.remove(temp_cfg_path)

    # 2. Тести валідатора схеми
    def test_validation_clean_record_success(self):
        df_rec, warns = self.predictor.validator.validate_record(self.example_record)
        self.assertEqual(df_rec.shape[1], 30)

    def test_validation_missing_feature_raises(self):
        bad_rec = self.example_record.copy()
        del bad_rec["mean radius"]
        with self.assertRaises(InputValidationError):
            self.predictor.validator.validate_record(bad_rec)

    def test_validation_extra_feature_raises(self):
        bad_rec = {**self.example_record, "unexpected_column": 12.3}
        with self.assertRaises(InputValidationError):
            self.predictor.validator.validate_record(bad_rec)

    def test_validation_wrong_type_raises(self):
        bad_rec = {**self.example_record, "mean radius": "not_a_number"}
        with self.assertRaises(InputValidationError):
            self.predictor.validator.validate_record(bad_rec)

    def test_validation_nan_raises(self):
        bad_rec = {**self.example_record, "mean texture": np.nan}
        with self.assertRaises(InputValidationError):
            self.predictor.validator.validate_record(bad_rec)

    def test_validation_infinity_raises(self):
        bad_rec = {**self.example_record, "mean smoothness": np.inf}
        with self.assertRaises(InputValidationError):
            self.predictor.validator.validate_record(bad_rec)

    def test_validation_order_invariant(self):
        reversed_rec = {k: self.example_record[k] for k in reversed(list(self.example_record.keys()))}
        df1, _ = self.predictor.validator.validate_record(self.example_record)
        df2, _ = self.predictor.validator.validate_record(reversed_rec)
        np.testing.assert_array_equal(df1.values, df2.values)

    def test_validation_out_of_range_warning(self):
        bad_rec = {**self.example_record, "mean radius": 100.0}
        _, warns = self.predictor.validator.validate_record(bad_rec)
        self.assertTrue(len(warns) > 0)

    def test_batch_limit_exceeded_raises(self):
        records = [self.example_record] * 201
        with self.assertRaises(InputValidationError):
            self.predictor.validator.validate_batch(records, max_batch_size=200)

    # 3. Інтеграційні тести відтворюваності прогнозу
    def test_prediction_reproducibility_tolerance(self):
        res = self.predictor.predict_record(self.example_record)
        expected_prob = self.predictor.metadata["expected_inference_prob"]
        diff = abs(res["probability"] - expected_prob)
        self.assertLess(diff, self.config.numeric_tolerance)

    # 4. Процедурний Rebuild: перевірка числової тотожності ваг та прогнозів
    def test_rebuild_procedure_equivalence(self):
        rebuilt_path = rebuild_model(output_model_path="models/test_rebuilt.joblib")
        orig_pipe = joblib.load("models/selected_pipeline.joblib")
        rebuilt_pipe = joblib.load(rebuilt_path)

        # Перевірка ваг логістичної регресії
        orig_coef = orig_pipe.named_steps["model"].coef_
        rebuilt_coef = rebuilt_pipe.named_steps["model"].coef_
        np.testing.assert_allclose(orig_coef, rebuilt_coef, atol=1e-12)

        # Перевірка прогнозу на контрольному прикладі
        df_rec, _ = self.predictor.validator.validate_record(self.example_record)
        p_orig = orig_pipe.predict_proba(df_rec)[0, 1]
        p_rebuilt = rebuilt_pipe.predict_proba(df_rec)[0, 1]
        self.assertLess(abs(p_orig - p_rebuilt), self.config.numeric_tolerance)

        if os.path.exists(rebuilt_path):
            os.remove(rebuilt_path)

if __name__ == "__main__":
    unittest.main()