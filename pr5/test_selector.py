import copy
import os
import unittest
import numpy as np
import pandas as pd

from pr5 import check_constraints, normalize_value, rank_models, validate_config


class TestModelSelector(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "scenario": "test_scenario",
            "constraints": {
                "recall": {"min": 0.80},
                "latency": {"max": 10.0}
            },
            "criteria": {
                "f1": {"direction": "maximize", "weight": 0.60, "worst": 0.70, "best": 0.95},
                "latency": {"direction": "minimize", "weight": 0.40, "worst": 10.0, "best": 2.0}
            }
        }

    # 1. Нормалізація критерію виграшу
    def test_normalize_maximize(self):
        spec = {"direction": "maximize", "worst": 0.50, "best": 1.00}
        self.assertAlmostEqual(normalize_value(0.75, spec), 0.50)
        self.assertAlmostEqual(normalize_value(1.00, spec), 1.00)
        self.assertAlmostEqual(normalize_value(0.50, spec), 0.00)

    # 2. Нормалізація критерію витрат
    def test_normalize_minimize(self):
        spec = {"direction": "minimize", "worst": 10.0, "best": 2.0}
        self.assertAlmostEqual(normalize_value(6.0, spec), 0.50)
        self.assertAlmostEqual(normalize_value(2.0, spec), 1.00)
        self.assertAlmostEqual(normalize_value(10.0, spec), 0.00)

    # 3. Обрізання значень поза орієнтирами (clip)
    def test_normalize_clipping(self):
        spec = {"direction": "maximize", "worst": 0.60, "best": 0.90}
        self.assertEqual(normalize_value(0.40, spec), 0.0)
        self.assertEqual(normalize_value(0.99, spec), 1.0)

    # 4. Точна межа обмеження є допустимою (inclusive)
    def test_boundary_constraints_inclusive(self):
        row = {"recall": 0.80, "latency": 10.0}
        constraints = {"recall": {"min": 0.80}, "latency": {"max": 10.0}}
        is_ok, violations = check_constraints(row, constraints)
        self.assertTrue(is_ok)
        self.assertEqual(violations, [])

    # 5. Вилучення непридатної моделі з фіксацією порушення
    def test_rejection_of_invalid_model(self):
        row = {"recall": 0.75, "latency": 12.0}
        constraints = {"recall": {"min": 0.80}, "latency": {"max": 10.0}}
        is_ok, violations = check_constraints(row, constraints)
        self.assertFalse(is_ok)
        self.assertEqual(len(violations), 2)

    # 6. Неправильна сума ваг викликає помилку
    def test_invalid_weights_sum(self):
        bad_config = copy.deepcopy(self.mock_config)
        bad_config["criteria"]["f1"]["weight"] = 0.80  # 0.80 + 0.40 = 1.20 != 1.0
        with self.assertRaises(ValueError):
            validate_config(bad_config)

    # 7. Відсутня метрика викликає помилку
    def test_missing_metric_raises_error(self):
        # Модель проходить усі обмеження (recall >= 0.80, latency <= 10.0),
        # але не містить обов'язкової для розрахунку балу метрики 'f1'
        df = pd.DataFrame([{"model_name": "M1", "recall": 0.90, "latency": 5.0}])
        with self.assertRaises(ValueError):
            rank_models(df, self.mock_config)

    # 8. Відсутність допустимих моделей
    def test_empty_eligible_models(self):
        df = pd.DataFrame([
            {"model_name": "M1", "recall": 0.70, "latency": 15.0, "f1": 0.75},  # порушує обидва
            {"model_name": "M2", "recall": 0.60, "latency": 12.0, "f1": 0.80}   # порушує обидва
        ])
        eligible, rejected = rank_models(df, self.mock_config)
        self.assertTrue(eligible.empty)
        self.assertEqual(len(rejected), 2)

    # 9. Детерміноване розв'язання нічиїх (ties resolution)
    def test_deterministic_ties_resolution(self):
        df = pd.DataFrame([
            {"model_name": "Model_B", "recall": 0.90, "latency": 6.0, "f1": 0.825},
            {"model_name": "Model_A", "recall": 0.90, "latency": 6.0, "f1": 0.825}
        ])
        eligible, _ = rank_models(df, self.mock_config)
        self.assertEqual(len(eligible), 2)
        self.assertEqual(eligible.iloc[0]["score"], eligible.iloc[1]["score"])
        # Model_A має бути першою за алфавітом
        self.assertEqual(eligible.iloc[0]["model_name"], "Model_A")
        self.assertEqual(eligible.iloc[1]["model_name"], "Model_B")

    # 10. Невідомий напрям оптимізації
    def test_unknown_direction_raises(self):
        spec = {"direction": "sideways", "worst": 0.0, "best": 1.0}
        with self.assertRaises(ValueError):
            normalize_value(0.5, spec)

    # 11. Наявність нечислових даних в обмеженнях
    def test_non_numeric_constraint_value(self):
        row = {"recall": "unknown", "latency": 5.0}
        constraints = {"recall": {"min": 0.80}}
        is_ok, violations = check_constraints(row, constraints)
        self.assertFalse(is_ok)
        self.assertIn("non-numeric value", violations[0])


if __name__ == "__main__":
    unittest.main()