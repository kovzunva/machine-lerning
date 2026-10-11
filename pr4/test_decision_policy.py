import unittest
from pr4 import TriageDecisionPolicy

class TestTriageDecisionPolicy(unittest.TestCase):
    def setUp(self):
        # Ініціалізація політики з нашими порогами:
        # threshold_binary=0.96 (вибраний робочий поріг)
        # threshold_low=0.35, threshold_high=0.96 (для три-зонної схеми)
        self.policy = TriageDecisionPolicy(
            threshold_binary=0.96,
            threshold_low=0.35,
            threshold_high=0.96
        )

    def test_score_below_low(self):
        """Тест: бал нижчий за поріг ручного перегляду -> плановий огляд"""
        self.assertEqual(self.policy.decide_tri_zone(0.10), "routine_screening")

    def test_score_equals_low(self):
        """Тест: бал точно дорівнює нижньому порогу -> зона консиліуму"""
        self.assertEqual(self.policy.decide_tri_zone(0.35), "advanced_ultrasound_review")

    def test_score_between_thresholds(self):
        """Тест: бал між порогами -> зона консиліуму"""
        self.assertEqual(self.policy.decide_tri_zone(0.60), "advanced_ultrasound_review")

    def test_score_equals_high(self):
        """Тест: бал точно дорівнює верхньому порогу -> термінова біопсія"""
        self.assertEqual(self.policy.decide_tri_zone(0.96), "urgent_biopsy")

    def test_score_above_high(self):
        """Тест: високий бал ризику -> термінова біопсія"""
        self.assertEqual(self.policy.decide_tri_zone(0.99), "urgent_biopsy")

    def test_binary_decision(self):
        """Тест: перевірка однопорогового правила"""
        self.assertEqual(self.policy.decide_binary(0.95), "routine_screening")
        self.assertEqual(self.policy.decide_binary(0.96), "urgent_biopsy")

    def test_out_of_bounds(self):
        """Тест: захист від некоректних значень ймовірності (<0 або >1)"""
        with self.assertRaises(ValueError):
            self.policy.decide_binary(-0.05)
        with self.assertRaises(ValueError):
            self.policy.decide_binary(1.20)

    def test_invalid_threshold_order(self):
        """Тест: помилка, якщо нижній поріг більший за верхній"""
        invalid_policy = TriageDecisionPolicy(
            threshold_binary=0.96,
            threshold_low=0.98,
            threshold_high=0.50
        )
        with self.assertRaises(ValueError):
            invalid_policy.decide_tri_zone(0.60)

    def test_batch_preserves_order(self):
        """Тест: коректність обробки списку об'єктів"""
        scores = [0.10, 0.50, 0.98]
        expected = ["routine_screening", "advanced_ultrasound_review", "urgent_biopsy"]
        actual = [self.policy.decide_tri_zone(s) for s in scores]
        self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()