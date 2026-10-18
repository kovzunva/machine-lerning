import unittest
from pr4 import TriageDecisionPolicy

class TestTriageDecisionPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = TriageDecisionPolicy(
            threshold_binary=0.96,
            threshold_low=0.20,
            threshold_high=0.70,
            capacity_quota=0.40
        )

    def test_score_below_low(self):
        self.assertEqual(self.policy.decide_tri_zone(0.10), "routine_screening")

    def test_score_equals_low(self):
        self.assertEqual(self.policy.decide_tri_zone(0.20), "advanced_ultrasound_review")

    def test_score_between_thresholds(self):
        self.assertEqual(self.policy.decide_tri_zone(0.50), "advanced_ultrasound_review")

    def test_score_equals_high(self):
        self.assertEqual(self.policy.decide_tri_zone(0.70), "urgent_biopsy")

    def test_score_above_high(self):
        self.assertEqual(self.policy.decide_tri_zone(0.85), "urgent_biopsy")

    def test_binary_decision_boundary(self):
        self.assertEqual(self.policy.decide_binary(0.95), "routine_screening")
        self.assertEqual(self.policy.decide_binary(0.96), "urgent_biopsy")

    def test_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.policy.decide_binary(-0.01)
        with self.assertRaises(ValueError):
            self.policy.decide_binary(1.05)

    def test_invalid_threshold_order(self):
        bad_policy = TriageDecisionPolicy(threshold_binary=0.5, threshold_low=0.8, threshold_high=0.4)
        with self.assertRaises(ValueError):
            bad_policy.decide_tri_zone(0.5)

    def test_top_k_empty_batch(self):
        self.assertEqual(self.policy.decide_top_k([]), [])

    def test_top_k_quota_strictness(self):
        # 10 об'єктів при квоті 40% -> рівно 4 направлення
        scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.95]
        actions = self.policy.decide_top_k(scores)
        self.assertEqual(actions.count("urgent_biopsy"), 4)

    def test_top_k_ties_resolution(self):
        # Перевірка детермінованого розв'язання однакових балів (ties)
        scores = [0.8, 0.8, 0.8, 0.1, 0.1]
        # При max_k=2 мають бути взяті перші два об'єкти з балом 0.8
        actions = self.policy.decide_top_k(scores, max_k=2)
        self.assertEqual(actions, ["urgent_biopsy", "urgent_biopsy", "routine_screening", "routine_screening", "routine_screening"])

if __name__ == "__main__":
    unittest.main()