import unittest
import numpy as np
import pandas as pd
from interpretation_audit import (
    validate_artifacts,
    build_cluster_profiles,
    compute_boundary_diagnostics,
    find_cluster_representatives,
    align_cluster_runs,
    calculate_assignment_stability,
    top_features_jaccard,
    apply_action_policy
)


class TestInterpretationAudit(unittest.TestCase):

    def test_validate_artifacts_missing_key(self):
        bundle = {"X_train": np.zeros((10, 64))}
        is_ok, errs = validate_artifacts(bundle)
        self.assertFalse(is_ok)
        self.assertTrue(any("Missing required artifact" in e for e in errs))

    def test_build_cluster_profiles_shape(self):
        X = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 5.0], [5.1, 5.2]])
        labels = np.array([0, 0, 1, 1])
        df_prof = build_cluster_profiles(X, labels, feature_names=["f1", "f2"])
        self.assertEqual(len(df_prof), 4)  # 2 кластери * 2 ознаки
        self.assertIn("std_profile", df_prof.columns)

    def test_compute_boundary_diagnostics_margin(self):
        # Точка (0, 0) лежить рівно між центрами (-1, 0) та (1, 0)
        X = np.array([[0.0, 0.0]])
        centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
        diag = compute_boundary_diagnostics(X, centers)
        self.assertAlmostEqual(diag.iloc[0]["distance_to_center"], 1.0)
        self.assertAlmostEqual(diag.iloc[0]["distance_to_second"], 1.0)
        self.assertAlmostEqual(diag.iloc[0]["boundary_margin"], 0.0)

    def test_find_cluster_representatives_count(self):
        X = np.random.RandomState(42).normal(size=(30, 5))
        centers = np.random.RandomState(42).normal(size=(2, 5))
        labels = np.array([0] * 15 + [1] * 15)
        ids = np.arange(30)
        reps = find_cluster_representatives(X, labels, centers, ids, n_per_kind=2)
        # 2 кластери * 3 види (typical, boundary, distant) * 2 об'єкти = 12
        self.assertEqual(len(reps), 12)

    def test_align_cluster_runs_permutation(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        cand = np.array([1, 1, 2, 2, 0, 0])  # переставлені мітки
        aligned = align_cluster_runs(ref, cand)
        np.testing.assert_array_equal(aligned, ref)

    def test_calculate_assignment_stability(self):
        runs = [
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1])  # 2-й об'єкт змінив кластер
        ]
        stab, _ = calculate_assignment_stability(runs)
        self.assertEqual(stab[0], 1.0)
        self.assertAlmostEqual(stab[1], 2/3)

    def test_top_features_jaccard_identical(self):
        prof = pd.DataFrame({
            "cluster": [0, 0],
            "feature": ["f1", "f2"],
            "std_profile": [2.0, 1.5]
        })
        j = top_features_jaccard(prof, prof, top_k=2)
        self.assertEqual(j[0], 1.0)

    def test_apply_action_policy_priority(self):
        # Якщо об'єкт підпадає і під manual (comb >= 0.99), і під data_quality (rec >= 0.95),
        # пріоритет має manual_review
        p_comb = np.array([0.995, 0.50])
        p_rec = np.array([0.980, 0.96])
        stab = np.array([0.90, 0.90])
        cfg = {"manual_review_threshold": 0.99, "data_quality_threshold": 0.95, "uncertain_stability_threshold": 0.80}
        actions, _ = apply_action_policy(p_comb, p_rec, stab, cfg)
        self.assertEqual(actions[0], "manual_review")
        self.assertEqual(actions[1], "data_quality_check")

    def test_apply_action_policy_uncertain(self):
        p_comb = np.array([0.20])
        p_rec = np.array([0.20])
        stab = np.array([0.60])  # нижче 0.80
        cfg = {"manual_review_threshold": 0.99, "data_quality_threshold": 0.95, "uncertain_stability_threshold": 0.80}
        actions, _ = apply_action_policy(p_comb, p_rec, stab, cfg)
        self.assertEqual(actions[0], "uncertain_segment")

    def test_apply_action_policy_no_action(self):
        p_comb = np.array([0.10])
        p_rec = np.array([0.10])
        stab = np.array([0.95])
        cfg = {"manual_review_threshold": 0.99, "data_quality_threshold": 0.95, "uncertain_stability_threshold": 0.80}
        actions, _ = apply_action_policy(p_comb, p_rec, stab, cfg)
        self.assertEqual(actions[0], "no_action")


if __name__ == "__main__":
    unittest.main()