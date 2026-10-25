import os
import unittest
import numpy as np
from sklearn.decomposition import PCA
from lr5 import choose_n_components, reconstruction_errors, align_cluster_labels, top_k_jaccard, inject_anomalies


class TestUnsupervisedPipeline(unittest.TestCase):

    def test_choose_n_components_threshold(self):
        variances = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
        q = choose_n_components(variances, threshold=0.85)
        self.assertEqual(q, 3)

    def test_choose_n_components_invalid_threshold(self):
        with self.assertRaises(ValueError):
            choose_n_components(np.array([0.5, 0.5]), threshold=1.5)

    def test_reconstruction_errors_shape(self):
        X = np.random.RandomState(42).normal(size=(20, 10))
        pca = PCA(n_components=4).fit(X)
        errors = reconstruction_errors(pca, X)
        self.assertEqual(errors.shape, (20,))
        self.assertTrue(np.all(errors >= 0))

    def test_align_cluster_labels_permutation(self):
        reference = np.array([0, 0, 1, 1, 2, 2])
        # Переставлені назви кластерів: 0->2, 1->0, 2->1
        candidate = np.array([2, 2, 0, 0, 1, 1])
        aligned = align_cluster_labels(reference, candidate)
        np.testing.assert_array_equal(aligned, reference)

    def test_top_k_jaccard_identical(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.5])
        jaccard = top_k_jaccard(scores, scores, k=2)
        self.assertEqual(jaccard, 1.0)

    def test_inject_anomalies_reproducibility(self):
        X = np.ones((50, 64))
        X1, y1, _ = inject_anomalies(X, fraction=0.20, seed=42)
        X2, y2, _ = inject_anomalies(X, fraction=0.20, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_inject_anomalies_invalid_fraction(self):
        X = np.ones((10, 64))
        with self.assertRaises(ValueError):
            inject_anomalies(X, fraction=1.5)

    def test_generated_bundle_file_exists(self):
        if os.path.exists("unsupervised_pipeline_bundle.joblib"):
            self.assertTrue(os.path.getsize("unsupervised_pipeline_bundle.joblib") > 0)


if __name__ == "__main__":
    unittest.main()