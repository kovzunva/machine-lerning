from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def validate_artifacts(artifacts: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Перевіряє програмний контракт завантажених артефактів."""
    errors = []
    required_keys = ["X_train", "X_test", "y_train", "y_test", "pca", "kmeans", "if_64", "if_pca"]
    for k in required_keys:
        if k not in artifacts or artifacts[k] is None:
            errors.append(f"Missing required artifact: {k}")

    if not errors:
        if artifacts["X_train"].shape[1] != 64:
            errors.append(f"X_train must have 64 features, got {artifacts['X_train'].shape[1]}")
        if artifacts["pca"].n_components_ != 22:
            errors.append(f"PCA must have 22 components, got {artifacts['pca'].n_components_}")
        if artifacts["kmeans"].n_clusters != 9:
            errors.append(f"KMeans must have 9 clusters, got {artifacts['kmeans'].n_clusters}")

    return len(errors) == 0, errors


def build_cluster_profiles(X: np.ndarray, labels: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """Обчислює стандартизовані z-score профілі кластерів відносно загальної вибірки."""
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"pixel_{i//8}_{i%8}" for i in range(n_features)]

    mean_global = np.mean(X, axis=0)
    std_global = np.std(X, axis=0)
    std_global[std_global == 0] = 1.0  # Захист від ділення на нуль

    records = []
    unique_labels = sorted(np.unique(labels))
    for k in unique_labels:
        mask = (labels == k)
        cluster_mean = np.mean(X[mask], axis=0)
        cluster_median = np.median(X[mask], axis=0)
        z_profile = (cluster_mean - mean_global) / std_global

        for j in range(n_features):
            records.append({
                "cluster": int(k),
                "feature": feature_names[j],
                "mean_raw": float(cluster_mean[j]),
                "median_raw": float(cluster_median[j]),
                "std_profile": float(z_profile[j])
            })

    return pd.DataFrame(records)


def compute_boundary_diagnostics(X: np.ndarray, centers: np.ndarray) -> pd.DataFrame:
    """Розраховує відстань до центру кластера та нормований boundary margin для кожного об'єкта."""
    # Матриця відстаней до всіх центроїдів (n_samples, n_clusters)
    dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    sorted_dists = np.sort(dists, axis=1)

    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]
    margin = (d2 - d1) / np.maximum(d2, 1e-7)

    return pd.DataFrame({
        "distance_to_center": np.round(d1, 4),
        "distance_to_second": np.round(d2, 4),
        "boundary_margin": np.round(margin, 4)
    })


def find_cluster_representatives(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    object_ids: np.ndarray,
    n_per_kind: int = 3
) -> pd.DataFrame:
    """Знаходить типові, граничні та віддалені об'єкти для кожного кластера."""
    diag = compute_boundary_diagnostics(X, centers)
    diag["cluster"] = labels
    diag["object_id"] = object_ids

    records = []
    unique_clusters = sorted(np.unique(labels))

    for k in unique_clusters:
        sub = diag[diag["cluster"] == k]

        # 1. Типові (найменша відстань до центру)
        typical = sub.sort_values(by="distance_to_center", ascending=True).head(n_per_kind)
        for _, r in typical.iterrows():
            records.append({
                "cluster": int(k), "kind": "typical", "object_id": int(r["object_id"]),
                "distance": r["distance_to_center"], "boundary_margin": r["boundary_margin"]
            })

        # 2. Граничні (найменший margin між найближчим і другим центрами)
        boundary = sub.sort_values(by="boundary_margin", ascending=True).head(n_per_kind)
        for _, r in boundary.iterrows():
            records.append({
                "cluster": int(k), "kind": "boundary", "object_id": int(r["object_id"]),
                "distance": r["distance_to_center"], "boundary_margin": r["boundary_margin"]
            })

        # 3. Віддалені (найбільша відстань до свого центру)
        distant = sub.sort_values(by="distance_to_center", ascending=False).head(n_per_kind)
        for _, r in distant.iterrows():
            records.append({
                "cluster": int(k), "kind": "distant", "object_id": int(r["object_id"]),
                "distance": r["distance_to_center"], "boundary_margin": r["boundary_margin"]
            })

    return pd.DataFrame(records)


def align_cluster_runs(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Вирівнює номери кластерів алгоритмом Угорського."""
    labels_ref = np.unique(reference)
    labels_cand = np.unique(candidate)
    k = max(len(labels_ref), len(labels_cand))

    cost_matrix = np.zeros((k, k), dtype=int)
    for i, r in enumerate(labels_ref):
        for j, c in enumerate(labels_cand):
            cost_matrix[i, j] = -np.sum((reference == r) & (candidate == c))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {labels_cand[c]: labels_ref[r] for r, c in zip(row_ind, col_ind) if c < len(labels_cand) and r < len(labels_ref)}
    return np.array([mapping.get(val, val) for val in candidate])


def calculate_assignment_stability(runs: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Обчислює частку узгоджених призначень для кожного об'єкта після вирівнювання."""
    ref = runs[0]
    aligned_runs = [ref]
    for r in runs[1:]:
        aligned_runs.append(align_cluster_runs(ref, r))

    stacked = np.vstack(aligned_runs)
    # Стійкість об'єкта = частка повторів, де його кластер збігся з першим запуском
    stability = np.mean(stacked == ref, axis=0)
    return stability, aligned_runs


def top_features_jaccard(profiles_base: pd.DataFrame, profiles_other: pd.DataFrame, top_k: int = 5) -> Dict[int, float]:
    """Обчислює коефіцієнт Жаккара між top-k найбільш відхиленими ознаками для кожного кластера."""
    jaccards = {}
    clusters = profiles_base["cluster"].unique()

    for k in sorted(clusters):
        sub_b = profiles_base[profiles_base["cluster"] == k]
        sub_o = profiles_other[profiles_other["cluster"] == k]

        top_b = set(sub_b.assign(abs_profile=sub_b["std_profile"].abs()).nlargest(top_k, "abs_profile")["feature"])
        top_o = set(sub_o.assign(abs_profile=sub_o["std_profile"].abs()).nlargest(top_k, "abs_profile")["feature"])

        inter = len(top_b.intersection(top_o))
        union = len(top_b.union(top_o))
        jaccards[int(k)] = float(inter / union) if union > 0 else 0.0

    return jaccards


def apply_action_policy(
    p_comb: np.ndarray,
    p_rec: np.ndarray,
    stability: np.ndarray,
    policy_config: Dict[str, float]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Застосовує детерміновану пріоритетну політику реагування з обов'язковим логуванням правил.
    """
    th_manual = policy_config.get("manual_review_threshold", 0.99)
    th_rec = policy_config.get("data_quality_threshold", 0.95)
    th_stab = policy_config.get("uncertain_stability_threshold", 0.80)

    actions = []
    audit_logs = []

    for i in range(len(p_comb)):
        triggered = []
        action = "no_action"

        if p_comb[i] >= th_manual:
            triggered.append(f"p_comb ({p_comb[i]:.4f}) >= {th_manual}")
        if p_rec[i] >= th_rec:
            triggered.append(f"p_rec ({p_rec[i]:.4f}) >= {th_rec}")
        if stability[i] < th_stab:
            triggered.append(f"stability ({stability[i]:.4f}) < {th_stab}")

        # Пріоритетний вибір дії
        if p_comb[i] >= th_manual:
            action = "manual_review"
        elif p_rec[i] >= th_rec:
            action = "data_quality_check"
        elif stability[i] < th_stab:
            action = "uncertain_segment"
        else:
            action = "no_action"

        actions.append(action)
        audit_logs.append({
            "object_idx": i,
            "action": action,
            "triggered_rules": triggered
        })

    return np.array(actions), audit_logs