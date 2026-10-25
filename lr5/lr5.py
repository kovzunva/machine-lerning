import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# =====================================================================
# 1. ЧИСТІ ФУНКЦІЇ ДЛЯ ТЕСТУВАННЯ ТА ОБЧИСЛЕНЬ
# =====================================================================

def choose_n_components(explained_variance_ratio: np.ndarray, threshold: float = 0.90) -> int:
    """Знаходить мінімальну кількість компонент для заданого порогу дисперсії."""
    if not (0.0 < threshold <= 1.0):
        raise ValueError("Порогове значення дисперсії має належати інтервалу (0, 1].")
    cum_var = np.cumsum(explained_variance_ratio)
    idx = np.where(cum_var >= threshold)[0]
    if len(idx) == 0:
        return len(explained_variance_ratio)
    return int(idx[0] + 1)


def reconstruction_errors(pca: PCA, X_scaled: np.ndarray) -> np.ndarray:
    """Обчислює середньоквадратичну похибку відновлення (MSE) кожного об'єкта."""
    Z = pca.transform(X_scaled)
    X_rec = pca.inverse_transform(Z)
    return np.mean((X_scaled - X_rec) ** 2, axis=1)


def align_cluster_labels(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """
    Вирівнює номери кластерів candidate відносно reference за допомогою
    алгоритму Угорського (linear_sum_assignment) для максимального збігу.
    """
    labels_ref = np.unique(reference)
    labels_cand = np.unique(candidate)
    k = max(len(labels_ref), len(labels_cand))

    cost_matrix = np.zeros((k, k), dtype=int)
    for i, r in enumerate(labels_ref):
        for j, c in enumerate(labels_cand):
            cost_matrix[i, j] = -np.sum((reference == r) & (candidate == c))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {labels_cand[c]: labels_ref[r] for r, c in zip(row_ind, col_ind) if c < len(labels_cand) and r < len(labels_ref)}

    aligned = np.array([mapping.get(val, val) for val in candidate])
    return aligned


def top_k_jaccard(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Обчислює коефіцієнт Жаккара між списками top-k об'єктів за двома оцінками."""
    if k <= 0:
        raise ValueError("k має бути більше нуля.")
    top_a = set(np.argsort(scores_a)[-k:])
    top_b = set(np.argsort(scores_b)[-k:])
    inter = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))
    return float(inter / union) if union > 0 else 0.0


def inject_anomalies(X_test: np.ndarray, fraction: float = 0.18, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Контрольовано створює аномалії на тесті: додавання шуму,
    інверсія інтенсивностей та просторова перестановка квадрантів.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError("Частка аномалій має бути в межах (0, 1).")

    rng = np.random.RandomState(seed)
    n_samples = X_test.shape[0]
    n_anom = int(n_samples * fraction)
    anom_indices = rng.choice(n_samples, size=n_anom, replace=False)

    X_corrupted = X_test.copy()
    y_anom = np.zeros(n_samples, dtype=int)
    y_anom[anom_indices] = 1

    sub_groups = np.array_split(anom_indices, 3)
    
    # 1. Гауссів шум
    noise = rng.normal(loc=0.5, scale=0.3, size=X_corrupted[sub_groups[0]].shape)
    X_corrupted[sub_groups[0]] = np.clip(X_corrupted[sub_groups[0]] + noise, 0.0, 1.0)

    # 2. Інверсія пікселів
    X_corrupted[sub_groups[1]] = 1.0 - X_corrupted[sub_groups[1]]

    # 3. Перестановка квадрантів (для 8x8)
    for idx in sub_groups[2]:
        img = X_corrupted[idx].reshape(8, 8)
        top_left = img[:4, :4].copy()
        bottom_right = img[4:, 4:].copy()
        img[:4, :4] = bottom_right
        img[4:, 4:] = top_left
        X_corrupted[idx] = img.ravel()

    meta = {
        "n_anomalies": n_anom,
        "indices": anom_indices.tolist(),
        "kinds": {"noise": len(sub_groups[0]), "inversion": len(sub_groups[1]), "quadrant_swap": len(sub_groups[2])}
    }
    return X_corrupted, y_anom, meta


# =====================================================================
# 2. ГОЛОВНИЙ ЕКСПЕРИМЕНТАЛЬНИЙ КОНВЕЄР
# =====================================================================

def main():
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №5: КЛАСТЕРИЗАЦІЯ, PCA ТА ДЕТЕКЦІЯ АНОМАЛІЙ")
    print("=" * 80)

    cfg_path = os.path.join("configs", "experiment_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "random_state": 42,
            "test_size": 0.30,
            "pca_variance_threshold": 0.90,
            "k_range": list(range(6, 15)),
            "clustering_seeds": [11, 23, 42, 67, 101],
            "anomaly_fraction": 0.18,
            "anomaly_threshold_quantile": 0.95
        }

    seed = cfg["random_state"]
    digits = load_digits()
    X_raw = digits.data
    y_true_all = digits.target

    # Розподіл Train / Test
    X_train_raw, X_test_raw, y_train_labels, y_test_labels = train_test_split(
        X_raw, y_true_all, test_size=cfg["test_size"], random_state=seed, stratify=y_true_all
    )
    print(f"Дані зафіксовано: Train = {X_train_raw.shape[0]} об'єктів, Test = {X_test_raw.shape[0]} об'єктів (64 ознаки).")

    # Масштабування строго на Train
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 1. PCA та вибір робочої розмірності
    print("\n[КРОК 1] Діагностика PCA та вибір розмірності q...")
    pca_full = PCA(random_state=seed)
    pca_full.fit(X_train_scaled)
    
    q = choose_n_components(pca_full.explained_variance_ratio_, cfg["pca_variance_threshold"])
    explained_q = np.sum(pca_full.explained_variance_ratio_[:q])
    print(f"  -> Поріг {cfg['pca_variance_threshold']*100:.0f}% дисперсії досягнуто при q = {q} компонентах (фактично: {explained_q*100:.2f}%).")

    pca_q = PCA(n_components=q, random_state=seed)
    Z_train = pca_q.fit_transform(X_train_scaled)
    Z_test = pca_q.transform(X_test_scaled)

    # Збереження похибок відновлення
    train_rec_errors = reconstruction_errors(pca_q, X_train_scaled)
    rec_error_threshold = float(np.percentile(train_rec_errors, cfg["anomaly_threshold_quantile"] * 100))
    print(f"  -> Навчальна похибка реконструкції: mean = {np.mean(train_rec_errors):.5f}, поріг (95% quantile) = {rec_error_threshold:.5f}")

    # Таблиця дисперсії
    df_var = pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance_ratio": np.round(pca_full.explained_variance_ratio_, 5),
        "cumulative_variance": np.round(np.cumsum(pca_full.explained_variance_ratio_), 5)
    })
    df_var.to_csv("pca_variance_spectrum.csv", index=False)

    # 2. Порівняльна кластеризація K-Means у двох просторах
    print("\n[КРОК 2] Кластеризація K-Means у 64-вимірному та PCA-просторах...")
    cluster_records = []
    runs_64 = {k: [] for k in cfg["k_range"]}
    runs_pca = {k: [] for k in cfg["k_range"]}

    for k in cfg["k_range"]:
        for s in cfg["clustering_seeds"]:
            # 64 ознаки
            t0 = time.perf_counter()
            km_64 = KMeans(n_clusters=k, random_state=s, n_init=10).fit(X_train_scaled)
            t_64 = time.perf_counter() - t0
            runs_64[k].append(km_64.labels_)

            sil_64 = silhouette_score(X_train_scaled, km_64.labels_)
            db_64 = davies_bouldin_score(X_train_scaled, km_64.labels_)
            ch_64 = calinski_harabasz_score(X_train_scaled, km_64.labels_)

            # PCA простір
            t0 = time.perf_counter()
            km_pca = KMeans(n_clusters=k, random_state=s, n_init=10).fit(Z_train)
            t_pca = time.perf_counter() - t0
            runs_pca[k].append(km_pca.labels_)

            sil_pca = silhouette_score(Z_train, km_pca.labels_)
            db_pca = davies_bouldin_score(Z_train, km_pca.labels_)
            ch_pca = calinski_harabasz_score(Z_train, km_pca.labels_)

            cluster_records.append({
                "k": k, "seed": s,
                "sil_64": sil_64, "db_64": db_64, "ch_64": ch_64, "time_64": t_64,
                "sil_pca": sil_pca, "db_pca": db_pca, "ch_pca": ch_pca, "time_pca": t_pca
            })

    df_cluster_all = pd.DataFrame(cluster_records)
    df_cluster_all.to_csv("clustering_runs_raw.csv", index=False)

    # Зведені характеристики та розрахунок стабільності (ARI між seeds)
    summary_cluster = []
    for k in cfg["k_range"]:
        sub = df_cluster_all[df_cluster_all["k"] == k]
        
        # Попарний ARI між повторами
        ari_list_64 = []
        ari_list_pca = []
        for i in range(len(cfg["clustering_seeds"])):
            for j in range(i + 1, len(cfg["clustering_seeds"])):
                ari_list_64.append(adjusted_rand_score(runs_64[k][i], runs_64[k][j]))
                ari_list_pca.append(adjusted_rand_score(runs_pca[k][i], runs_pca[k][j]))

        summary_cluster.append({
            "k": k,
            "sil_64_mean": round(sub["sil_64"].mean(), 4),
            "db_64_mean": round(sub["db_64"].mean(), 4),
            "ch_64_mean": round(sub["ch_64"].mean(), 1),
            "ari_64_stability": round(np.mean(ari_list_64), 4),
            "time_64_mean": round(sub["time_64"].mean(), 3),
            "sil_pca_mean": round(sub["sil_pca"].mean(), 4),
            "db_pca_mean": round(sub["db_pca"].mean(), 4),
            "ch_pca_mean": round(sub["ch_pca"].mean(), 1),
            "ari_pca_stability": round(np.mean(ari_list_pca), 4),
            "time_pca_mean": round(sub["time_pca"].mean(), 3),
        })

    df_cluster_summary = pd.DataFrame(summary_cluster)
    df_cluster_summary.to_csv("clustering_summary.csv", index=False)

    # Вибір оптимального K за максимумом Silhouette
    best_k_64 = int(df_cluster_summary.loc[df_cluster_summary["sil_64_mean"].idxmax()]["k"])
    best_k_pca = int(df_cluster_summary.loc[df_cluster_summary["sil_pca_mean"].idxmax()]["k"])
    print(f"  -> Оптимальне K за Silhouette: 64 ознаки -> K={best_k_64}; PCA -> K={best_k_pca}")

    # Фінальні моделі K-Means для обраного K=10 (або найкращого)
    chosen_k = best_k_pca
    km_final_64 = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X_train_scaled)
    km_final_pca = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(Z_train)

    # Стійкість призначення конкретних об'єктів для PCA
    ref_labels = runs_pca[chosen_k][0]
    aligned_runs = [align_cluster_labels(ref_labels, r) for r in runs_pca[chosen_k]]
    obj_stability = np.mean([aligned_runs[i] == aligned_runs[0] for i in range(len(aligned_runs))], axis=0)
    unstable_indices = np.argsort(obj_stability)[:15]

    df_unstable = pd.DataFrame({
        "object_idx": unstable_indices,
        "stability_rate": np.round(obj_stability[unstable_indices], 3),
        "true_digit": y_train_labels[unstable_indices]
    })
    df_unstable.to_csv("unstable_objects.csv", index=False)

    # Зовнішній аудит за мітками
    ext_audit = [
        {
            "space": "64 ознаки (train)",
            "ari": round(adjusted_rand_score(y_train_labels, km_final_64.labels_), 4),
            "nmi": round(normalized_mutual_info_score(y_train_labels, km_final_64.labels_), 4),
            "homogeneity": round(homogeneity_score(y_train_labels, km_final_64.labels_), 4),
            "completeness": round(completeness_score(y_train_labels, km_final_64.labels_), 4),
        },
        {
            "space": f"PCA {q} (train)",
            "ari": round(adjusted_rand_score(y_train_labels, km_final_pca.labels_), 4),
            "nmi": round(normalized_mutual_info_score(y_train_labels, km_final_pca.labels_), 4),
            "homogeneity": round(homogeneity_score(y_train_labels, km_final_pca.labels_), 4),
            "completeness": round(completeness_score(y_train_labels, km_final_pca.labels_), 4),
        },
        {
            "space": f"PCA {q} (test)",
            "ari": round(adjusted_rand_score(y_test_labels, km_final_pca.predict(Z_test)), 4),
            "nmi": round(normalized_mutual_info_score(y_test_labels, km_final_pca.predict(Z_test)), 4),
            "homogeneity": round(homogeneity_score(y_test_labels, km_final_pca.predict(Z_test)), 4),
            "completeness": round(completeness_score(y_test_labels, km_final_pca.predict(Z_test)), 4),
        }
    ]
    df_ext = pd.DataFrame(ext_audit)
    df_ext.to_csv("external_audit.csv", index=False)

    # 3. Виявлення аномалій
    print("\n[КРОК 3] Ін'єкція контрольованих аномалій та оцінка детекторів...")
    X_test_anom, y_test_anom, anom_meta = inject_anomalies(
        X_test_scaled, fraction=cfg["anomaly_fraction"], seed=cfg["random_state"]
    )
    Z_test_anom = pca_q.transform(X_test_anom)

    # Isolation Forest
    if_64 = IsolationForest(**cfg["isolation_forest"]).fit(X_train_scaled)
    if_pca = IsolationForest(**cfg["isolation_forest"]).fit(Z_train)

    score_train_if_64 = -if_64.score_samples(X_train_scaled)
    score_train_if_pca = -if_pca.score_samples(Z_train)
    thresh_if_64 = float(np.percentile(score_train_if_64, cfg["anomaly_threshold_quantile"] * 100))
    thresh_if_pca = float(np.percentile(score_train_if_pca, cfg["anomaly_threshold_quantile"] * 100))

    # Тестові оцінки
    score_test_if_64 = -if_64.score_samples(X_test_anom)
    score_test_if_pca = -if_pca.score_samples(Z_test_anom)
    score_test_rec = reconstruction_errors(pca_q, X_test_anom)

    # Нормалізація процентилями для об'єднаного score
    p_if = np.searchsorted(np.sort(score_train_if_pca), score_test_if_pca) / len(score_train_if_pca)
    p_rec = np.searchsorted(np.sort(train_rec_errors), score_test_rec) / len(train_rec_errors)
    score_test_comb = np.maximum(p_if, p_rec)
    thresh_comb = cfg["anomaly_threshold_quantile"]

    # Розрахунок метрик виявлення аномалій
    detectors = {
        "IF: 64 ознаки": (score_test_if_64, thresh_if_64),
        "IF: PCA": (score_test_if_pca, thresh_if_pca),
        "Похибка PCA": (score_test_rec, rec_error_threshold),
        "Об'єднаний score": (score_test_comb, thresh_comb),
    }

    k_anom = int(np.sum(y_test_anom))
    anom_eval = []
    for name, (scores, th) in detectors.items():
        y_pred = (scores >= th).astype(int)
        roc = roc_auc_score(y_test_anom, scores)
        ap = average_precision_score(y_test_anom, scores)
        prec = precision_score(y_test_anom, y_pred, zero_division=0)
        rec = recall_score(y_test_anom, y_pred, zero_division=0)
        f1 = f1_score(y_test_anom, y_pred, zero_division=0)
        top_k_indices = np.argsort(scores)[-k_anom:]
        prec_at_k = np.mean(y_test_anom[top_k_indices])
        tn, fp, fn, tp = confusion_matrix(y_test_anom, y_pred, labels=[0, 1]).ravel()

        anom_eval.append({
            "Метод": name,
            "ROC-AUC": round(roc, 4),
            "AP": round(ap, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "Precision@k": round(prec_at_k, 4),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
        })

    df_anom_eval = pd.DataFrame(anom_eval)
    df_anom_eval.to_csv("anomaly_detection_metrics.csv", index=False)

    jaccard_if = top_k_jaccard(score_test_if_64, score_test_if_pca, k_anom)
    print(f"  -> Jaccard між top-k аномалій IF (64) та IF (PCA): {jaccard_if:.4f}")

    # 4. Генерація графіків
    print("\n[КРОК 4] Генерація графічних артефактів...")
    
    # Рис 1: Дисперсія PCA
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(np.arange(1, len(pca_full.explained_variance_ratio_) + 1), np.cumsum(pca_full.explained_variance_ratio_), lw=2, color="#2b5c8f")
    ax[0].axhline(0.90, color="gray", linestyle="--", label="Поріг 90%")
    ax[0].axvline(q, color="#d95f02", linestyle=":", label=f"q = {q}")
    ax[0].set_xlabel("Кількість компонент")
    ax[0].set_ylabel("Кумулятивна пояснена дисперсія")
    ax[0].set_title("Вибір робочої розмірності PCA")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    components_test = [2, 5, 10, 15, 20, q, 30, 40, 50, 64]
    components_test = sorted(list(set(components_test)))
    rec_errors_comp = []
    for c in components_test:
        p_c = PCA(n_components=c, random_state=seed).fit(X_train_scaled)
        rec_errors_comp.append(np.mean(reconstruction_errors(p_c, X_train_scaled)))
    ax[1].plot(components_test, rec_errors_comp, "o-", color="#d95f02", lw=2)
    ax[1].axvline(q, color="gray", linestyle=":", label=f"q = {q}")
    ax[1].set_xlabel("Кількість компонент")
    ax[1].set_ylabel("Середня похибка відновлення (MSE)")
    ax[1].set_title("Похибка відновлення від розмірності")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("lr5_fig1_pca_diagnostics.png", dpi=300)
    plt.close()

    # Рис 2: Перші 6 компонент як 8х8 образи
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(pca_q.components_[i].reshape(8, 8), cmap="coolwarm")
        ax.set_title(f"PC{i+1}")
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Вага піксельної ознаки")
    plt.suptitle("Перші шість головних компонент як просторові шаблони 8x8")
    plt.savefig("lr5_fig2_pca_components_weights.png", dpi=300)
    plt.close()

    # Рис 3: Внутрішні метрики кластеризації
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    ax[0, 0].plot(df_cluster_summary["k"], df_cluster_summary["sil_64_mean"], "o-", label="64 ознаки", color="#2b5c8f")
    ax[0, 0].plot(df_cluster_summary["k"], df_cluster_summary["sil_pca_mean"], "s-", label=f"PCA ({q})", color="#d95f02")
    ax[0, 0].set_ylabel("Silhouette (більше — краще)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()

    ax[0, 1].plot(df_cluster_summary["k"], df_cluster_summary["db_64_mean"], "o-", label="64 ознаки", color="#2b5c8f")
    ax[0, 1].plot(df_cluster_summary["k"], df_cluster_summary["db_pca_mean"], "s-", label=f"PCA ({q})", color="#d95f02")
    ax[0, 1].set_ylabel("Davies-Bouldin (менше — краще)")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(df_cluster_summary["k"], df_cluster_summary["ch_64_mean"], "o-", label="64 ознаки", color="#2b5c8f")
    ax[1, 0].plot(df_cluster_summary["k"], df_cluster_summary["ch_pca_mean"], "s-", label=f"PCA ({q})", color="#d95f02")
    ax[1, 0].set_ylabel("Calinski-Harabasz (більше — краще)")
    ax[1, 0].set_xlabel("Кількість кластерів K")
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(df_cluster_summary["k"], df_cluster_summary["time_64_mean"], "o-", label="64 ознаки", color="#2b5c8f")
    ax[1, 1].plot(df_cluster_summary["k"], df_cluster_summary["time_pca_mean"], "s-", label=f"PCA ({q})", color="#d95f02")
    ax[1, 1].set_ylabel("Середній час fit, с")
    ax[1, 1].set_xlabel("Кількість кластерів K")
    ax[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr5_fig3_clustering_metrics.png", dpi=300)
    plt.close()

    # Рис 4: Стійкість кластеризації (ARI між seeds)
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_cluster_summary["k"], df_cluster_summary["ari_64_stability"], "o-", label="64 ознаки", color="#2b5c8f", lw=2)
    plt.plot(df_cluster_summary["k"], df_cluster_summary["ari_pca_stability"], "s-", label=f"PCA ({q})", color="#d95f02", lw=2)
    plt.xlabel("Кількість кластерів K")
    plt.ylabel("Середній попарний ARI між seeds")
    plt.title("Стійкість K-Means до випадкової ініціалізації")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lr5_fig4_cluster_stability.png", dpi=300)
    plt.close()

    # Рис 5: 2D-проєкція кластерів та аномалій
    pca_2d = PCA(n_components=2, random_state=seed)
    Z_train_2d = pca_2d.fit_transform(X_train_scaled)
    Z_test_2d = pca_2d.transform(X_test_anom)
    centers_scaled = pca_q.inverse_transform(km_final_pca.cluster_centers_)
    centers_2d = pca_2d.transform(centers_scaled)

    plt.figure(figsize=(8.5, 6))
    plt.scatter(Z_train_2d[:, 0], Z_train_2d[:, 1], c=km_final_pca.labels_, cmap="tab10", alpha=0.35, s=25, label="Train об'єкти")
    anom_mask = y_test_anom == 1
    plt.scatter(Z_test_2d[anom_mask, 0], Z_test_2d[anom_mask, 1], c="red", marker="x", s=50, lw=1.5, label="Контрольні аномалії")
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="P", s=140, edgecolor="yellow", lw=1.5, label="Центроїди")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Кластери та аномалії у 2D-проєкції (пояснена дисперсія: {np.sum(pca_full.explained_variance_ratio_[:2])*100:.1f}%)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr5_fig5_clusters_2d_projection.png", dpi=300)
    plt.close()

    # Рис 6: Порівняння детекції аномалій
    plt.figure(figsize=(9, 4.5))
    x_pos = np.arange(len(df_anom_eval))
    width = 0.25
    plt.bar(x_pos - width, df_anom_eval["ROC-AUC"], width, label="ROC-AUC", color="#2b5c8f")
    plt.bar(x_pos, df_anom_eval["AP"], width, label="Average Precision", color="#d95f02")
    plt.bar(x_pos + width, df_anom_eval["Recall"], width, label="Recall за train-порогом", color="#2ca02c")
    plt.xticks(x_pos, df_anom_eval["Метод"])
    plt.ylabel("Значення метрики")
    plt.title("Виявлення контрольних аномалій різними підходами")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("lr5_fig6_anomaly_detection_benchmark.png", dpi=300)
    plt.close()

    # Збереження цілісного артефакту
    bundle = {
        "scaler": scaler,
        "pca_q": pca_q,
        "q_components": q,
        "kmeans_64": km_final_64,
        "kmeans_pca": km_final_pca,
        "if_64": if_64,
        "if_pca": if_pca,
        "thresholds": {
            "rec_error": rec_error_threshold,
            "if_64": thresh_if_64,
            "if_pca": thresh_if_pca,
            "comb": thresh_comb
        }
    }
    joblib.dump(bundle, "unsupervised_pipeline_bundle.joblib", compress=3)
    print("Всі артефакти та графіки успішно створено.")

    # Вивід підсумків у консоль
    print("\n" + "=" * 60)
    print("ФАКТИЧНІ РЕЗУЛЬТАТИ ДЛЯ ЗВІТУ (LR5)")
    print("=" * 60)
    print(f"Розмірність q: {q} (дисперсія: {explained_q*100:.2f}%)")
    print(f"Похибка реконструкції: train mean = {np.mean(train_rec_errors):.5f}, поріг = {rec_error_threshold:.5f}")
    print("\nЗведена кластеризація (K-Means):")
    print(df_cluster_summary[["k", "sil_64_mean", "sil_pca_mean", "ari_64_stability", "ari_pca_stability", "time_64_mean", "time_pca_mean"]].to_string(index=False))
    print("\nЗовнішній аудит (ARI, NMI, Homogeneity, Completeness):")
    print(df_ext.to_string(index=False))
    print("\nДетекція аномалій на тестовій вибірці:")
    print(df_anom_eval.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()