import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from interpretation_audit import (
    validate_artifacts,
    build_cluster_profiles,
    compute_boundary_diagnostics,
    find_cluster_representatives,
    calculate_assignment_stability,
    top_features_jaccard,
    apply_action_policy
)

# Допоміжна функція генерації Markdown-таблиць без сторонньої залежності tabulate
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Немає даних_\n"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"
    ]
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(row[h]) for h in headers) + " |"
        lines.append(line)
    return "\n".join(lines)


def inject_anomalies_reproducible(X_test: np.ndarray, fraction: float = 0.18, seed: int = 42):
    rng = np.random.RandomState(seed)
    n_samples = X_test.shape[0]
    n_anom = int(n_samples * fraction)
    anom_indices = rng.choice(n_samples, size=n_anom, replace=False)

    X_corrupted = X_test.copy()
    y_anom = np.zeros(n_samples, dtype=int)
    y_anom[anom_indices] = 1

    sub_groups = np.array_split(anom_indices, 3)
    # Шум
    noise = rng.normal(loc=0.5, scale=0.3, size=X_corrupted[sub_groups[0]].shape)
    X_corrupted[sub_groups[0]] = np.clip(X_corrupted[sub_groups[0]] + noise, 0.0, 1.0)
    # Інверсія
    X_corrupted[sub_groups[1]] = 1.0 - X_corrupted[sub_groups[1]]
    # Перестановка квадрантів
    for idx in sub_groups[2]:
        img = X_corrupted[idx].reshape(8, 8)
        tl = img[:4, :4].copy()
        br = img[4:, 4:].copy()
        img[:4, :4] = br
        img[4:, 4:] = tl
        X_corrupted[idx] = img.ravel()

    return X_corrupted, y_anom, anom_indices


def main():
    print("=" * 80)
    print("ПРАКТИЧНА РОБОТА №6: ІНТЕРПРЕТАЦІЯ ТА АУДИТ НАВЧАННЯ БЕЗ УЧИТЕЛЯ")
    print("=" * 80)

    cfg_path = os.path.join("configs", "interpretation_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "random_state": 42, "test_size": 0.30, "pca_q": 22, "k_clusters": 9,
            "clustering_seeds": [11, 23, 42, 67, 101], "boundary_margin_threshold": 0.05,
            "object_stability_threshold": 0.80, "top_features_k": 5,
            "action_policy": {
                "manual_review_threshold": 0.99, "data_quality_threshold": 0.95,
                "uncertain_stability_threshold": 0.80
            }
        }

    seed = cfg["random_state"]
    digits = load_digits()
    X_raw = digits.data
    y_labels = digits.target
    feature_names = [f"pixel_{i//8}_{i%8}" for i in range(64)]

    # 1. Відтворення спліту ЛР5
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_labels, test_size=cfg["test_size"], random_state=seed, stratify=y_labels
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test_clean = scaler.transform(X_test_raw)

    pca = PCA(n_components=cfg["pca_q"], random_state=seed)
    Z_train = pca.fit_transform(X_train)
    Z_test_clean = pca.transform(X_test_clean)

    # Кластеризація (5 повторів)
    k = cfg["k_clusters"]
    runs = []
    kmeans_models = []
    for s in cfg["clustering_seeds"]:
        km = KMeans(n_clusters=k, random_state=s, n_init=10).fit(Z_train)
        runs.append(km.labels_)
        kmeans_models.append(km)

    km_base = kmeans_models[0]
    base_labels = runs[0]
    base_centers = km_base.cluster_centers_

    # Детектори аномалій
    if_64 = IsolationForest(n_estimators=150, random_state=seed).fit(X_train)
    if_pca = IsolationForest(n_estimators=150, random_state=seed).fit(Z_train)

    train_rec_errors = np.mean((X_train - pca.inverse_transform(Z_train)) ** 2, axis=1)

    # Перевірка контракту артефактів
    bundle = {
        "X_train": X_train, "X_test": X_test_clean, "y_train": y_train, "y_test": y_test,
        "pca": pca, "kmeans": km_base, "if_64": if_64, "if_pca": if_pca
    }
    is_valid, val_errors = validate_artifacts(bundle)
    if not is_valid:
        print("Помилка контракту артефактів:", val_errors)
        sys.exit(1)
    print("Контракт артефактів успішно верифіковано.")

    # 2. Розрахунок профілів та стійкості
    print("\n[КРОК 1] Розрахунок профілів кластерів та стійкості призначення...")
    stability, aligned_runs = calculate_assignment_stability(runs)
    df_profiles_base = build_cluster_profiles(X_train, base_labels, feature_names)
    df_profiles_base.to_csv("cluster_profiles_long.csv", index=False)

    # Стійкість top-5 ознак відносно повторних запусків
    jaccard_per_run = []
    for r_labels in aligned_runs[1:]:
        df_prof_other = build_cluster_profiles(X_train, r_labels, feature_names)
        j_dict = top_features_jaccard(df_profiles_base, df_prof_other, cfg["top_features_k"])
        jaccard_per_run.append(j_dict)

    df_jaccard = pd.DataFrame(jaccard_per_run)
    mean_jaccard_per_cluster = df_jaccard.mean(axis=0).to_dict()

    # Зведена таблиця кластерів (Таблиця 1)
    diag_train = compute_boundary_diagnostics(Z_train, base_centers)
    diag_train["cluster"] = base_labels
    diag_train["stability"] = stability

    summary_cluster_records = []
    for c_id in range(k):
        sub = diag_train[diag_train["cluster"] == c_id]
        n_objs = len(sub)
        n_low_margin = (sub["boundary_margin"] < cfg["boundary_margin_threshold"]).sum()
        n_unstable = (sub["stability"] < cfg["object_stability_threshold"]).sum()
        mean_stab = sub["stability"].mean()
        j_top5 = mean_jaccard_per_cluster.get(c_id, 1.0)
        mean_dist = sub["distance_to_center"].mean()

        summary_cluster_records.append({
            "cluster": c_id,
            "size": n_objs,
            "margin_lt_005": int(n_low_margin),
            "unstable_objs": int(n_unstable),
            "mean_stability": round(float(mean_stab), 4),
            "jaccard_top5": round(float(j_top5), 4),
            "mean_distance": round(float(mean_dist), 4)
        })

    df_cluster_summary = pd.DataFrame(summary_cluster_records)
    df_cluster_summary.to_csv("cluster_summary_table.csv", index=False)
    print("\nТаблиця 1: Зведена характеристика та стійкість кластерів:")
    print(df_cluster_summary.to_string(index=False))

    # 3. Представники кластерів
    object_ids_train = np.arange(len(X_train))
    df_reps = find_cluster_representatives(Z_train, base_labels, base_centers, object_ids_train, n_per_kind=3)
    df_reps.to_csv("cluster_representatives.csv", index=False)
    print(f"\nЗбережено представників кластерів ({len(df_reps)} записів).")

    # 4. Аналіз тестових аномалій та правил реагування
    print("\n[КРОК 2] Аудит детекції аномалій та виконання правил реагування...")
    X_test_corrupted, y_test_anom, anom_indices = inject_anomalies_reproducible(
        X_test_clean, fraction=0.18, seed=seed
    )
    Z_test_corrupted = pca.transform(X_test_corrupted)

    score_train_if_pca = -if_pca.score_samples(Z_train)
    score_test_if_pca = -if_pca.score_samples(Z_test_corrupted)
    score_test_rec = np.mean((X_test_corrupted - pca.inverse_transform(Z_test_corrupted)) ** 2, axis=1)

    # Емпіричні процентилі
    sorted_train_if = np.sort(score_train_if_pca)
    sorted_train_rec = np.sort(train_rec_errors)

    p_if_test = np.searchsorted(sorted_train_if, score_test_if_pca) / len(sorted_train_if)
    p_rec_test = np.searchsorted(sorted_train_rec, score_test_rec) / len(sorted_train_rec)
    p_comb_test = np.maximum(p_if_test, p_rec_test)

    # Кластеризація та діагностика на тесті
    test_cluster_labels = km_base.predict(Z_test_corrupted)
    diag_test = compute_boundary_diagnostics(Z_test_corrupted, base_centers)

    # Оцінка стійкості на тесті (через наближення за маргіном)
    test_stability = np.clip(1.0 - (cfg["boundary_margin_threshold"] - diag_test["boundary_margin"]).clip(lower=0) * 4, 0.5, 1.0)

    # Застосування політики реагування
    actions, audit_logs = apply_action_policy(
        p_comb_test, p_rec_test, test_stability.values, cfg["action_policy"]
    )

    df_test_results = pd.DataFrame({
        "object_idx": np.arange(len(X_test_corrupted)),
        "is_controlled_anomaly": y_test_anom,
        "assigned_cluster": test_cluster_labels,
        "boundary_margin": diag_test["boundary_margin"],
        "percentile_if_pca": np.round(p_if_test, 4),
        "percentile_reconstruction": np.round(p_rec_test, 4),
        "percentile_combined": np.round(p_comb_test, 4),
        "action": actions
    })
    df_test_results.to_csv("test_triage_audit.csv", index=False)

    # Таблиця наслідків правил (Таблиця 3)
    policy_summary_records = []
    actions_order = ["manual_review", "data_quality_check", "uncertain_segment", "no_action"]
    for act in actions_order:
        sub = df_test_results[df_test_results["action"] == act]
        n_clean = int((sub["is_controlled_anomaly"] == 0).sum())
        n_anom = int((sub["is_controlled_anomaly"] == 1).sum())
        policy_summary_records.append({
            "action": act,
            "clean_samples": n_clean,
            "controlled_anomalies": n_anom,
            "total_samples": len(sub)
        })

    df_policy_summary = pd.DataFrame(policy_summary_records)
    df_policy_summary.to_csv("policy_evaluation_summary.csv", index=False)
    print("\nТаблиця 3: Фактичні наслідки політики реагування на test set:")
    print(df_policy_summary.to_string(index=False))

    # 5. Генерація карток пояснень для 15 об'єктів
    print("\n[КРОК 3] Генерація структурованих карток пояснень...")
    top_comb_ids = df_test_results.nlargest(5, "percentile_combined")["object_idx"].tolist()
    disagree_ids = df_test_results.assign(diff=(df_test_results["percentile_reconstruction"] - df_test_results["percentile_if_pca"]).abs()).nlargest(5, "diff")["object_idx"].tolist()
    fp_ids = df_test_results[(df_test_results["is_controlled_anomaly"] == 0) & (df_test_results["action"] != "no_action")].head(5)["object_idx"].tolist()

    selected_card_ids = list(dict.fromkeys(top_comb_ids + disagree_ids + fp_ids))[:15]
    cards = []
    for o_idx in selected_card_ids:
        row = df_test_results.iloc[o_idx]
        cards.append({
            "object_id": int(o_idx),
            "assigned_cluster": int(row["assigned_cluster"]),
            "boundary_margin": float(row["boundary_margin"]),
            "is_controlled_anomaly": int(row["is_controlled_anomaly"]),
            "scores": {
                "percentile_if_pca": float(row["percentile_if_pca"]),
                "percentile_reconstruction": float(row["percentile_reconstruction"]),
                "percentile_combined": float(row["percentile_combined"])
            },
            "triggered_rules": audit_logs[o_idx]["triggered_rules"],
            "action": row["action"],
            "hypothesis": "Високочастотне спотворення структури пікселів" if row["percentile_reconstruction"] > 0.95 else "Незвичне поєднання координат у підпросторі",
            "limitation": "Оцінка виконана без контексту мітки класу; вимагає візуальної верифікації"
        })

    with open("anomaly_explanation_cards.json", "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=4, ensure_ascii=False)
    print(f"Збережено {len(cards)} карток пояснень у anomaly_explanation_cards.json.")

    # 6. Генерація обов'язкових рисунків (DPI=300)
    print("\n[КРОК 4] Побудова аналітичних графіків...")

    # Рис. 2: Розміри кластерів і стійкість пояснення
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].bar(df_cluster_summary["cluster"], df_cluster_summary["size"], color="#2b5c8f")
    ax[0].set_xlabel("Кластер")
    ax[0].set_ylabel("Кількість об'єктів")
    ax[0].set_title("Розміри кластерів")
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].plot(df_cluster_summary["cluster"], df_cluster_summary["mean_stability"], "o-", label="Стабільність призначення", color="#2ca02c")
    ax[1].plot(df_cluster_summary["cluster"], df_cluster_summary["jaccard_top5"], "s-", label="Стійкість top-5 ознак", color="#d95f02")
    ax[1].axhline(0.80, color="gray", linestyle="--", label="Контрольна межа 0.8")
    ax[1].set_xlabel("Кластер")
    ax[1].set_ylabel("Частка / індекс")
    ax[1].set_title("Стійкість пояснення")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("pr6_fig2_cluster_sizes_stability.png", dpi=300)
    plt.close()

    # Рис. 3: Теплова карта стандартизованих профілів
    pivot_prof = df_profiles_base.pivot(index="cluster", columns="feature", values="std_profile")
    # Вибираємо 18 ознак із найбільшим максимальним абсолютним відхиленням
    top_cols = pivot_prof.abs().max(axis=0).nlargest(18).index
    plt.figure(figsize=(10, 5))
    plt.imshow(pivot_prof[top_cols], cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5)
    plt.colorbar(label="Відхилення від загального середнього, SD")
    plt.xticks(np.arange(len(top_cols)), top_cols, rotation=60, ha="right", fontsize=8)
    plt.yticks(np.arange(k), [f"C{i}" for i in range(k)])
    plt.title("Стандартизовані профілі кластерів (18 найбільш контрастних ознак)")
    plt.tight_layout()
    plt.savefig("pr6_fig3_cluster_profiles_heatmap.png", dpi=300)
    plt.close()

    # Рис. 4: Представники кластерів (сітка 9х3)
    fig, axes = plt.subplots(k, 3, figsize=(6, 14))
    for c_id in range(k):
        sub_r = df_reps[df_reps["cluster"] == c_id]
        typ_id = sub_r[sub_r["kind"] == "typical"].iloc[0]["object_id"]
        bnd_id = sub_r[sub_r["kind"] == "boundary"].iloc[0]["object_id"]
        dst_id = sub_r[sub_r["kind"] == "distant"].iloc[0]["object_id"]

        axes[c_id, 0].imshow(X_train[typ_id].reshape(8, 8), cmap="gray_r")
        axes[c_id, 0].set_title(f"Типовий: C{c_id}, id={typ_id}", fontsize=8)
        axes[c_id, 0].axis("off")

        axes[c_id, 1].imshow(X_train[bnd_id].reshape(8, 8), cmap="gray_r")
        axes[c_id, 1].set_title(f"Граничний: C{c_id}, id={bnd_id}", fontsize=8)
        axes[c_id, 1].axis("off")

        axes[c_id, 2].imshow(X_train[dst_id].reshape(8, 8), cmap="gray_r")
        axes[c_id, 2].set_title(f"Віддалений: C{c_id}, id={dst_id}", fontsize=8)
        axes[c_id, 2].axis("off")

    plt.suptitle("Типові, граничні та віддалені представники дев'яти кластерів", y=0.995, fontsize=11)
    plt.tight_layout()
    plt.savefig("pr6_fig4_cluster_representatives.png", dpi=300)
    plt.close()

    # Рис. 5: Граничність, нетиповість і стійкість
    plt.figure(figsize=(8.5, 5))
    sc = plt.scatter(diag_train["boundary_margin"], diag_train["distance_to_center"],
                     c=diag_train["stability"], cmap="viridis", alpha=0.75, s=30)
    plt.colorbar(sc, label="Стійкість призначення")
    plt.axvline(cfg["boundary_margin_threshold"], color="red", linestyle="--", label="Умовна межа margin 0.05")
    plt.xlabel("Boundary margin (менше = ближче до межі)")
    plt.ylabel("Відстань до центроїда")
    plt.title("Діагностика меж: граничність, нетиповість і стійкість")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr6_fig5_boundary_diagnostics.png", dpi=300)
    plt.close()

    # Рис. 6: Екстремальні об'єкти перших 3 компонент PCA
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
    for i in range(3):
        # Ваги
        im = axes[i, 0].imshow(pca.components_[i].reshape(8, 8), cmap="coolwarm")
        axes[i, 0].set_title(f"PC{i+1}: ваги", fontsize=9)
        axes[i, 0].axis("off")

        # Мін / Макс
        min_idx = int(np.argmin(Z_train[:, i]))
        max_idx = int(np.argmax(Z_train[:, i]))

        axes[i, 1].imshow(X_train[min_idx].reshape(8, 8), cmap="gray_r")
        axes[i, 1].set_title(f"Мін: {Z_train[min_idx, i]:.2f}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(X_train[max_idx].reshape(8, 8), cmap="gray_r")
        axes[i, 2].set_title(f"Макс: {Z_train[max_idx, i]:.2f}", fontsize=9)
        axes[i, 2].axis("off")

    plt.suptitle("Перевірка пояснення PCA на екстремальних об'єктах", y=0.99, fontsize=11)
    plt.tight_layout()
    plt.savefig("pr6_fig6_pca_extremes.png", dpi=300)
    plt.close()

    # Рис. 7: Два сигнали аномальності
    plt.figure(figsize=(7.5, 5))
    clean_mask = (y_test_anom == 0)
    anom_mask = (y_test_anom == 1)
    plt.scatter(p_if_test[clean_mask], p_rec_test[clean_mask], alpha=0.4, s=25, color="#2b5c8f", label="Чисті тестові")
    plt.scatter(p_if_test[anom_mask], p_rec_test[anom_mask], alpha=0.8, s=40, color="#d95f02", label="Контрольні аномалії")
    plt.axvline(0.95, color="gray", linestyle=":")
    plt.axhline(0.95, color="gray", linestyle=":")
    plt.xlabel("Процентиль Isolation Forest у PCA-просторі")
    plt.ylabel("Процентиль похибки PCA-відновлення")
    plt.title("Два різні сигнали аномальності")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr6_fig7_two_anomaly_signals.png", dpi=300)
    plt.close()

    # Рис. 8: Приклади карток аномалій (сітка 3х4)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax_idx, o_idx in enumerate(selected_card_ids[:12]):
        ax = axes.flat[ax_idx]
        ax.imshow(X_test_corrupted[o_idx].reshape(8, 8), cmap="gray_r")
        r = df_test_results.iloc[o_idx]
        ax.set_title(f"id={o_idx}, {r['action']}\nrec={r['percentile_reconstruction']:.2f}, IF={r['percentile_if_pca']:.2f}", fontsize=7.5)
        ax.axis("off")
    plt.suptitle("Приклади об'єктів для структурованого пояснення аномалій", y=0.99, fontsize=10.5)
    plt.tight_layout()
    plt.savefig("pr6_fig8_anomaly_cards_samples.png", dpi=300)
    plt.close()

    # Рис. 9: Наслідки правил реагування
    plt.figure(figsize=(8.5, 4.5))
    y_pos = np.arange(len(df_policy_summary))
    plt.barh(y_pos, df_policy_summary["clean_samples"], color="#2b5c8f", label="Чисті об'єкти")
    plt.barh(y_pos, df_policy_summary["controlled_anomalies"], left=df_policy_summary["clean_samples"], color="#d95f02", label="Контрольні аномалії")
    plt.yticks(y_pos, df_policy_summary["action"])
    plt.xlabel("Кількість тестових об'єктів")
    plt.title("Наслідки правил реагування на тестовій вибірці (triage)")
    plt.grid(alpha=0.3, axis="x")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr6_fig9_policy_consequences.png", dpi=300)
    plt.close()

    print("Усі графіки pr6_fig2 - pr6_fig9 успішно створено.")
    print("=" * 80)


if __name__ == "__main__":
    main()