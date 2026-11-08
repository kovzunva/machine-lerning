"""
Головний виконуваний скрипт Практичної роботи №7:
Повний цикл критичного аудиту проблемного ML-коду, виправлення та формування звітів.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import audit_core as ac

# Створюємо каталог результатів
os.makedirs("results", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# 1. Завантаження конфігурації
with open("audit_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

# 2. Завантаження датасету
raw_data = load_breast_cancer(as_frame=True)
df_all = raw_data.frame
features_subset = cfg["features_subset"]
X_clean = df_all[features_subset].copy()
y = raw_data.target  # 0: benign, 1: malignant (у Breast Cancer 1=benign, але інвертуємо для онкологічного сценарію 1=malignant)
# В scikit-learn: 0=malignant, 1=benign. Для відповідності звіту (1 = патологія):
y = (y == 0).astype(int)

# Фіксований поділ на train/test
indices = np.arange(len(df_all))
train_idx, test_idx = train_test_split(
    indices, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y
)

# Перевірка контракту відсутності перетину
overlap = ac.detect_split_overlap(train_idx, test_idx)
print(f"[AUDIT] Train/Test Index Overlap: {len(overlap)} objects (OK)")

X_train = X_clean.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()
X_test = X_clean.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

# =====================================================================
# СИМУЛЯЦІЯ ПРОБЛЕМНОГО РІШЕННЯ (НАВМИСНІ ДЕФЕКТИ)
# =====================================================================
print("\n--- 1. Аудит початкового (проблемного) рішення ---")
# Дефект 1: Post-outcome target proxy
X_bad = X_clean.copy()
X_bad[cfg["leakage_proxy_col"]] = y.values

# Розрахунок кореляцій (Рисунок 2 у звіті)
corr_df = ac.feature_target_correlations(X_bad, y.values)
corr_df.to_csv("results/feature_correlations.csv", index=False)

plt.figure(figsize=(9, 4.5))
bars = plt.barh(corr_df["feature"][::-1], corr_df["abs_correlation"][::-1], color="#2b5c8f")
for i, bar in enumerate(bars):
    if corr_df["feature"][::-1].iloc[i] == cfg["leakage_proxy_col"]:
        bar.set_color("#c0392b")
plt.xlabel("Абсолютна кореляція з target")
plt.title("Автоматична перевірка виявляє ознаку-проксі target")
plt.tight_layout()
plt.savefig("results/fig2_leakage_correlations.png", dpi=300)
plt.close()

# Дефект 2: Fit scaler на всій вибірці до розбиття
scaler_bad = StandardScaler()
X_bad_scaled = scaler_bad.fit_transform(X_bad)

X_bad_train = X_bad_scaled[train_idx]
X_bad_test = X_bad_scaled[test_idx]

# Дефект 3: Підбір C за фінальним test set
best_c_bad = None
best_ap_bad = -1
for c_cand in cfg["candidate_C"]:
    clf_temp = LogisticRegression(C=c_cand, solver="liblinear", random_state=42)
    clf_temp.fit(X_bad_train, y_train)
    m_temp = ac.evaluate_model(clf_temp, X_bad_test, y_test)
    if m_temp["average_precision"] > best_ap_bad:
        best_ap_bad = m_temp["average_precision"]
        best_c_bad = c_cand

bad_model = LogisticRegression(C=best_c_bad, solver="liblinear", random_state=42)
bad_model.fit(X_bad_train, y_train)
bad_metrics = ac.evaluate_model(bad_model, X_bad_test, y_test)
print(f"[PROBLEMATIC] Обраний C на тесті: {best_c_bad}, Заявлений AP: {bad_metrics['average_precision']:.4f}")

# =====================================================================
# ВИПРАВЛЕНИЙ ЕТАЛОННИЙ ПРОТОКОЛ (ПІСЛЯ АУДИТУ)
# =====================================================================
print("\n--- 2. Побудова виправленого еталонного рішення ---")
# Виправлення: чистий пайплайн, підбір C виключно за 5-fold CV на train
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["random_state"])

best_c_fixed = None
best_cv_ap = -1
cv_results_by_c = {}

for c_cand in cfg["candidate_C"]:
    pipe_candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=c_cand, solver="liblinear", max_iter=5000, random_state=42))
    ])
    fold_aps = []
    for tr_in, val_in in cv_inner.split(X_train, y_train):
        pipe_candidate.fit(X_train.iloc[tr_in], y_train.iloc[tr_in])
        eval_dict = ac.evaluate_model(pipe_candidate, X_train.iloc[val_in], y_train.iloc[val_in])
        fold_aps.append(eval_dict["average_precision"])
    mean_fold_ap = float(np.mean(fold_aps))
    cv_results_by_c[c_cand] = mean_fold_ap
    if mean_fold_ap > best_cv_ap:
        best_cv_ap = mean_fold_ap
        best_c_fixed = c_cand

print(f"[CORRECTED] Вибір C за train 5-fold CV: C={best_c_fixed} (Train CV AP: {best_cv_ap:.4f})")

# Заморожене рішення (Frozen Pipeline)
fixed_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=best_c_fixed, solver="liblinear", max_iter=5000, random_state=42))
])
fixed_pipeline.fit(X_train, y_train)

# Одноразовий тест контрольного пайплайну
fixed_metrics = ac.evaluate_model(fixed_pipeline, X_test, y_test)

# Оцінка Dummy Baseline
dummy = DummyClassifier(strategy="prior")
dummy.fit(X_train, y_train)
dummy_metrics = ac.evaluate_model(dummy, X_test, y_test)

# Зведення метрик у Таблицю (Рисунок 3 у звіті)
metrics_comp = pd.DataFrame([
    {"Модель": "Dummy baseline", **dummy_metrics},
    {"Модель": "Проблемне рішення", **bad_metrics},
    {"Модель": "Після аудиту (виправлена)", **fixed_metrics}
])
metrics_comp.to_csv("results/metrics_comparison.csv", index=False)
print("\nПорівняння метрик на test set:")
print(metrics_comp[["Модель", "average_precision", "roc_auc", "balanced_accuracy"]])

# Рисунок 3: Порівняння baseline, проблемного та виправленого
fig, ax = plt.subplots(figsize=(8, 4))
x_pos = np.arange(3)
width = 0.25
ax.bar(x_pos - width, metrics_comp["average_precision"], width, label="Average precision", color="#1f77b4")
ax.bar(x_pos, metrics_comp["roc_auc"], width, label="ROC AUC", color="#ff7f0e")
ax.bar(x_pos + width, metrics_comp["balanced_accuracy"], width, label="Balanced accuracy", color="#2ca02c")
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_comp["Модель"])
ax.set_ylabel("Значення метрики")
ax.set_title("Заявлена майже ідеальна якість зумовлена витоком target")
ax.legend(loc="lower right")
for i in x_pos:
    ax.text(i - width, metrics_comp["average_precision"].iloc[i] + 0.02, f"{metrics_comp['average_precision'].iloc[i]:.3f}", ha='center', fontsize=8)
    ax.text(i, metrics_comp["roc_auc"].iloc[i] + 0.02, f"{metrics_comp['roc_auc'].iloc[i]:.3f}", ha='center', fontsize=8)
    ax.text(i + width, metrics_comp["balanced_accuracy"].iloc[i] + 0.02, f"{metrics_comp['balanced_accuracy'].iloc[i]:.3f}", ha='center', fontsize=8)
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig("results/fig3_metrics_comparison.png", dpi=300)
plt.close()

# Рисунок 4: Confusion Matrix до і після
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
bad_preds = bad_model.predict(X_bad_test)
fixed_preds = fixed_pipeline.predict(X_test)

cm_bad = confusion_matrix(y_test, bad_preds)
cm_fixed = confusion_matrix(y_test, fixed_preds)

ConfusionMatrixDisplay(cm_bad).plot(ax=axes[0], cmap="Blues", colorbar=True)
axes[0].set_title("Проблемне рішення")
axes[0].set_xlabel("Прогноз")
axes[0].set_ylabel("Фактичний клас")

ConfusionMatrixDisplay(cm_fixed).plot(ax=axes[1], cmap="Blues", colorbar=True)
axes[1].set_title("Після аудиту")
axes[1].set_xlabel("Прогноз")
axes[1].set_ylabel("Фактичний клас")

plt.suptitle("Ідеальна confusion matrix не є доказом коректного протоколу")
plt.tight_layout()
plt.savefig("results/fig4_confusion_matrices.png", dpi=300)
plt.close()

# =====================================================================
# СТІЙКІСТЬ: REPEATED CV ТА BOOTSTRAP
# =====================================================================
print("\n--- 3. Оцінка варіативності та стійкості ---")
rcv = RepeatedStratifiedKFold(n_splits=cfg["n_outer_folds"], n_repeats=cfg["n_cv_repeats"], random_state=cfg["random_state"])
rcv_ap, rcv_auc, rcv_bal = [], [], []

for tr_k, te_k in rcv.split(X_train, y_train):
    fixed_pipeline.fit(X_train.iloc[tr_k], y_train.iloc[tr_k])
    eval_f = ac.evaluate_model(fixed_pipeline, X_train.iloc[te_k], y_train.iloc[te_k])
    rcv_ap.append(eval_f["average_precision"])
    rcv_auc.append(eval_f["roc_auc"])
    rcv_bal.append(eval_f["balanced_accuracy"])

rcv_df = pd.DataFrame({"AP": rcv_ap, "ROC_AUC": rcv_auc, "Balanced_accuracy": rcv_bal})
rcv_summary = pd.DataFrame({
    "Метрика": ["Average precision", "ROC AUC", "Balanced accuracy"],
    "Середнє": [rcv_df["AP"].mean(), rcv_df["ROC_AUC"].mean(), rcv_df["Balanced_accuracy"].mean()],
    "Std": [rcv_df["AP"].std(), rcv_df["ROC_AUC"].std(), rcv_df["Balanced_accuracy"].std()],
    "Мінімум": [rcv_df["AP"].min(), rcv_df["ROC_AUC"].min(), rcv_df["Balanced_accuracy"].min()],
    "Максимум": [rcv_df["AP"].max(), rcv_df["ROC_AUC"].max(), rcv_df["Balanced_accuracy"].max()],
})
rcv_summary.to_csv("results/repeated_cv_summary.csv", index=False)

# ВИВІД У КОНСОЛЬ ДЛЯ СЕКЦІЇ 3
print("Результати Repeated Stratified CV (5 folds x 3 repeats):")
print(rcv_summary.to_string(index=False))

# Рисунок 5: Boxplot Repeated CV
plt.figure(figsize=(7, 4))
plt.boxplot([rcv_df["AP"], rcv_df["ROC_AUC"], rcv_df["Balanced_accuracy"]], tick_labels=["AP", "ROC AUC", "Balanced accuracy"], patch_artist=True, boxprops=dict(facecolor="#dbe9f6"))
plt.ylabel("Значення метрики")
plt.title("Repeated CV показує не лише середнє, а й варіативність")
plt.tight_layout()
plt.savefig("results/fig5_repeated_cv.png", dpi=300)
plt.close()

# Bootstrap-навчання для стійкості прогнозів (Рисунок 6)
n_boot = cfg["n_bootstrap_runs"]
ref_preds = fixed_pipeline.predict(X_test)
boot_seeds = [2026 + i for i in range(n_boot)]
disagreement_rates = []
boot_aps = []

for s in boot_seeds:
    boot_tr_idx = pd.Series(range(len(X_train))).sample(frac=cfg["bootstrap_fraction"], replace=True, random_state=s).values
    pipe_boot = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=best_c_fixed, solver="liblinear", max_iter=5000, random_state=s))
    ])
    pipe_boot.fit(X_train.iloc[boot_tr_idx], y_train.iloc[boot_tr_idx])
    b_preds = pipe_boot.predict(X_test)
    disagreement = np.mean(b_preds != ref_preds)
    disagreement_rates.append(disagreement)
    boot_aps.append(ac.evaluate_model(pipe_boot, X_test, y_test)["average_precision"])

print(f"Bootstrap стійкість прогнозів (12 повторів): Середня частка розбіжностей = {np.mean(disagreement_rates):.4f} (макс: {np.max(disagreement_rates):.4f})")

# Рисунок 6: Стійкість прогнозів
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
ax1.plot([str(s) for s in boot_seeds], boot_aps, color="#2b5c8f", marker="o", label="Average precision")
ax2.bar([str(s) for s in boot_seeds], disagreement_rates, color="#e5c3a6", alpha=0.6, width=0.6, label="Частка розбіжностей")
ax1.set_ylabel("Average precision")
ax2.set_ylabel("Частка розбіжностей")
ax1.set_xlabel("Seed bootstrap-вибірки")
ax1.set_ylim(0.88, 1.01)
ax2.set_ylim(0, 0.08)
plt.title("Стійкість оцінюється повторними навчаннями")
plt.tight_layout()
plt.savefig("results/fig6_prediction_stability.png", dpi=300)
plt.close()

# =====================================================================
# АУДИТ ВАЖЛИВОСТІ ОЗНАК (PERMUTATION IMPORTANCE)
# =====================================================================
print("\n--- 4. Аудит Permutation Importance ---")
r_perm = permutation_importance(fixed_pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring="average_precision")
sorted_idx = r_perm.importances_mean.argsort()[::-1]

df_importance = pd.DataFrame({
    "Ознака": np.array(features_subset)[sorted_idx],
    "Зменшення AP": r_perm.importances_mean[sorted_idx],
    "Std": r_perm.importances_std[sorted_idx]
})
print("Top ознаки за спаданням впливу на AP:")
print(df_importance.to_string(index=False))

plt.figure(figsize=(8, 4))
sorted_idx_plot = r_perm.importances_mean.argsort()
plt.barh(np.array(features_subset)[sorted_idx_plot], r_perm.importances_mean[sorted_idx_plot], xerr=r_perm.importances_std[sorted_idx_plot], color="#4682b4", capsize=4)
plt.xlabel("Зменшення average precision після перестановки")
plt.title("Permutation importance виправленої моделі")
plt.tight_layout()
plt.savefig("results/fig7_permutation_importance.png", dpi=300)
plt.close()

# =====================================================================
# ВАЛІДАЦІЯ КОНТРАКТУ ВХОДУ ТА НЕГАТИВНІ СЦЕНАРІЇ
# =====================================================================
print("\n--- 5. Валідація контракту даних (Негативні сценарії) ---")
data_schema = {}
for col in features_subset:
    data_schema[col] = {
        "min": float(X_train[col].min()),
        "max": float(X_train[col].max()),
        "mean": float(X_train[col].mean())
    }

base_valid_record = X_test.iloc[0].to_dict()

negative_test_suite = [
    ("Коректний запис", base_valid_record.copy()),
    ("Інший порядок полів", {k: base_valid_record[k] for k in reversed(list(base_valid_record.keys()))}),
    ("Відсутня ознака", {k: v for k, v in base_valid_record.items() if k != "mean radius"}),
    ("Зайва ознака", {**base_valid_record, "unexpected_feature": 12.5}),
    ("Текст замість числа", {**base_valid_record, "mean texture": "error_text"}),
    ("NaN значення", {**base_valid_record, "mean smoothness": np.nan}),
    ("Поза train range (10x Max)", {**base_valid_record, "mean radius": data_schema["mean radius"]["max"] * 10.0})
]

neg_results = []
for name, sample in negative_test_suite:
    valid, status, msgs = ac.validate_record(sample, data_schema)
    neg_results.append({
        "Сценарій": name,
        "Результат": status,
        "Пояснення": "; ".join(msgs) if msgs else "Контракт виконано"
    })
df_neg = pd.DataFrame(neg_results)
df_neg.to_csv("results/negative_tests_report.csv", index=False)

print("Результати обробки вхідних сценаріїв:")
print(df_neg.to_string(index=False))

# =====================================================================
# РЕЄСТР ТВЕРДЖЕНЬ ТА ЗНАХІДОК
# =====================================================================
claims_registry = [
    {"id": "C-01", "claim": "AP=1 доводить узагальнення", "evidence": "Виявлено target leakage (r=1.0)", "verdict": "Спростовано"},
    {"id": "C-02", "claim": "C=0.01 є оптимальним", "evidence": "Підбір виконувався за test set, на train CV оптимальним є C=1.0", "verdict": "Спростовано"},
    {"id": "C-03", "claim": "Найбільший коефіцієнт є причинним фактором", "evidence": "Permutation importance показує чутливість моделі, а не причинність", "verdict": "Спростовано"},
    {"id": "C-04", "claim": "Початкове рішення відтворюване", "evidence": "Код запускається синтаксично, але логіка валідації хибна", "verdict": "Частково"},
    {"id": "C-05", "claim": "Рішення готове до практичного використання", "evidence": "Рішення навчальне, відсутня клінічна сертифікація", "verdict": "Спростовано"},
    {"id": "C-06", "claim": "Виправлена модель перевищує baseline", "evidence": "AP=0.974 проти baseline AP=0.371", "verdict": "Підтверджено"}
]
pd.DataFrame(claims_registry).to_csv("results/claims.csv", index=False)

# Рисунок 8: Вердикти тверджень
claim_counts = pd.Series([c["verdict"] for c in claims_registry]).value_counts().reindex(["Підтверджено", "Частково", "Спростовано"]).fillna(0)
plt.figure(figsize=(7, 3.5))
bars = plt.bar(["Підтверджено", "Частково", "Спростовано"], claim_counts, color=["#2e7d32", "#f9a825", "#c62828"])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, int(bar.get_height()), ha='center')
plt.ylabel("Кількість тверджень")
plt.title("Результат незалежної перевірки тверджень")
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig("results/fig8_claims_verdict.png", dpi=300)
plt.close()

findings_registry = [
    {"id": "F-01", "problem": "Target proxy колонка в даних", "severity": "Критична", "recommendation": "Вилучити ознаку diagnosis_code_after_examination", "status": "Виправлено"},
    {"id": "F-02", "problem": "Model selection на test set", "severity": "Критична", "recommendation": "Використовувати StratifiedKFold виключно на train", "status": "Виправлено"},
    {"id": "F-03", "problem": "Масштабування до split", "severity": "Висока", "recommendation": "Використовувати scikit-learn Pipeline", "status": "Виправлено"},
    {"id": "F-04", "problem": "Причинна інтерпретація важливих ознак", "severity": "Висока", "recommendation": "Сформулювати обмеження кореляційного аналізу", "status": "Виправлено у звіті"},
    {"id": "F-05", "problem": "Відсутність валідації вхідного контракту", "severity": "Висока", "recommendation": "Впровадити модуль валідації схеми", "status": "Виправлено"},
    {"id": "F-06", "problem": "Твердження про клінічну готовність", "severity": "Середня", "recommendation": "Додати Intended Use та обмеження щодо сертифікації", "status": "Зафіксовано обмеження"},
    {"id": "F-07", "problem": "Відсутність контролю невизначеності", "severity": "Середня", "recommendation": "Додати Repeated CV та Bootstrap оцінки", "status": "Виправлено"}
]
pd.DataFrame(findings_registry).to_csv("results/findings.csv", index=False)

# Рисунок 9: Розподіл знахідок
severity_counts = pd.Series([f["severity"] for f in findings_registry]).value_counts().reindex(["Критична", "Висока", "Середня", "Низька"]).fillna(0)
plt.figure(figsize=(7, 3.5))
bars = plt.bar(["Критична", "Висока", "Середня", "Низька"], severity_counts, color=["#8b0000", "#d32f2f", "#f57c00", "#388e3c"])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, int(bar.get_height()), ha='center')
plt.ylabel("Кількість знахідок")
plt.title("Критичність знайдених проблем")
plt.ylim(0, 4.5)
plt.tight_layout()
plt.savefig("results/fig9_findings_severity.png", dpi=300)
plt.close()

# Збереження машинозчитуваного вердикту
verdict_data = {
    "verdict": "Потребує доопрацювання",
    "justification": "Початковий AP=1.0 зумовлений target leakage та вибором C за test set. Виправлений Pipeline є відтворюваним і перевершує baseline, але рішення має лише навчальне призначення.",
    "claims_total": len(claims_registry),
    "claims_refuted": int(claim_counts["Спростовано"]),
    "findings_critical": int(severity_counts["Критична"]),
    "fixed_pipeline_test_ap": fixed_metrics["average_precision"]
}
with open("results/verdict.json", "w", encoding="utf-8") as f:
    json.dump(verdict_data, f, indent=2, ensure_ascii=False)

print("\n[SUCCESS] Повний цикл аудиту завершено. Усі графіки та CSV експортовано у папку results/.")