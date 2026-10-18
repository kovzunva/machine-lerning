import json
import warnings
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. Завантаження даних та строге приведення класів:
# В оригінальному sklearn breast_cancer: 0 = Malignant, 1 = Benign.
# За умовою ЛР2/ПР4: 1 = Malignant (позитивний, рак), 0 = Benign (доброякісний).
raw_data = load_breast_cancer()
X = raw_data.data
y = (raw_data.target == 0).astype(int)  # Тепер 1 = Malignant (212 шт), 0 = Benign (357 шт)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# 2. Pipeline
base_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, random_state=RANDOM_STATE)),
    ]
)

# 3. Чесні OOF-прогнози
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
raw_oof = cross_val_predict(
    base_pipeline,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

# Калібратор через вкладену CV
calibrated_model = CalibratedClassifierCV(
    estimator=base_pipeline, method="sigmoid", cv=3
)
calibrated_oof = cross_val_predict(
    calibrated_model,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

# Оскільки сира логістична регресія має менший Brier score, зберігаємо її бали як основні
chosen_scores = raw_oof

folds = np.zeros(len(y_train), dtype=int)
for fold_idx, (_, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
    folds[val_idx] = fold_idx

df_oof = pd.DataFrame({
    "object_id": np.arange(len(y_train)),
    "y_true": y_train,
    "raw_score": raw_oof,
    "calibrated_score": calibrated_oof,
    "fold": folds,
})
df_oof.to_csv("pr4_oof_predictions.csv", index=False)

brier_raw = brier_score_loss(y_train, raw_oof)
brier_cal = brier_score_loss(y_train, calibrated_oof)

# 4. Сітка порогів
thresholds = np.concatenate([[0.001], np.linspace(0.01, 0.99, 99), [0.999]])
thresholds = np.unique(np.round(thresholds, 4))

scenarios = {
    "Симетричний (контрольний)": {"c_fp": 1.0, "c_fn": 1.0},
    "Клінічний скринінг (реалістичний)": {"c_fp": 1.0, "c_fn": 8.0},
    "Ургентний / Стресовий": {"c_fp": 1.0, "c_fn": 25.0},
}

n_train = len(y_train)
records = []
for t in thresholds:
    y_pred = (chosen_scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
    row = {
        "threshold": float(t),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_train, y_pred, zero_division=0)),
        "recall": float(recall_score(y_train, y_pred, zero_division=0)),
        "f1": float(f1_score(y_train, y_pred, zero_division=0)),
        "alert_rate": float((tp + fp) / n_train),
    }
    for sc_name, sc_cost in scenarios.items():
        row[f"cost_{sc_name}"] = float((sc_cost["c_fp"] * fp + sc_cost["c_fn"] * fn) / n_train)
    records.append(row)

df_thresh = pd.DataFrame(records)
df_thresh.to_csv("pr4_threshold_grid.csv", index=False)

# 5. Вибір робочої точки за квоти q <= 0.40
sc_target = "cost_Клінічний скринінг (реалістичний)"
valid_sub = df_thresh[df_thresh["alert_rate"] <= 0.40]

if valid_sub.empty:
    best_constr = df_thresh.loc[df_thresh["alert_rate"].idxmin()]
else:
    best_constr = valid_sub.loc[valid_sub[sc_target].idxmin()]

t_selected = float(best_constr["threshold"])

# Необмежений оптимум
unconstrained = df_thresh.loc[df_thresh[sc_target].idxmin()]

# 6. Bootstrap для оцінки стійкості порога
n_bootstraps = 200
boot_thresholds = []
for b in range(n_bootstraps):
    boot_idx = np.random.choice(n_train, size=n_train, replace=True)
    b_scores = chosen_scores[boot_idx]
    b_true = y_train[boot_idx]
    
    b_costs = []
    for t in thresholds:
        q = np.mean(b_scores >= t)
        if q <= 0.40:
            c = (1.0 * np.sum((b_scores >= t) & (b_true == 0)) + 8.0 * np.sum((b_scores < t) & (b_true == 1))) / n_train
            b_costs.append((c, t))
    if b_costs:
        b_costs.sort()
        boot_thresholds.append(b_costs[0][1])

# 7. Графіки
# Рис 1: Калібрування
prob_true_raw, prob_pred_raw = calibration_curve(y_train, raw_oof, n_bins=10, strategy="uniform")
prob_true_cal, prob_pred_cal = calibration_curve(y_train, calibrated_oof, n_bins=10, strategy="uniform")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot([0, 1], [0, 1], "k--", label="Ідеальна калібрація")
ax[0].plot(prob_pred_raw, prob_true_raw, "o-", label=f"Початкова LR (Brier={brier_raw:.4f})")
ax[0].plot(prob_pred_cal, prob_true_cal, "s-", label=f"Калібрована (Brier={brier_cal:.4f})")
ax[0].set_xlabel("Середній прогнозований бал")
ax[0].set_ylabel("Фактична частка Malignant (y=1)")
ax[0].set_title("Калібрувальна крива (5-fold OOF)")
ax[0].grid(alpha=0.3)
ax[0].legend()

ax[1].bar(["Початкова LR", "Калібрована LR"], [brier_raw, brier_cal], color=["#2b5c8f", "#d95f02"], width=0.45)
ax[1].set_ylabel("Brier score (менше — краще)")
ax[1].set_title("Порівняння Brier Score")
for i, v in enumerate([brier_raw, brier_cal]):
    ax[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("pr4_fig1_calibration.png", dpi=300)
plt.close()

# Рис 2: Метрики від порога
plt.figure(figsize=(10, 4.5))
plt.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#2ca02c")
plt.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall", color="#d62728")
plt.plot(df_thresh["threshold"], df_thresh["f1"], label="F1", color="#1f77b4")
plt.plot(df_thresh["threshold"], df_thresh["alert_rate"], "--", label="Alert Rate (q)", color="#9467bd")
plt.axvline(0.5, color="gray", linestyle=":", label="Дефолт t=0.5")
plt.xlabel("Поріг рішення t")
plt.ylabel("Значення метрики")
plt.title("Залежність метрик від порога (5-fold OOF)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig2_metrics_vs_threshold.png", dpi=300)
plt.close()

# Рис 3: Вартість
plt.figure(figsize=(9, 4.5))
for sc_name, col in zip(scenarios.keys(), ["#1f77b4", "#ff7f0e", "#d62728"]):
    plt.plot(df_thresh["threshold"], df_thresh[f"cost_{sc_name}"], label=sc_name, color=col)
plt.xlabel("Поріг рішення t")
plt.ylabel("Середня вартість на об'єкт (C_avg)")
plt.title("Криві середньої вартості помилок")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig3_cost_curves.png", dpi=300)
plt.close()

# Рис 4: Вибір за квоти
plt.figure(figsize=(8, 4.5))
plt.plot(df_thresh["alert_rate"] * 100, df_thresh[sc_target], color="gray", label="Усі пороги")
plt.scatter(valid_sub["alert_rate"] * 100, valid_sub[sc_target], color="#2ca02c", s=15, label="Допустимі точки (q <= 40%)")
plt.scatter([best_constr["alert_rate"] * 100], [best_constr[sc_target]], color="red", s=80, zorder=5, label=f"Обраний поріг t={t_selected}")
plt.axvline(40.0, color="darkred", linestyle="--", label="Ліміт місткості 40%")
plt.xlabel("Частка направлених на біопсію (q), %")
plt.ylabel("Середня вартість помилок")
plt.title("Вибір робочої точки за обмеження місткості")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig4_capacity_tradeoff.png", dpi=300)
plt.close()

# Рис 5: Чутливість
ratios = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
sens_data = []
for r in ratios:
    costs_r = (1.0 * df_thresh["fp"] + r * df_thresh["fn"]) / n_train
    t_emp = df_thresh.loc[costs_r.idxmin(), "threshold"]
    valid_costs = costs_r[df_thresh["alert_rate"] <= 0.40]
    t_c = df_thresh.loc[valid_costs.idxmin(), "threshold"] if not valid_costs.empty else t_emp
    sens_data.append({"ratio": r, "t_theo": 1.0 / (1.0 + r), "t_emp": t_emp, "t_constr": t_c})
df_sens = pd.DataFrame(sens_data)

plt.figure(figsize=(8, 4.5))
plt.plot(df_sens["ratio"], df_sens["t_theo"], "o-", label="Теоретичний поріг t*")
plt.plot(df_sens["ratio"], df_sens["t_emp"], "s-", label="Емпіричний мінімум (OOF)")
plt.plot(df_sens["ratio"], df_sens["t_constr"], "^--", label="За квоти q <= 40%")
plt.xlabel("Співвідношення c_FN / c_FP")
plt.ylabel("Поріг t")
plt.title("Чутливість вибору порога")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig5_sensitivity.png", dpi=300)
plt.close()

# 8. Клас політики з підтримкою Top-K та ties
@dataclass(frozen=True)
class TriageDecisionPolicy:
    threshold_binary: float
    threshold_low: float
    threshold_high: float
    capacity_quota: float = 0.40

    def decide_binary(self, score: float) -> str:
        if not (0.0 <= score <= 1.0):
            raise ValueError("Score must be in [0, 1]")
        return "urgent_biopsy" if score >= self.threshold_binary else "routine_screening"

    def decide_tri_zone(self, score: float) -> str:
        if not (0.0 <= score <= 1.0):
            raise ValueError("Score must be in [0, 1]")
        if self.threshold_low > self.threshold_high:
            raise ValueError("threshold_low cannot exceed threshold_high")
        if score >= self.threshold_high:
            return "urgent_biopsy"
        elif score >= self.threshold_low:
            return "advanced_ultrasound_review"
        return "routine_screening"

    def decide_top_k(self, scores: List[float], max_k: int = None) -> List[str]:
        if not scores:
            return []
        for s in scores:
            if not (0.0 <= s <= 1.0):
                raise ValueError("All scores must be in [0, 1]")
        n = len(scores)
        k = max_k if max_k is not None else int(np.floor(self.capacity_quota * n))
        k = min(n, max(0, k))
        indexed = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
        actions = ["routine_screening"] * n
        for rank in range(k):
            orig_idx = indexed[rank][0]
            actions[orig_idx] = "urgent_biopsy"
        return actions

# JSON конфігурація
policy_config = {
    "policy_id": "BREAST_CANCER_TRIAGE_POLICY",
    "version": "1.2.0",
    "created_date": "2026-10-18",
    "score_semantics": "Probability of malignancy P(y=1, Malignant | x)",
    "threshold_binary": float(t_selected),
    "threshold_low": 0.20,
    "threshold_high": 0.70,
    "capacity_constraint_q_max": 0.40,
    "cost_scenario": {"c_fp": 1.0, "c_fn": 8.0, "unit": "clinical_risk_units"},
    "actions": {
        "negative": "routine_screening",
        "review": "advanced_ultrasound_review",
        "positive": "urgent_biopsy",
    },
    "provenance": {
        "dataset": "Breast Cancer Wisconsin Diagnostic (1=Malignant, 0=Benign)",
        "train_samples": 455,
        "validation_method": "5-fold Stratified KFold OOF",
        "random_state": RANDOM_STATE,
        "model": "StandardScaler + LogisticRegression(C=1.0)",
    },
}

with open("policy_config.json", "w", encoding="utf-8") as f:
    json.dump(policy_config, f, indent=4, ensure_ascii=False)

# 9. Оцінювання на D_test (один раз)
base_pipeline.fit(X_train, y_train)
test_probs = base_pipeline.predict_proba(X_test)[:, 1]

# а) Дефолт t=0.5
test_pred_def = (test_probs >= 0.5).astype(int)
tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, test_pred_def, labels=[0, 1]).ravel()
rec_d = recall_score(y_test, test_pred_def)
prec_d = precision_score(y_test, test_pred_def)
f1_d = f1_score(y_test, test_pred_def)
cost_d = (1.0 * fp_d + 8.0 * fn_d) / len(y_test)
q_d = (tp_d + fp_d) / len(y_test)

# б) Обраний робочий поріг t_selected
test_pred_sel = (test_probs >= t_selected).astype(int)
tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test, test_pred_sel, labels=[0, 1]).ravel()
rec_s = recall_score(y_test, test_pred_sel)
prec_s = precision_score(y_test, test_pred_sel)
f1_s = f1_score(y_test, test_pred_sel)
cost_s = (1.0 * fp_s + 8.0 * fn_s) / len(y_test)
q_s = (tp_s + fp_s) / len(y_test)

# в) Top-K (квота рівно 40%)
policy = TriageDecisionPolicy(threshold_binary=t_selected, threshold_low=0.20, threshold_high=0.70)
test_actions = policy.decide_top_k(list(test_probs), max_k=int(np.floor(0.40 * len(y_test))))
test_pred_k = np.array([1 if a == "urgent_biopsy" else 0 for a in test_actions])
tn_k, fp_k, fn_k, tp_k = confusion_matrix(y_test, test_pred_k, labels=[0, 1]).ravel()
rec_k = recall_score(y_test, test_pred_k)
prec_k = precision_score(y_test, test_pred_k)
f1_k = f1_score(y_test, test_pred_k)
cost_k = (1.0 * fp_k + 8.0 * fn_k) / len(y_test)
q_k = (tp_k + fp_k) / len(y_test)

brier_test = brier_score_loss(y_test, test_probs)
roc_test = roc_auc_score(y_test, test_probs)

row_def_oof = df_thresh.loc[df_thresh["threshold"].sub(0.5).abs().idxmin()]

print(f"Brier: Raw={brier_raw:.4f}, Calibrated={brier_cal:.4f}")
if boot_thresholds:
    print(f"Bootstrap 95% CI для порога: [{np.percentile(boot_thresholds, 2.5):.3f}, {np.percentile(boot_thresholds, 97.5):.3f}]")
print(f"Дефолт t=0.5 (OOF): q={row_def_oof['alert_rate']*100:.2f}%, TP={row_def_oof['tp']}, FP={row_def_oof['fp']}, TN={row_def_oof['tn']}, FN={row_def_oof['fn']}, Prec={row_def_oof['precision']:.4f}, Rec={row_def_oof['recall']:.4f}, Cost={row_def_oof[sc_target]:.4f}")
print(f"Необмежений мінімум t={unconstrained['threshold']:.3f} (OOF): q={unconstrained['alert_rate']*100:.2f}%, Cost={unconstrained[sc_target]:.4f}")
print(f"ОБРАНИЙ робочий поріг t={t_selected:.3f} (OOF): q={best_constr['alert_rate']*100:.2f}%, TP={best_constr['tp']}, FP={best_constr['fp']}, TN={best_constr['tn']}, FN={best_constr['fn']}, Prec={best_constr['precision']:.4f}, Rec={best_constr['recall']:.4f}, Cost={best_constr[sc_target]:.4f}")
print("-"*60)
print(f"ТЕСТ D_test (n={len(y_test)}: 72 Benign, 42 Malignant):")
print(f"1) Дефолт t=0.500: TP={tp_d}, FP={fp_d}, TN={tn_d}, FN={fn_d} | Prec={prec_d:.4f}, Rec={rec_d:.4f}, F1={f1_d:.4f} | q={q_d*100:.2f}%, Cost={cost_d:.4f}")
print(f"2) Фіксований t={t_selected:.3f}: TP={tp_s}, FP={fp_s}, TN={tn_s}, FN={fn_s} | Prec={prec_s:.4f}, Rec={rec_s:.4f}, F1={f1_s:.4f} | q={q_s*100:.2f}%, Cost={cost_s:.4f}")
print(f"3) Top-K (квота 40%): TP={tp_k}, FP={fp_k}, TN={tn_k}, FN={fn_k} | Prec={prec_k:.4f}, Rec={rec_k:.4f}, F1={f1_k:.4f} | q={q_k*100:.2f}%, Cost={cost_k:.4f}")
print(f"Test ROC-AUC={roc_test:.4f}, Test Brier={brier_test:.4f}")