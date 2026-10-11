import json
import warnings
from dataclasses import dataclass
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

# Ігноруємо застарілі системні попередження майбутніх версій
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. Завантаження даних та фіксація спліту 80/20
data = load_breast_cancer()
X = data.data
y = data.target  # 0: Benign, 1: Malignant

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# 2. Базовий конвеєр (LR2)
# Прибрано застарілий параметр penalty='l2' — регуляризація L2 діє за замовчуванням
base_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(C=1.0, random_state=RANDOM_STATE),
        ),
    ]
)

# 3. Чесні OOF-прогнози
outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)
raw_oof = cross_val_predict(
    base_pipeline,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

# Sigmoid-калібрування через вкладену CV
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

# Збереження OOF у CSV
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

# 4. Калібрувальна крива та Brier Score
brier_raw = brier_score_loss(y_train, raw_oof)
brier_cal = brier_score_loss(y_train, calibrated_oof)
roc_raw = roc_auc_score(y_train, raw_oof)
roc_cal = roc_auc_score(y_train, calibrated_oof)

prob_true_raw, prob_pred_raw = calibration_curve(
    y_train, raw_oof, n_bins=10, strategy="uniform"
)
prob_true_cal, prob_pred_cal = calibration_curve(
    y_train, calibrated_oof, n_bins=10, strategy="uniform"
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot([0, 1], [0, 1], "k--", label="Ідеальна калібрація")
axes[0].plot(
    prob_pred_raw,
    prob_true_raw,
    marker="o",
    label=f"Початкова LR (Brier={brier_raw:.4f})",
)
axes[0].plot(
    prob_pred_cal,
    prob_true_cal,
    marker="s",
    label=f"Калібрована (Brier={brier_cal:.4f})",
)
axes[0].set_xlabel("Середній прогнозований бал")
axes[0].set_ylabel("Фактична частка Malignant (y=1)")
axes[0].set_title("Калібрувальна крива (5-fold OOF)")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].bar(
    ["Початкова LR", "Калібрована LR"],
    [brier_raw, brier_cal],
    color=["#2b5c8f", "#d95f02"],
    width=0.45,
)
axes[1].set_ylabel("Brier score (менше — краще)")
axes[1].set_title("Порівняння Brier Score")
axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate([brier_raw, brier_cal]):
  axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("pr4_fig1_calibration.png", dpi=300)
plt.close()

# 5. Оцінка сітки порогів та сценарії вартості
thresholds = np.linspace(0.01, 0.99, 99)

scenarios = {
    "Симетричний (контрольний)": {"c_fp": 1.0, "c_fn": 1.0},
    "Клінічний скринінг (реалістичний)": {"c_fp": 1.0, "c_fn": 8.0},
    "Ургентний / Стресовий": {"c_fp": 1.0, "c_fn": 25.0},
}

scores_to_use = calibrated_oof
n_train = len(y_train)

records = []
for t in thresholds:
  y_pred = (scores_to_use >= t).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
  prec = precision_score(y_train, y_pred, zero_division=0)
  rec = recall_score(y_train, y_pred, zero_division=0)
  f1 = f1_score(y_train, y_pred, zero_division=0)
  alert_rate = (tp + fp) / n_train

  row = {
      "threshold": t,
      "tn": tn,
      "fp": fp,
      "fn": fn,
      "tp": tp,
      "precision": prec,
      "recall": rec,
      "f1": f1,
      "alert_rate": alert_rate,
  }
  for sc_name, sc_cost in scenarios.items():
    c_tot = sc_cost["c_fp"] * fp + sc_cost["c_fn"] * fn
    row[f"cost_{sc_name}"] = c_tot / n_train
  records.append(row)

df_thresh = pd.DataFrame(records)
df_thresh.to_csv("pr4_threshold_grid.csv", index=False)

# Графік 2: Метрики та навантаження
plt.figure(figsize=(10, 4.5))
plt.plot(
    df_thresh["threshold"],
    df_thresh["precision"],
    label="Precision",
    color="#2ca02c",
)
plt.plot(
    df_thresh["threshold"], df_thresh["recall"], label="Recall", color="#d62728"
)
plt.plot(df_thresh["threshold"], df_thresh["f1"], label="F1", color="#1f77b4")
plt.plot(
    df_thresh["threshold"],
    df_thresh["alert_rate"],
    label="Alert Rate (q)",
    color="#9467bd",
    linestyle="--",
)
plt.axvline(0.5, color="gray", linestyle=":", label="Дефолт t=0.5")
plt.xlabel("Поріг рішення t")
plt.ylabel("Значення метрики")
plt.title("Залежність метрик та частки спрацювань від порога")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig2_metrics_vs_threshold.png", dpi=300)
plt.close()

# Графік 3: Вартість рішень
plt.figure(figsize=(9, 4.5))
colors = ["#1f77b4", "#ff7f0e", "#d62728"]
for (sc_name, sc_cost), col in zip(scenarios.items(), colors):
  plt.plot(
      df_thresh["threshold"],
      df_thresh[f"cost_{sc_name}"],
      label=f"{sc_name} (c_FN/c_FP={sc_cost['c_fn']/sc_cost['c_fp']:.0f})",
      color=col,
  )
plt.xlabel("Поріг рішення t")
plt.ylabel("Середня вартість на об'єкт (C_avg)")
plt.title("Криві середньої вартості помилок для різних клінічних сценаріїв")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig3_cost_curves.png", dpi=300)
plt.close()

# 6. Ресурсне обмеження (Клінічна місткість біопсій q <= 0.40)
Q_MAX = 0.40
sc_target = "Клінічний скринінг (реалістичний)"
valid_subset = df_thresh[df_thresh["alert_rate"] <= Q_MAX]
best_constrained_row = valid_subset.loc[
    valid_subset[f"cost_{sc_target}"].idxmin()
]
t_constrained = best_constrained_row["threshold"]

unconstrained_row = df_thresh.loc[df_thresh[f"cost_{sc_target}"].idxmin()]
t_unconstrained = unconstrained_row["threshold"]

# Графік 4: Вибір допустимої точки
plt.figure(figsize=(8, 4.5))
plt.plot(
    df_thresh["alert_rate"] * 100,
    df_thresh[f"cost_{sc_target}"],
    color="gray",
    label="Усі пороги",
)
plt.scatter(
    valid_subset["alert_rate"] * 100,
    valid_subset[f"cost_{sc_target}"],
    color="#2ca02c",
    s=15,
    label="Допустимі точки (q <= 40%)",
)
plt.scatter(
    [best_constrained_row["alert_rate"] * 100],
    [best_constrained_row[f"cost_{sc_target}"]],
    color="red",
    s=80,
    zorder=5,
    label=f"Обрана робоча точка t={t_constrained:.3f}",
)
plt.axvline(
    Q_MAX * 100,
    color="darkred",
    linestyle="--",
    label=f"Ліміт місткості {Q_MAX*100:.0f}%",
)
plt.xlabel("Частка направлених на біопсію пацієнток (q), %")
plt.ylabel("Середня вартість помилок")
plt.title("Вибір робочої точки за обмеження місткості")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig4_capacity_tradeoff.png", dpi=300)
plt.close()

# 7. Чутливість (sensitivity curve)
ratios = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
sens_records = []
for r in ratios:
  t_theo = 1.0 / (1.0 + r)
  costs_r = (1.0 * df_thresh["fp"] + r * df_thresh["fn"]) / n_train
  t_emp = df_thresh.loc[costs_r.idxmin(), "threshold"]
  valid_idx = df_thresh["alert_rate"] <= Q_MAX
  t_constr_r = df_thresh.loc[costs_r[valid_idx].idxmin(), "threshold"]
  sens_records.append({
      "ratio": r,
      "t_theo": t_theo,
      "t_emp": t_emp,
      "t_constr": t_constr_r,
  })
df_sens = pd.DataFrame(sens_records)

plt.figure(figsize=(8, 4.5))
plt.plot(
    df_sens["ratio"],
    df_sens["t_theo"],
    marker="o",
    label="Теоретичний поріг t*",
)
plt.plot(
    df_sens["ratio"],
    df_sens["t_emp"],
    marker="s",
    label="Емпіричний мінімум (OOF)",
)
plt.plot(
    df_sens["ratio"],
    df_sens["t_constr"],
    marker="^",
    linestyle="--",
    label=f"З обмеженням q <= {Q_MAX*100:.0f}%",
)
plt.xlabel("Відношення c_FN / c_FP")
plt.ylabel("Поріг t")
plt.title("Чутливість вибору порога до співвідношення вартостей")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pr4_fig5_sensitivity.png", dpi=300)
plt.close()

# 8. Три-зонна політика (відкладене рішення)
t_low = float(np.percentile(calibrated_oof[calibrated_oof < 0.6], 70))
t_high = float(t_constrained)
if t_low >= t_high:
  t_low = t_high * 0.6

# 9. Фіксація JSON-конфігурації
policy_config = {
    "policy_id": "BREAST_CANCER_TRIAGE_POLICY",
    "version": "1.0.0",
    "created_date": "2026-10-11",
    "score_semantics": "Calibrated probability of malignancy (y=1, Malignant)",
    "threshold_binary": round(float(t_constrained), 4),
    "threshold_low": round(float(t_low), 4),
    "threshold_high": round(float(t_high), 4),
    "capacity_constraint_q_max": Q_MAX,
    "cost_scenario": {
        "c_fp": 1.0,
        "c_fn": 8.0,
        "unit": "clinical_risk_units",
    },
    "actions": {
        "negative": "routine_screening",
        "review": "advanced_ultrasound_review",
        "positive": "urgent_biopsy",
    },
    "random_state": RANDOM_STATE,
}

with open("policy_config.json", "w", encoding="utf-8") as f:
  json.dump(policy_config, f, indent=4, ensure_ascii=False)


# Клас реалізації політики
@dataclass(frozen=True)
class TriageDecisionPolicy:

  threshold_binary: float
  threshold_low: float
  threshold_high: float

  def decide_binary(self, score: float) -> str:
    if not (0.0 <= score <= 1.0):
      raise ValueError("Score must be in [0, 1]")
    return (
        "urgent_biopsy"
        if score >= self.threshold_binary
        else "routine_screening"
    )

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


# 10. ФІНАЛЬНЕ ОЦІНЮВАННЯ НА TEST SET
calibrated_model.fit(X_train, y_train)
test_probs = calibrated_model.predict_proba(X_test)[:, 1]

test_preds = (test_probs >= t_constrained).astype(int)
tn_t, fp_t, fn_t, tp_t = confusion_matrix(
    y_test, test_preds, labels=[0, 1]
).ravel()
prec_t = precision_score(y_test, test_preds, zero_division=0)
rec_t = recall_score(y_test, test_preds, zero_division=0)
f1_t = f1_score(y_test, test_preds, zero_division=0)
brier_t = brier_score_loss(y_test, test_probs)
roc_t = roc_auc_score(y_test, test_probs)
alert_t = (tp_t + fp_t) / len(y_test)
cost_t = (1.0 * fp_t + 8.0 * fn_t) / len(y_test)

def_preds = (test_probs >= 0.5).astype(int)
tn_def, fp_def, fn_def, tp_def = confusion_matrix(
    y_test, def_preds, labels=[0, 1]
).ravel()
cost_def = (1.0 * fp_def + 8.0 * fn_def) / len(y_test)

print(f"Brier score: Початковий={brier_raw:.4f}, Калібрований={brier_cal:.4f}")
print(
    f"Дефолтний поріг t=0.5 (OOF): Prec={df_thresh.loc[df_thresh['threshold'].sub(0.5).abs().idxmin(), 'precision']:.4f}, Rec={df_thresh.loc[df_thresh['threshold'].sub(0.5).abs().idxmin(), 'recall']:.4f}, Cost={df_thresh.loc[df_thresh['threshold'].sub(0.5).abs().idxmin(), f'cost_{sc_target}']:.4f}"
)
print(
    f"Необмежений мінімум t={t_unconstrained:.3f} (OOF): Черга q={unconstrained_row['alert_rate']*100:.2f}%, Cost={unconstrained_row[f'cost_{sc_target}']:.4f}"
)
print(
    f"ОБРАНИЙ робочий поріг t={t_constrained:.3f} (OOF): Черга q={best_constrained_row['alert_rate']*100:.2f}%, Prec={best_constrained_row['precision']:.4f}, Rec={best_constrained_row['recall']:.4f}, Cost={best_constrained_row[f'cost_{sc_target}']:.4f}"
)
print("-" * 50)
print("ФІНАЛЬНИЙ ТЕСТ (D_test):")
print(f"TP={tp_t}, FP={fp_t}, TN={tn_t}, FN={fn_t}")
print(
    f"Precision={prec_t:.4f}, Recall={rec_t:.4f}, F1={f1_t:.4f}, ROC-AUC={roc_t:.4f}"
)
print(
    f"Brier={brier_t:.4f}, Alert Rate (q)={alert_t*100:.2f}%, Cost={cost_t:.4f}"
)
print(
    f"Порівняння вартості з t=0.5 на тесті: Обраний={cost_t:.4f} проти t=0.5={cost_def:.4f} (FN зменшено з {fn_def} до {fn_t})"
)