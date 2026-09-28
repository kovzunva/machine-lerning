import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# 1. Завантаження набору даних (Варіант 6)
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y_original = data.target.copy()

# За замовчуванням у scikit-learn: 0 - malignant, 1 - benign.
# Для медичної задачі позитивним класом (y=1) робимо malignant (злоякісна пухлина):
y = 1 - y_original

print(f"Розмір набору даних: {X.shape}")
print(f"Пропущені значення: {X.isnull().sum().sum()}")
print(f"Дублікати: {X.duplicated().sum()}")
print("\nРозподіл класів (0 - Benign, 1 - Malignant):")
print(y.value_counts().sort_index())
print(f"Частка позитивного класу (Malignant): {y.mean():.4f}")

# Графік розподілу класів (діагностика)
plt.figure(figsize=(6, 4))
plt.bar(['Benign (0)', 'Malignant (1)'], y.value_counts().sort_index(), color=['#2b5c8f', '#d95f02'], edgecolor='black')
plt.ylabel('Кількість спостережень')
plt.title('Розподіл об\'єктів за класами')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_distribution_lr2.png')
plt.close()

# 2. Стратифіковане розділення даних (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Моделі в межах Pipeline
models = {
    'Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'GaussianNB': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ])
}

# 4. 5-fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc'
}

cv_records = []
for name, model in models.items():
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_records.append({
        'Model': name,
        'Accuracy': f"{res['test_accuracy'].mean():.4f} +/- {res['test_accuracy'].std():.4f}",
        'Precision': f"{res['test_precision'].mean():.4f} +/- {res['test_precision'].std():.4f}",
        'Recall': f"{res['test_recall'].mean():.4f} +/- {res['test_recall'].std():.4f}",
        'F1': f"{res['test_f1'].mean():.4f} +/- {res['test_f1'].std():.4f}",
        'ROC-AUC': f"{res['test_roc_auc'].mean():.4f} +/- {res['test_roc_auc'].std():.4f}"
    })

cv_df = pd.DataFrame(cv_records)
print("\n--- Результати 5-fold Stratified Cross-Validation ---")
print(cv_df.to_string(index=False))

# 5. Фінальне навчання Logistic Regression та тест
best_model = models['Logistic Regression']
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- Фінальні метрики Logistic Regression на test set ---")
print(f"Accuracy = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall = {rec:.4f}")
print(f"F1 = {f1:.4f}")
print(f"ROC-AUC = {auc:.4f}")

# 6. Матриця помилок
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Malignant (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion matrix на test set')
plt.tight_layout()
plt.savefig('confusion_matrix_lr2.png')
plt.close()

# 7. Аналіз порогів прийняття рішень
thresholds = [0.3, 0.5, 0.7]
thresh_rows = []
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    thresh_rows.append({
        'Threshold': t,
        'Precision': precision_score(y_test, y_pred_t, zero_division=0),
        'Recall': recall_score(y_test, y_pred_t, zero_division=0),
        'F1': f1_score(y_test, y_pred_t, zero_division=0),
        'FP': fp_t,
        'FN': fn_t
    })

thresh_df = pd.DataFrame(thresh_rows)
print("\n--- Вплив Decision Threshold на помилки та метрики ---")
print(thresh_df.to_string(index=False))

plt.figure(figsize=(7, 5))
plt.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='o', label='Precision')
plt.plot(thresh_df['Threshold'], thresh_df['Recall'], marker='s', label='Recall')
plt.plot(thresh_df['Threshold'], thresh_df['F1'], marker='^', label='F1')
plt.xlabel('Decision threshold')
plt.ylabel('Значення метрики')
plt.title('Вплив threshold на метрики')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('threshold_metrics_lr2.png')
plt.close()

# 8. Побудова ROC-кривої
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.4f})", color='steelblue', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-крива')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_lr2.png')
plt.close()

# 9. Таблиця помилок
errors_df = X_test.copy()
errors_df['actual'] = y_test
errors_df['predicted'] = y_pred
errors_df['probability'] = y_proba
errors_df['error_type'] = np.where(
    (errors_df['actual'] == 0) & (errors_df['predicted'] == 1), 'FP',
    np.where((errors_df['actual'] == 1) & (errors_df['predicted'] == 0), 'FN', 'None')
)
errors_df = errors_df[errors_df['error_type'] != 'None'].copy()
errors_df['distance_to_threshold'] = (errors_df['probability'] - 0.5).abs()

print("\n--- Помилково класифіковані об'єкти ---")
cols_to_show = ['actual', 'predicted', 'probability', 'error_type', 'distance_to_threshold', 'mean radius', 'mean texture', 'mean concavity', 'worst radius']
print(errors_df[cols_to_show].to_string())