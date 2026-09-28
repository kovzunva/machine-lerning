import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

warnings.filterwarnings('ignore')

# 1. Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
    df = pd.read_csv(url, names=columns)
except Exception:
    df = pd.read_csv('car.data', names=columns)

# 2. Формування бінарної цільової змінної: 0 - unacc, 1 - acc/good/vgood
df['target'] = df['class'].apply(lambda x: 0 if x == 'unacc' else 1)

print("Розподіл бінарних класів:")
print(df['target'].value_counts())
print(f"Частка позитивного класу: {df['target'].mean():.4f}")

X = df.drop(columns=['class', 'target'])
y = df['target']

# 3. Стратифіковане розділення даних (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Препроцесинг: OrdinalEncoder для збереження монотонності
categories_order = [
    ['low', 'med', 'high', 'vhigh'],
    ['low', 'med', 'high', 'vhigh'],
    ['2', '3', '4', '5more'],
    ['2', '4', 'more'],
    ['small', 'med', 'big'],
    ['low', 'med', 'high']
]

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=categories_order), X.columns.tolist())
    ]
)

# 5. Побудова моделей
models = {
    'Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'GaussianNB': Pipeline([
        ('prep', preprocessor),
        ('clf', GaussianNB())
    ])
}

# 6. Стратифікована крос-валідація на 5 фолдах
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results_list = []
for name, model in models.items():
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_results_list.append({
        'Model': name,
        'Accuracy': f"{res['test_accuracy'].mean():.4f} +/- {res['test_accuracy'].std():.4f}",
        'Precision': f"{res['test_precision'].mean():.4f} +/- {res['test_precision'].std():.4f}",
        'Recall': f"{res['test_recall'].mean():.4f} +/- {res['test_recall'].std():.4f}",
        'F1': f"{res['test_f1'].mean():.4f} +/- {res['test_f1'].std():.4f}",
        'ROC-AUC': f"{res['test_roc_auc'].mean():.4f} +/- {res['test_roc_auc'].std():.4f}"
    })

cv_df = pd.DataFrame(cv_results_list)
print("\n--- Результати 5-fold Stratified Cross-Validation ---")
print(cv_df.to_string(index=False))

# 7. Навчання та фінальне тестування найкращої моделі (Logistic Regression)
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

# 8. Матриця помилок
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Unacceptable (0)', 'Acceptable (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion matrix на test set')
plt.tight_layout()
plt.savefig('confusion_matrix_lr2.png')
plt.close()

# 9. Дослідження порогів класифікації (thresholds)
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

# Графік залежності метрик від threshold
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

# 10. Побудова ROC-кривої
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

# 11. Таблиця аналізу помилок (FP та FN)
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

print("\n--- Помилково класифіковані об'єкти (перші 10) ---")
print(errors_df[['actual', 'predicted', 'probability', 'error_type', 'distance_to_threshold'] + list(X.columns)].head(10).to_string())