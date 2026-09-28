import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
    df = pd.read_csv(url, names=columns)
except Exception:
    df = pd.read_csv('car.data', names=columns)

# Переведення категоріальної мітки в числовий рейтинг для регресії
target_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
df['target'] = df['class'].map(target_mapping)

X = df.drop(columns=['class', 'target'])
y = df['target']

# 2. Формування train та test вибірок
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Налаштування препроцесингу
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

# 4. Базова та кандидатні моделі
models = {
    'Baseline': DummyRegressor(strategy='mean'),
    'Linear Regression': Pipeline([
        ('prep', preprocessor),
        ('reg', LinearRegression())
    ]),
    'Ridge (d=2)': Pipeline([
        ('prep', preprocessor),
        ('poly', PolynomialFeatures(degree=2)),
        ('reg', Ridge(alpha=1.0))
    ]),
    'Decision Tree': Pipeline([
        ('prep', preprocessor),
        ('reg', DecisionTreeRegressor(random_state=42))
    ])
}

# 5. Крос-валідація
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_records = []

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error'
    )
    rmse_scores = -scores
    cv_records.append({
        'Model': name,
        'Mean RMSE': rmse_scores.mean(),
        'Std RMSE': rmse_scores.std()
    })

results_df = pd.DataFrame(cv_records)
print("Результати 5-fold cross-validation:")
print(results_df.to_string(index=False))

# 6. Графік порівняння моделей
plt.figure(figsize=(8, 5))
plt.bar(
    results_df['Model'],
    results_df['Mean RMSE'],
    yerr=results_df['Std RMSE'],
    capsize=5,
    color='steelblue',
    edgecolor='black'
)
plt.ylabel('RMSE')
plt.title('Порівняння моделей за 5-fold cross-validation')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('models_cv_comparison.png')
plt.close()

# 7. Фінальне навчання обраної найкращої моделі (Decision Tree) та тест
best_model_name = 'Decision Tree'
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

y_pred_test = best_model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"\nФінальні результати на test set для {best_model_name}:")
print(f"MAE = {mae_test:.4f}")
print(f"RMSE = {rmse_test:.4f}")
print(f"R2 = {r2_test:.4f}")

# 8. Графік фактичних та передбачених значень
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_test, alpha=0.35, color='darkblue')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.xlabel('Фактичні значення')
plt.ylabel('Передбачені значення')
plt.title('Фактичні та передбачені значення на test set')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted_pr2.png')
plt.close()