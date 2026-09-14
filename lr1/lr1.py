import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Завантаження набору даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
    df = pd.read_csv(url, names=columns)
except Exception:
    df = pd.read_csv('car.data', names=columns)

# Переведення порядкової цільової ознаки у числовий бал для задачі регресії
target_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
df['target'] = df['class'].map(target_mapping)

X = df.drop(columns=['class', 'target'])
y = df['target']

# Візуалізація розподілу цільової величини
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', hue='target', palette='Blues', legend=False)
plt.title('Розподіл цільової змінної')
plt.xlabel('Числовий бал прийнятності')
plt.ylabel('Кількість об`єктів')
plt.tight_layout()
plt.savefig('target_distribution.png')
plt.close()

# Розділення вибірки на тренувальну та тестову
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Попередня обробка категоріальних порядкових ознак
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

# Формування базової та кандидатних моделей
models = {
    'Baseline': DummyRegressor(strategy='mean'),
    'Linear Regression': Pipeline([
        ('prep', preprocessor),
        ('reg', LinearRegression())
    ]),
    'Polynomial Regression (d=2)': Pipeline([
        ('prep', preprocessor),
        ('poly', PolynomialFeatures(degree=2)),
        ('reg', LinearRegression())
    ]),
    'Ridge (d=2)': Pipeline([
        ('prep', preprocessor),
        ('poly', PolynomialFeatures(degree=2)),
        ('reg', Ridge(alpha=1.0))
    ])
}

# Крос-валідація на навчальній вибірці
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error'
    )
    rmse_scores = -scores
    cv_results[name] = (rmse_scores.mean(), rmse_scores.std())
    print(f"{name}: CV RMSE = {rmse_scores.mean():.4f} +/- {rmse_scores.std():.4f}")

# Фінальне навчання найкращої моделі та оцінка на тесті
best_model = models['Ridge (d=2)']
best_model.fit(X_train, y_train)

y_pred_test = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"\nТестові метрики: MAE = {mae:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

# Оцінка базової моделі на тесті
baseline = models['Baseline']
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
print(f"Базова модель на тесті: RMSE = {rmse_base:.4f}")

# Графік відповідності фактичних та передбачених значень
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([0, 3], [0, 3], color='red', linestyle='--')
plt.title('Фактичні та передбачені значення')
plt.xlabel('Фактичні значення')
plt.ylabel('Передбачені значення')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Графік залишків
residuals = y_test - y_pred_test
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_test, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title('Графік залишків')
plt.xlabel('Передбачені значення')
plt.ylabel('Залишки')
plt.tight_layout()
plt.savefig('residuals.png')
plt.close()

# Вивід спостережень із найбільшими похибками
errors_df = X_test.copy()
errors_df['true'] = y_test
errors_df['pred'] = y_pred_test
errors_df['abs_error'] = np.abs(residuals)
worst_five = errors_df.sort_values(by='abs_error', ascending=False).head(5)
print("\nНайбільші похибки:")
print(worst_five[['buying', 'safety', 'persons', 'true', 'pred', 'abs_error']])