import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# 1. Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
    df = pd.read_csv(url, names=columns)
except Exception:
    df = pd.read_csv('car.data', names=columns)

# 2. Первинний аудит даних
print("--- Розмірність таблиці ---")
print(df.shape)

print("\n--- Інформація про стовпці та типи ---")
print(df.info())

print("\n--- Кількість унікальних значень у кожній ознаці ---")
print(df.nunique())

print("\n--- Перевірка на пропущені значення ---")
print(df.isnull().sum())

print("\n--- Кількість повних дублікатів ---")
print(df.duplicated().sum())

print("\n--- Розподіл значень за кожною ознакою ---")
for col in df.columns:
    print(f"\nРозподіл для '{col}':")
    print(df[col].value_counts())

# 3. Візуалізація розподілів
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 3.1. Розподіл цільової змінної
plt.figure(figsize=(7, 4))
order_target = ['unacc', 'acc', 'good', 'vgood']
sns.countplot(data=df, x='class', order=order_target, palette='viridis')
plt.title('Розподіл цільової змінної')
plt.xlabel('Клас прийнятності')
plt.ylabel('Кількість')
plt.show()

# 3.2. Взаємозв'язок цільової ознаки з рівнем безпеки
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='safety', hue='class', hue_order=order_target, palette='viridis')
plt.title("Залежність прийнятності від рівня безпеки")
plt.xlabel("Безпека")
plt.ylabel("Кількість")
plt.legend(title="Клас")
plt.show()

# 3.3. Взаємозв'язок цільової ознаки з ціною купівлі
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='buying', hue='class', hue_order=order_target, palette='viridis')
plt.title("Залежність прийнятності від ціни купівлі")
plt.xlabel("Ціна купівлі")
plt.ylabel("Кількість")
plt.legend(title="Клас")
plt.show()

# 4. Кореляційний аналіз
ordinal_mapping = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
    'persons': {'2': 2, '4': 4, 'more': 6},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2},
    'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
}

df_encoded = df.copy()
for col, mapping in ordinal_mapping.items():
    df_encoded[col] = df_encoded[col].map(mapping)

corr_matrix = df_encoded.corr(method='spearman')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Кореляційна матриця Спірмена")
plt.show()

# 5. Побудова відтворюваного Preprocessing Pipeline
X = df.drop(columns=['class'])
y = df['class']

# Розділення на train та test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Визначення порядку категорій для OrdinalEncoder
categories_order = [
    ['low', 'med', 'high', 'vhigh'],     # buying
    ['low', 'med', 'high', 'vhigh'],     # maint
    ['2', '3', '4', '5more'],             # doors
    ['2', '4', 'more'],                  # persons
    ['small', 'med', 'big'],             # lug_boot
    ['low', 'med', 'high']               # safety
]

# Створення пайплайна
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=categories_order), X.columns.tolist())
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_train_proc = pipeline.fit_transform(X_train)
X_test_proc = pipeline.transform(X_test)

# 6. Перевірка підготовлених даних
print("\n--- Результати перевірки підготовленого набору даних ---")
print(f"Розмірність X_train після обробки: {X_train_proc.shape}")
print(f"Розмірність X_test після обробки: {X_test_proc.shape}")
print(f"Кількість пропусків у тренувальній вибірці: {np.isnan(X_train_proc).sum()}")
print(f"Кількість пропусків у тестовій вибірці: {np.isnan(X_test_proc).sum()}")
print(f"Перші 3 рядки трансформованого X_train:\n{X_train_proc[:3]}")