import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Завантаження даних Car Evaluation
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
try:
    df = pd.read_csv(url, names=columns)
except Exception:
    df = pd.read_csv('car.data', names=columns)

# Числове кодування цільової змінної для регресії
target_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
df['target'] = df['class'].map(target_mapping)

X = df.drop(columns=['class', 'target'])
y = df['target'].values

# Порядкове кодування
categories_order = [
    ['low', 'med', 'high', 'vhigh'],
    ['low', 'med', 'high', 'vhigh'],
    ['2', '3', '4', '5more'],
    ['2', '4', 'more'],
    ['small', 'med', 'big'],
    ['low', 'med', 'high']
]
encoder = OrdinalEncoder(categories=categories_order)
X_encoded = encoder.fit_transform(X)

# 2. Розбиття на Train (80%) та Test (20%)
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 3. Шкала складності: d від 1 до 3
degrees = [1, 2, 3]
complexity_rows = []

for d in degrees:
    n_feat = comb(6 + d, d) - 1
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', LinearRegression())
    ])
    res = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )
    train_rmse = -res['train_score'].mean()
    cv_rmse = -res['test_score'].mean()
    cv_std = res['test_score'].std(ddof=1)
    complexity_rows.append({
        'degree': d,
        'n_features': n_feat,
        'train_rmse': train_rmse,
        'cv_rmse': cv_rmse,
        'cv_std': cv_std,
        'gap': cv_rmse - train_rmse
    })

complexity_df = pd.DataFrame(complexity_rows)
print("\n--- ТАБЛИЦЯ 1: Залежність похибки від степеня полінома ---")
print(complexity_df.to_string(index=False))

# Графік 1: Крива валідації складності
plt.figure(figsize=(7, 4.5))
plt.plot(complexity_df['degree'], complexity_df['train_rmse'], marker='o', label='Train RMSE', color='tab:blue')
plt.errorbar(complexity_df['degree'], complexity_df['cv_rmse'], yerr=complexity_df['cv_std'],
             marker='s', capsize=4, label='CV RMSE ± SD', color='tab:orange')
plt.xlabel('Степінь полінома d')
plt.ylabel('RMSE')
plt.title('Крива валідації для поліноміальної складності')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pr3_validation_curve.png')
plt.close()

# Графік 2: Криві навчання для d=1 та d=3
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for idx, d_val in enumerate([1, 3]):
    pipe_lc = Pipeline([
        ('poly', PolynomialFeatures(degree=d_val, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', LinearRegression())
    ])
    train_sizes, train_scores, test_scores = learning_curve(
        pipe_lc, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring='neg_root_mean_squared_error',
        shuffle=True, random_state=RANDOM_STATE
    )
    axes[idx].plot(train_sizes, -train_scores.mean(axis=1), marker='o', label='Train RMSE', color='tab:blue')
    axes[idx].plot(train_sizes, -test_scores.mean(axis=1), marker='s', label='CV RMSE', color='tab:orange')
    axes[idx].fill_between(train_sizes, -test_scores.mean(axis=1) - test_scores.std(axis=1),
                           -test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15, color='tab:orange')
    axes[idx].set_title(f'Крива навчання (d = {d_val})')
    axes[idx].set_xlabel('Кількість навчальних об\'єктів')
    axes[idx].set_ylabel('RMSE')
    axes[idx].grid(alpha=0.3)
    axes[idx].legend()
plt.tight_layout()
plt.savefig('pr3_learning_curves.png')
plt.close()

# 4. Регуляризація для складного простору d=3
TARGET_DEGREE = 3
alphas = np.logspace(-4, 2, 13)
l1_ratios = [0.2, 0.5, 0.8]

def evaluate_regularized(model_cls, kwargs_dict):
    records = []
    for a in alphas:
        cur_kwargs = kwargs_dict.copy()
        cur_kwargs['alpha'] = a
        pipe_reg = Pipeline([
            ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
            ('scale', StandardScaler()),
            ('model', model_cls(**cur_kwargs))
        ])
        cv_res = cross_validate(
            pipe_reg, X_train, y_train, cv=cv,
            scoring='neg_root_mean_squared_error',
            return_train_score=True
        )
        t_rmse = -cv_res['train_score'].mean()
        c_rmse = -cv_res['test_score'].mean()
        c_std = cv_res['test_score'].std(ddof=1)
        records.append({'alpha': a, 'train_rmse': t_rmse, 'cv_rmse': c_rmse, 'cv_std': c_std, 'gap': c_rmse - t_rmse})
    df_rec = pd.DataFrame(records)
    best_row = df_rec.loc[df_rec['cv_rmse'].idxmin()]
    return df_rec, best_row

ridge_df, ridge_best = evaluate_regularized(Ridge, {'random_state': RANDOM_STATE})
lasso_df, lasso_best = evaluate_regularized(Lasso, {'random_state': RANDOM_STATE, 'max_iter': 10000, 'tol': 1e-3})

# ElasticNet
en_records = []
en_best_row = None
best_en_score = float('inf')
for l1_r in l1_ratios:
    for a in alphas:
        pipe_en = Pipeline([
            ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
            ('scale', StandardScaler()),
            ('model', ElasticNet(alpha=a, l1_ratio=l1_r, random_state=RANDOM_STATE, max_iter=10000, tol=1e-3))
        ])
        cv_res = cross_validate(
            pipe_en, X_train, y_train, cv=cv,
            scoring='neg_root_mean_squared_error',
            return_train_score=True
        )
        t_rmse = -cv_res['train_score'].mean()
        c_rmse = -cv_res['test_score'].mean()
        c_std = cv_res['test_score'].std(ddof=1)
        row_dict = {'alpha': a, 'l1_ratio': l1_r, 'train_rmse': t_rmse, 'cv_rmse': c_rmse, 'cv_std': c_std, 'gap': c_rmse - t_rmse}
        en_records.append(row_dict)
        if c_rmse < best_en_score:
            best_en_score = c_rmse
            en_best_row = row_dict

en_df = pd.DataFrame(en_records)

# Норми коефіцієнтів
def get_coef_metrics(estimator):
    pipe_fit = Pipeline([
        ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', estimator)
    ])
    pipe_fit.fit(X_train, y_train)
    coef = pipe_fit.named_steps['model'].coef_
    l1_norm = np.sum(np.abs(coef))
    l2_norm = np.linalg.norm(coef)
    n_nonzero = np.sum(np.abs(coef) > 1e-5)
    return l1_norm, l2_norm, n_nonzero

r_l1, r_l2, r_nz = get_coef_metrics(Ridge(alpha=ridge_best['alpha'], random_state=RANDOM_STATE))
l_l1, l_l2, l_nz = get_coef_metrics(Lasso(alpha=lasso_best['alpha'], random_state=RANDOM_STATE, max_iter=10000, tol=1e-3))
en_l1, en_l2, en_nz = get_coef_metrics(ElasticNet(alpha=en_best_row['alpha'], l1_ratio=en_best_row['l1_ratio'], random_state=RANDOM_STATE, max_iter=10000, tol=1e-3))
lin_l1, lin_l2, lin_nz = get_coef_metrics(LinearRegression())

reg_summary = pd.DataFrame([
    {'Model': 'Linear Regression (d=3)', 'alpha': '-', 'l1_ratio': '-', 'train_rmse': complexity_df.loc[2, 'train_rmse'],
     'cv_rmse': complexity_df.loc[2, 'cv_rmse'], 'cv_std': complexity_df.loc[2, 'cv_std'], 'gap': complexity_df.loc[2, 'gap'], 'N_nonzero': lin_nz, 'L2_norm': lin_l2},
    {'Model': 'Ridge (d=3)', 'alpha': f"{ridge_best['alpha']:.4f}", 'l1_ratio': '-', 'train_rmse': ridge_best['train_rmse'],
     'cv_rmse': ridge_best['cv_rmse'], 'cv_std': ridge_best['cv_std'], 'gap': ridge_best['gap'], 'N_nonzero': r_nz, 'L2_norm': r_l2},
    {'Model': 'Lasso (d=3)', 'alpha': f"{lasso_best['alpha']:.4f}", 'l1_ratio': '-', 'train_rmse': lasso_best['train_rmse'],
     'cv_rmse': lasso_best['cv_rmse'], 'cv_std': lasso_best['cv_std'], 'gap': lasso_best['gap'], 'N_nonzero': l_nz, 'L2_norm': l_l2},
    {'Model': 'Elastic Net (d=3)', 'alpha': f"{en_best_row['alpha']:.4f}", 'l1_ratio': en_best_row['l1_ratio'], 'train_rmse': en_best_row['train_rmse'],
     'cv_rmse': en_best_row['cv_rmse'], 'cv_std': en_best_row['cv_std'], 'gap': en_best_row['gap'], 'N_nonzero': en_nz, 'L2_norm': en_l2}
])

print("\n--- ТАБЛИЦЯ 2: Порівняння моделей для простору d=3 ---")
print(reg_summary.to_string(index=False))

# Графік 3: Порівняння моделей за CV RMSE
plt.figure(figsize=(7, 4))
plt.barh(reg_summary['Model'], reg_summary['cv_rmse'], xerr=reg_summary['cv_std'], color=['#e66101', '#5e3c99', '#b2abd2', '#fdb863'], capsize=4)
plt.xlabel('CV RMSE (менше — краще)')
plt.title('Порівняння регуляризованих моделей (d=3)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('pr3_best_models.png')
plt.close()

# Графік 4: Залежність похибок від сили регуляризації
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, df_curr, title in zip(axes, [ridge_df, lasso_df, en_df[en_df['l1_ratio'] == en_best_row['l1_ratio']]],
                             ['Ridge', 'Lasso', f'ElasticNet (l1={en_best_row["l1_ratio"]})']):
    ax.plot(df_curr['alpha'], df_curr['train_rmse'], label='Train RMSE', color='tab:blue')
    ax.plot(df_curr['alpha'], df_curr['cv_rmse'], label='CV RMSE', color='tab:orange')
    ax.set_xscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig('pr3_alpha_vs_rmse.png')
plt.close()

# Графік 5: Регуляризаційні шляхи коефіцієнтів
pipe_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
    ('scale', StandardScaler())
])
X_poly_tr = pipe_poly.fit_transform(X_train)

ridge_coefs = [Ridge(alpha=a, random_state=RANDOM_STATE).fit(X_poly_tr, y_train).coef_ for a in alphas]
lasso_coefs = [Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=10000, tol=1e-3).fit(X_poly_tr, y_train).coef_ for a in alphas]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(alphas, ridge_coefs)
axes[0].set_xscale('log')
axes[0].set_xlabel('alpha')
axes[0].set_ylabel('Стандартизовані коефіцієнти')
axes[0].set_title('Регуляризаційні шляхи Ridge')
axes[0].grid(alpha=0.3)

axes[1].plot(alphas, lasso_coefs)
axes[1].set_xscale('log')
axes[1].set_xlabel('alpha')
axes[1].set_ylabel('Стандартизовані коефіцієнти')
axes[1].set_title('Регуляризаційні шляхи Lasso')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pr3_regularization_paths.png')
plt.close()

# 5. Перевірка стійкості (5 різних розбиттів)
seeds = [3, 11, 21, 42, 77]
stability_records = {'Ridge': [], 'Lasso': []}

for s in seeds:
    cv_s = KFold(n_splits=5, shuffle=True, random_state=s)
    r_scores = [(-cross_validate(Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
                                           ('scale', StandardScaler()),
                                           ('model', Ridge(alpha=a, random_state=s))]),
                                 X_train, y_train, cv=cv_s, scoring='neg_root_mean_squared_error')['test_score'].mean(), a) for a in alphas]
    best_r_score, best_r_a = min(r_scores, key=lambda x: x[0])
    stability_records['Ridge'].append({'alpha': best_r_a, 'score': best_r_score})
    
    l_scores = [(-cross_validate(Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
                                           ('scale', StandardScaler()),
                                           ('model', Lasso(alpha=a, random_state=s, max_iter=10000, tol=1e-3))]),
                                 X_train, y_train, cv=cv_s, scoring='neg_root_mean_squared_error')['test_score'].mean(), a) for a in alphas]
    best_l_score, best_l_a = min(l_scores, key=lambda x: x[0])
    stability_records['Lasso'].append({'alpha': best_l_a, 'score': best_l_score})

stab_summary = []
for m_name in ['Ridge', 'Lasso']:
    alphas_chosen = [item['alpha'] for item in stability_records[m_name]]
    scores_chosen = [item['score'] for item in stability_records[m_name]]
    stab_summary.append({
        'Model': m_name,
        'alpha_min': min(alphas_chosen),
        'alpha_max': max(alphas_chosen),
        'Mean CV RMSE': np.mean(scores_chosen),
        'SD': np.std(scores_chosen, ddof=1)
    })
print("\n--- ТАБЛИЦЯ 3: Стійкість вибору регуляризованих моделей ---")
print(pd.DataFrame(stab_summary).to_string(index=False))

# 6. Фінальне оцінювання на test set
final_models = {
    'Linear Regression (d=2)': Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scale', StandardScaler()), ('model', LinearRegression())]),
    'Linear Regression (d=3)': Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)), ('scale', StandardScaler()), ('model', LinearRegression())]),
    'Ridge (d=3)': Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)), ('scale', StandardScaler()), ('model', Ridge(alpha=ridge_best['alpha'], random_state=RANDOM_STATE))])
}

test_records = []
for name, m in final_models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    test_records.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    })
print("\n--- ТАБЛИЦЯ 4: Контрольні результати на test set ---")
print(pd.DataFrame(test_records).to_string(index=False))