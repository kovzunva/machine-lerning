import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
import time
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
try:
    df = pd.read_csv(url, names=columns)
except Exception:
    df = pd.read_csv('car.data', names=columns)

target_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
df['target'] = df['class'].map(target_mapping)
X = df.drop(columns=['class', 'target'])
y = df['target'].values

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

RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE
)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 2. Шкала складності з 5 рівнів (d=1, 2, 3 та interaction-only для d=4, 5)
complexity_rows = []
configs = [
    (1, False, "d=1"),
    (2, False, "d=2"),
    (3, False, "d=3"),
    (4, True, "d=4 (inter)"),
    (5, True, "d=5 (inter)")
]

for d_val, inter_flag, label in configs:
    t0 = time.perf_counter()
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d_val, interaction_only=inter_flag, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', LinearRegression())
    ])
    res = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )
    cv_time = (time.perf_counter() - t0) * 1000
    
    # Кількість ознак
    pipe.fit(X_train[:10], y_train[:10])
    n_feats = pipe.named_steps['poly'].n_output_features_
    
    train_rmse = -res['train_score'].mean()
    cv_rmse = -res['test_score'].mean()
    cv_std = res['test_score'].std(ddof=1)
    cv_se = cv_std / np.sqrt(5)
    
    complexity_rows.append({
        'Config': label,
        'n_features': n_feats,
        'train_rmse': train_rmse,
        'cv_rmse': cv_rmse,
        'cv_std': cv_std,
        'cv_se': cv_se,
        'gap': cv_rmse - train_rmse,
        'cv_time_ms': cv_time
    })

df_comp = pd.DataFrame(complexity_rows)
print("\n--- ТАБЛИЦЯ 1: Шкала складності (5 рівнів) ---")
print(df_comp.to_string(index=False))

# Графік 1: Validation curve
plt.figure(figsize=(7.5, 4.5))
plt.plot(range(len(configs)), df_comp['train_rmse'], marker='o', label='Train RMSE', color='tab:blue')
plt.errorbar(range(len(configs)), df_comp['cv_rmse'], yerr=df_comp['cv_std'],
             marker='s', capsize=4, label='CV RMSE ± SD', color='tab:orange')
plt.xticks(range(len(configs)), [c[2] for c in configs])
plt.xlabel('Рівень складності моделі')
plt.ylabel('RMSE')
plt.title('Крива валідації для контрольованої шкали складності')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pr3_validation_curve.png')
plt.close()

# Графік 2: Learning curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for idx, (d_val, inter_flag, label) in enumerate([(1, False, 'd=1 (Недонавчання)'), (3, False, 'd=3 (Підвищена дисперсія)')]):
    pipe_lc = Pipeline([
        ('poly', PolynomialFeatures(degree=d_val, interaction_only=inter_flag, include_bias=False)),
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
    axes[idx].set_title(f'Крива навчання: {label}')
    axes[idx].set_xlabel('Кількість навчальних об\'єктів')
    axes[idx].set_ylabel('RMSE')
    axes[idx].grid(alpha=0.3)
    axes[idx].legend()
plt.tight_layout()
plt.savefig('pr3_learning_curves.png')
plt.close()

# 3. Регуляризація на просторі d=3 (83 ознаки) з розширеною сіткою alpha
TARGET_DEGREE = 3
alphas = np.logspace(-6, 2, 17)  # Розширено вниз до 1e-6
l1_ratios = [0.2, 0.5, 0.8]

def evaluate_reg(model_cls, kwargs):
    records = []
    for a in alphas:
        cur_kw = kwargs.copy()
        cur_kw['alpha'] = a
        pipe_r = Pipeline([
            ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
            ('scale', StandardScaler()),
            ('model', model_cls(**cur_kw))
        ])
        res = cross_validate(pipe_r, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', return_train_score=True)
        t_r = -res['train_score'].mean()
        c_r = -res['test_score'].mean()
        c_sd = res['test_score'].std(ddof=1)
        records.append({'alpha': a, 'train_rmse': t_r, 'cv_rmse': c_r, 'cv_std': c_sd, 'gap': c_r - t_r})
    df_res = pd.DataFrame(records)
    best_r = df_res.loc[df_res['cv_rmse'].idxmin()]
    return df_res, best_r

ridge_df, ridge_best = evaluate_reg(Ridge, {'random_state': RANDOM_STATE})
lasso_df, lasso_best = evaluate_reg(Lasso, {'random_state': RANDOM_STATE, 'max_iter': 20000, 'tol': 1e-3})

en_records = []
best_en_score = float('inf')
en_best_r = None
for l1_r in l1_ratios:
    for a in alphas:
        pipe_en = Pipeline([
            ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
            ('scale', StandardScaler()),
            ('model', ElasticNet(alpha=a, l1_ratio=l1_r, random_state=RANDOM_STATE, max_iter=20000, tol=1e-3))
        ])
        res = cross_validate(pipe_en, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', return_train_score=True)
        t_r = -res['train_score'].mean()
        c_r = -res['test_score'].mean()
        c_sd = res['test_score'].std(ddof=1)
        row = {'alpha': a, 'l1_ratio': l1_r, 'train_rmse': t_r, 'cv_rmse': c_r, 'cv_std': c_sd, 'gap': c_r - t_r}
        en_records.append(row)
        if c_r < best_en_score:
            best_en_score = c_r
            en_best_r = row
en_df = pd.DataFrame(en_records)

def get_norms(estimator):
    p = Pipeline([
        ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', estimator)
    ])
    p.fit(X_train, y_train)
    c = p.named_steps['model'].coef_
    return np.sum(np.abs(c)), np.linalg.norm(c), np.sum(np.abs(c) > 1e-5)

lin_l1, lin_l2, lin_nz = get_norms(LinearRegression())
r_l1, r_l2, r_nz = get_norms(Ridge(alpha=ridge_best['alpha'], random_state=RANDOM_STATE))
l_l1, l_l2, l_nz = get_norms(Lasso(alpha=lasso_best['alpha'], random_state=RANDOM_STATE, max_iter=20000, tol=1e-3))
en_l1, en_l2, en_nz = get_norms(ElasticNet(alpha=en_best_r['alpha'], l1_ratio=en_best_r['l1_ratio'], random_state=RANDOM_STATE, max_iter=20000, tol=1e-3))

reg_table = pd.DataFrame([
    {'Model': 'Linear Regression (d=3)', 'alpha': '-', 'l1_ratio': '-', 'train_rmse': df_comp.loc[2, 'train_rmse'],
     'cv_rmse': df_comp.loc[2, 'cv_rmse'], 'cv_std': df_comp.loc[2, 'cv_std'], 'gap': df_comp.loc[2, 'gap'], 'L1_norm': lin_l1, 'L2_norm': lin_l2, 'N_nonzero': lin_nz},
    {'Model': 'Ridge (d=3)', 'alpha': f"{ridge_best['alpha']:.6f}", 'l1_ratio': '-', 'train_rmse': ridge_best['train_rmse'],
     'cv_rmse': ridge_best['cv_rmse'], 'cv_std': ridge_best['cv_std'], 'gap': ridge_best['gap'], 'L1_norm': r_l1, 'L2_norm': r_l2, 'N_nonzero': r_nz},
    {'Model': 'Lasso (d=3)', 'alpha': f"{lasso_best['alpha']:.6f}", 'l1_ratio': '-', 'train_rmse': lasso_best['train_rmse'],
     'cv_rmse': lasso_best['cv_rmse'], 'cv_std': lasso_best['cv_std'], 'gap': lasso_best['gap'], 'L1_norm': l_l1, 'L2_norm': l_l2, 'N_nonzero': l_nz},
    {'Model': 'Elastic Net (d=3)', 'alpha': f"{en_best_r['alpha']:.6f}", 'l1_ratio': en_best_r['l1_ratio'], 'train_rmse': en_best_r['train_rmse'],
     'cv_rmse': en_best_r['cv_rmse'], 'cv_std': en_best_r['cv_std'], 'gap': en_best_r['gap'], 'L1_norm': en_l1, 'L2_norm': en_l2, 'N_nonzero': en_nz}
])
print("\n--- ТАБЛИЦЯ 2: Порівняння регуляризованих моделей ---")
print(reg_table.to_string(index=False))

# Графіки регуляризації
plt.figure(figsize=(7, 4))
plt.barh(reg_table['Model'], reg_table['cv_rmse'], xerr=reg_table['cv_std'], color=['#e66101', '#5e3c99', '#b2abd2', '#fdb863'], capsize=4)
plt.xlabel('CV RMSE (менше — краще)')
plt.title('Порівняння моделей за CV RMSE')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('pr3_best_models.png')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, d_c, tit in zip(axes, [ridge_df, lasso_df, en_df[en_df['l1_ratio'] == en_best_r['l1_ratio']]],
                        ['Ridge', 'Lasso', f"ElasticNet (l1={en_best_r['l1_ratio']})"]):
    ax.plot(d_c['alpha'], d_c['train_rmse'], label='Train RMSE', color='tab:blue')
    ax.plot(d_c['alpha'], d_c['cv_rmse'], label='CV RMSE', color='tab:orange')
    ax.set_xscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('RMSE')
    ax.set_title(tit)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig('pr3_alpha_vs_rmse.png')
plt.close()

# Регуляризаційні шляхи
pipe_poly = Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)), ('scale', StandardScaler())])
X_poly_tr = pipe_poly.fit_transform(X_train)
ridge_coefs = [Ridge(alpha=a, random_state=RANDOM_STATE).fit(X_poly_tr, y_train).coef_ for a in alphas]
lasso_coefs = [Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=20000, tol=1e-3).fit(X_poly_tr, y_train).coef_ for a in alphas]

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

# 4. Стійкість вибору (5 seeds)
seeds = [3, 11, 21, 42, 77]
stab_data = {'Ridge': [], 'Lasso': [], 'ElasticNet': []}

for s in seeds:
    cv_s = KFold(n_splits=5, shuffle=True, random_state=s)
    
    # Ridge
    r_sc = [(-cross_validate(Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
                                       ('scale', StandardScaler()),
                                       ('model', Ridge(alpha=a, random_state=s))]),
                             X_train, y_train, cv=cv_s, scoring='neg_root_mean_squared_error')['test_score'].mean(), a) for a in alphas]
    best_r_sc, best_r_a = min(r_sc, key=lambda x: x[0])
    # fit for norms
    p_r = Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)), ('scale', StandardScaler()), ('model', Ridge(alpha=best_r_a, random_state=s))]).fit(X_train, y_train)
    stab_data['Ridge'].append({'alpha': best_r_a, 'score': best_r_sc, 'l2': np.linalg.norm(p_r.named_steps['model'].coef_)})
    
    # Lasso
    l_sc = [(-cross_validate(Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
                                       ('scale', StandardScaler()),
                                       ('model', Lasso(alpha=a, random_state=s, max_iter=20000, tol=1e-3))]),
                             X_train, y_train, cv=cv_s, scoring='neg_root_mean_squared_error')['test_score'].mean(), a) for a in alphas]
    best_l_sc, best_l_a = min(l_sc, key=lambda x: x[0])
    p_l = Pipeline([('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)), ('scale', StandardScaler()), ('model', Lasso(alpha=best_l_a, random_state=s, max_iter=20000, tol=1e-3))]).fit(X_train, y_train)
    stab_data['Lasso'].append({'alpha': best_l_a, 'score': best_l_sc, 'l2': np.linalg.norm(p_l.named_steps['model'].coef_), 'nz': np.sum(np.abs(p_l.named_steps['model'].coef_) > 1e-5)})

stab_res = []
for m_name in ['Ridge', 'Lasso']:
    alphas_c = [it['alpha'] for it in stab_data[m_name]]
    scores_c = [it['score'] for it in stab_data[m_name]]
    l2_c = [it['l2'] for it in stab_data[m_name]]
    stab_res.append({
        'Model': m_name,
        'alpha_min': min(alphas_c),
        'alpha_max': max(alphas_c),
        'Mean CV RMSE': np.mean(scores_c),
        'SD': np.std(scores_c, ddof=1),
        'L2_norm_mean': np.mean(l2_c)
    })
print("\n--- ТАБЛИЦЯ 3: Стійкість вибору гіперпараметрів ---")
print(pd.DataFrame(stab_res).to_string(index=False))

# 5. Фінальне оцінювання НА TEST SET
final_model = Pipeline([
    ('poly', PolynomialFeatures(degree=TARGET_DEGREE, include_bias=False)),
    ('scale', StandardScaler()),
    ('model', Lasso(alpha=lasso_best['alpha'], random_state=RANDOM_STATE, max_iter=20000, tol=1e-3))
])
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

test_metrics = {
    'Model': f"Lasso (d=3, alpha={lasso_best['alpha']:.6f})",
    'MAE': mean_absolute_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'R2': r2_score(y_test, y_pred)
}
print("\n--- ТАБЛИЦЯ 4: Фінальне оцінювання фіксованої моделі на test set ---")
print(pd.DataFrame([test_metrics]).to_string(index=False))