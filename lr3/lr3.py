import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import pairwise_distances

RANDOM_STATE = 42

def main():
    # ==========================================
    # 1. КОНТРОЛЬНИЙ ГЕОМЕТРИЧНИЙ ЕКСПЕРИМЕНТ (make_moons)
    # ==========================================
    print("1. Виконання геометричного експерименту на make_moons...")
    X_m, y_m = make_moons(n_samples=520, noise=0.22, random_state=RANDOM_STATE)
    X_m[:, 1] *= 18.0  # Штучне спотворення масштабу другої ознаки у 18 разів

    X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(
        X_m, y_m, test_size=0.28, stratify=y_m, random_state=RANDOM_STATE
    )

    models_moons = {
        'kNN (k=9)': KNeighborsClassifier(n_neighbors=9),
        'Linear SVM': SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x_min, x_max = X_m[:, 0].min() - 0.5, X_m[:, 0].max() + 0.5
    y_min, y_max = X_m[:, 1].min() - 5.0, X_m[:, 1].max() + 5.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    for col_idx, (m_name, base_clf) in enumerate(models_moons.items()):
        for row_idx, (s_name, scaler) in enumerate([('Без масштабування', None), ('StandardScaler', StandardScaler())]):
            if scaler:
                pipe = Pipeline([('scaler', scaler), ('clf', base_clf)])
            else:
                pipe = Pipeline([('clf', base_clf)])
            
            pipe.fit(X_m_tr, y_m_tr)
            f1_te = f1_score(y_m_te, pipe.predict(X_m_te))
            
            Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax = axes[row_idx, col_idx]
            ax.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')
            ax.scatter(X_m_tr[y_m_tr == 0, 0], X_m_tr[y_m_tr == 0, 1], c='royalblue', s=18, label='Клас 0', alpha=0.7)
            ax.scatter(X_m_tr[y_m_tr == 1, 0], X_m_tr[y_m_tr == 1, 1], c='darkorange', s=18, label='Клас 1', alpha=0.7)
            
            clf_step = pipe.named_steps['clf']
            if hasattr(clf_step, 'support_vectors_'):
                if scaler:
                    sv_orig = pipe.named_steps['scaler'].inverse_transform(clf_step.support_vectors_)
                else:
                    sv_orig = clf_step.support_vectors_
                ax.scatter(sv_orig[:, 0], sv_orig[:, 1], s=55, facecolors='none', edgecolors='purple', linewidths=1.2, label='Support Vectors')
                
            ax.set_title(f"{m_name} ({s_name})\nTest F1 = {f1_te:.3f}")
            ax.set_xlabel("Ознака 1")
            ax.set_ylabel("Ознака 2")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('lr3_fig1_moons.png')
    plt.close()

    # ==========================================
    # 2. ПІДГОТОВКА ІНДИВІДУАЛЬНОГО ДАТАСЕТУ (Breast Cancer Wisconsin)
    # ==========================================
    print("\n2. Підготовка Breast Cancer Wisconsin (Варіант 6)...")
    cancer_data = load_breast_cancer(as_frame=True)
    X = cancer_data.data.copy()
    y = 1 - cancer_data.target.copy()  # y=1: Malignant, y=0: Benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ==========================================
    # 3. ДОСЛІДЖЕННЯ МАСШТАБУВАННЯ
    # ==========================================
    print("3. Порівняння способів масштабування...")
    scalers = {
        'Без масштабування': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    base_models = {
        'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Linear SVM': SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    }

    scaling_rows = []
    for m_name, model in base_models.items():
        for s_name, sc in scalers.items():
            steps = [('scaler', sc)] if sc else []
            steps.append(('clf', model))
            pipe = Pipeline(steps)
            res = cross_validate(pipe, X_train, y_train, cv=cv, scoring='f1')
            scaling_rows.append({
                'Модель': m_name,
                'Масштабування': s_name,
                'CV F1': res['test_score'].mean(),
                'SD': res['test_score'].std(ddof=1)
            })

    df_scaling = pd.DataFrame(scaling_rows)
    print("\n--- ТАБЛИЦЯ 1: Вплив масштабування на CV F1 ---")
    print(df_scaling.to_string(index=False))

    plt.figure(figsize=(8, 4.5))
    pivot_sc = df_scaling.pivot(index='Модель', columns='Масштабування', values='CV F1')
    pivot_sc.plot(kind='bar', figsize=(8, 4.5), colormap='viridis', edgecolor='black')
    plt.title('Порівняння способів масштабування ознак')
    plt.ylabel('CV F1-score')
    plt.ylim(0.85, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('lr3_fig2_scaling.png')
    plt.close()

    # ==========================================
    # 4. НАЛАШТУВАННЯ МЕТРИЧНИХ КЛАСИФІКАТОРІВ (kNN)
    # ==========================================
    print("\n4. Налаштування kNN...")
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])

    param_grid_knn = {
        'clf__n_neighbors': [1, 3, 5, 7, 11],
        'clf__p': [1, 2],
        'clf__weights': ['uniform', 'distance']
    }

    grid_knn = GridSearchCV(knn_pipe, param_grid_knn, cv=cv, scoring='f1', n_jobs=1)
    grid_knn.fit(X_train, y_train)

    print(f"Найкращі параметри kNN: {grid_knn.best_params_}, CV F1 = {grid_knn.best_score_:.4f}")

    plt.figure(figsize=(8, 4.5))
    cv_res_knn = pd.DataFrame(grid_knn.cv_results_)
    for p_val, p_lbl in [(1, 'L1 (Manhattan)'), (2, 'L2 (Euclidean)')]:
        for w_val, w_lbl in [('uniform', 'рівномірні'), ('distance', 'зважені')]:
            sub = cv_res_knn[(cv_res_knn['param_clf__p'] == p_val) & (cv_res_knn['param_clf__weights'] == w_val)]
            sub = sub.sort_values('param_clf__n_neighbors')
            plt.plot(sub['param_clf__n_neighbors'], sub['mean_test_score'], marker='o', label=f"{p_lbl}, {w_lbl}")

    plt.title('Залежність CV F1-score для kNN від k, метрики та ваг')
    plt.xlabel('Кількість сусідів k')
    plt.ylabel('CV F1-score')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('lr3_fig3_knn_k.png')
    plt.close()

    # ==========================================
    # 5. НАЛАШТУВАННЯ МАШИН ОПОРНИХ ВЕКТОРІВ (SVM)
    # ==========================================
    print("\n5. Налаштування Linear SVM та RBF SVM...")
    lin_pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', random_state=RANDOM_STATE))])
    grid_lin = GridSearchCV(lin_pipe, {'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]}, cv=cv, scoring='f1', n_jobs=1)
    grid_lin.fit(X_train, y_train)
    print(f"Найкращий Linear SVM: C = {grid_lin.best_params_['clf__C']}, CV F1 = {grid_lin.best_score_:.4f}")

    rbf_pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', random_state=RANDOM_STATE))])
    param_grid_rbf = {
        'clf__C': [0.1, 1.0, 10.0, 100.0],
        'clf__gamma': ['scale', 0.001, 0.01, 0.1, 1.0]
    }
    grid_rbf = GridSearchCV(rbf_pipe, param_grid_rbf, cv=cv, scoring='f1', n_jobs=1)
    grid_rbf.fit(X_train, y_train)
    print(f"Найкращий RBF SVM: {grid_rbf.best_params_}, CV F1 = {grid_rbf.best_score_:.4f}")

    plt.figure(figsize=(7, 5))
    res_rbf_df = pd.DataFrame(grid_rbf.cv_results_)
    res_rbf_df['param_clf__gamma'] = res_rbf_df['param_clf__gamma'].astype(str)
    heatmap_matrix = res_rbf_df.pivot(index='param_clf__gamma', columns='param_clf__C', values='mean_test_score')

    plt.imshow(heatmap_matrix, cmap='YlGnBu', aspect='auto', origin='lower')
    plt.colorbar(label='CV F1-score')
    plt.xticks(range(len(heatmap_matrix.columns)), heatmap_matrix.columns)
    plt.yticks(range(len(heatmap_matrix.index)), heatmap_matrix.index)
    plt.xlabel('Параметр C')
    plt.ylabel('Параметр gamma')
    plt.title('Спільний вплив C і gamma на CV F1 для RBF SVM')

    for i in range(len(heatmap_matrix.index)):
        for j in range(len(heatmap_matrix.columns)):
            val = heatmap_matrix.iloc[i, j]
            plt.text(j, i, f"{val:.3f}", ha='center', va='center', color='black' if val < 0.9 else 'white')

    plt.tight_layout()
    plt.savefig('lr3_fig4_svm_heatmap.png')
    plt.close()

    # ==========================================
    # 6. КОНТРОЛЬОВАНИЙ ЕКСПЕРИМЕНТ З РОЗМІРНІСТЮ (Шум)
    # ==========================================
    print("\n6. Дослідження прокляття розмірності для kNN...")
    noise_counts = [0, 10, 50, 100, 200, 500]
    d_ratios = []
    f1_noise_scores = []
    rng = np.random.default_rng(RANDOM_STATE)

    scaler_base = StandardScaler()
    X_tr_scaled = scaler_base.fit_transform(X_train)

    for nc in noise_counts:
        if nc == 0:
            X_cur = X_tr_scaled
        else:
            noise = rng.normal(0, 1, size=(X_tr_scaled.shape[0], nc))
            X_cur = np.hstack([X_tr_scaled, noise])
        
        dist_mat = pairwise_distances(X_cur[:150])
        np.fill_diagonal(dist_mat, np.inf)
        d_min = dist_mat.min(axis=1)
        np.fill_diagonal(dist_mat, -np.inf)
        d_max = dist_mat.max(axis=1)
        d_ratios.append(np.median(d_min / d_max))
        
        knn_raw = KNeighborsClassifier(**{k.replace('clf__', ''): v for k, v in grid_knn.best_params_.items()})
        cv_res_n = cross_validate(knn_raw, X_cur, y_train, cv=cv, scoring='f1')
        f1_noise_scores.append(cv_res_n['test_score'].mean())

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color = 'tab:blue'
    ax1.set_xlabel('Кількість доданих шумових ознак')
    ax1.set_ylabel('Медіана d_min / d_max', color=color)
    ax1.plot(noise_counts, d_ratios, color=color, marker='s', linewidth=2, label='d_min / d_max')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('kNN CV F1-score', color=color)
    ax2.plot(noise_counts, f1_noise_scores, color=color, marker='o', linewidth=2, linestyle='--', label='kNN CV F1')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Концентрація відстаней і деградація kNN через шум')
    plt.tight_layout()
    plt.savefig('lr3_fig5_curse_dim.png')
    plt.close()

    # ==========================================
    # 7. ЗВЕДЕНЕ ПОРІВНЯННЯ МОДЕЛЕЙ ТА ІНЖЕНЕРНИХ ХАРАКТЕРИСТИК
    # ==========================================
    print("\n7. Зведене порівняння моделей...")
    nearest_cent = Pipeline([('scaler', StandardScaler()), ('clf', NearestCentroid())])
    log_reg = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=RANDOM_STATE))])

    candidate_models = {
        'Nearest Centroid': nearest_cent,
        'kNN (оптимальний)': grid_knn.best_estimator_,
        'Linear SVM': grid_lin.best_estimator_,
        'RBF SVM (оптимальний)': grid_rbf.best_estimator_,
        'Logistic Regression': log_reg
    }

    summary_list = []
    for name, m in candidate_models.items():
        cv_ev = cross_validate(m, X_train, y_train, cv=cv, scoring=['f1', 'accuracy'])
        
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        fit_time = (time.perf_counter() - t0) * 1000  # мс
        
        t0 = time.perf_counter()
        m.predict(X_test)
        pred_time = (time.perf_counter() - t0) * 1000  # мс
        
        buf = io.BytesIO()
        pickle.dump(m, buf)
        size_kb = len(buf.getvalue()) / 1024.0
        
        clf_inner = m.named_steps['clf']
        if hasattr(clf_inner, 'support_vectors_'):
            n_stored = clf_inner.support_vectors_.shape[0]
        elif hasattr(clf_inner, '_fit_X'):
            n_stored = clf_inner._fit_X.shape[0]
        elif hasattr(clf_inner, 'centroids_'):
            n_stored = clf_inner.centroids_.shape[0]
        else:
            n_stored = clf_inner.coef_.shape[1]
            
        summary_list.append({
            'Модель': name,
            'CV F1': cv_ev['test_f1'].mean(),
            'CV SD': cv_ev['test_f1'].std(ddof=1),
            'CV Accuracy': cv_ev['test_accuracy'].mean(),
            'Час навчання, мс': fit_time,
            'Час інференсу, мс': pred_time,
            'Розмір, КБ': size_kb,
            'Збережені вектори/точки': n_stored
        })

    df_summary = pd.DataFrame(summary_list)
    print("\n--- ТАБЛИЦЯ 2: Інженерне порівняння кандидатних моделей ---")
    print(df_summary.to_string(index=False))

    plt.figure(figsize=(8, 4.5))
    for idx, r in df_summary.iterrows():
        plt.scatter(r['Час інференсу, мс'], r['CV F1'], s=r['Розмір, КБ']*10 + 50, alpha=0.7, edgecolors='black', label=r['Модель'])
        plt.text(r['Час інференсу, мс'] * 1.05, r['CV F1'], r['Модель'], fontsize=9)

    plt.xscale('log')
    plt.xlabel('Час інференсу на тест-вибірці, мс (лог-шкала)')
    plt.ylabel('CV F1-score')
    plt.title('Інженерний компроміс: якість, швидкість і розмір моделі')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr3_fig6_tradeoff.png')
    plt.close()

    # ==========================================
    # 8. OUT-OF-FOLD АНАЛІЗ РОЗБІЖНОСТЕЙ (kNN vs RBF SVM)
    # ==========================================
    print("\n8. Аналіз out-of-fold розбіжностей...")
    knn_oof = cross_val_predict(grid_knn.best_estimator_, X_train, y_train, cv=cv, n_jobs=1)
    svm_oof = cross_val_predict(grid_rbf.best_estimator_, X_train, y_train, cv=cv, n_jobs=1)

    disagreements = np.flatnonzero(knn_oof != svm_oof)
    print(f"Знайдено розбіжностей між kNN та RBF SVM: {len(disagreements)}")

    oof_records = []
    for idx in disagreements:
        true_cls = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        k_pred = knn_oof[idx]
        s_pred = svm_oof[idx]
        oof_records.append({
            'index': idx,
            'Справжній клас': true_cls,
            'kNN прогноз': k_pred,
            'SVM прогноз': s_pred,
            'Хто правий': 'kNN' if k_pred == true_cls else ('SVM' if s_pred == true_cls else 'Ніхто')
        })

    df_dis = pd.DataFrame(oof_records)
    print("\n--- ТАБЛИЦЯ 3: Фрагмент розбіжностей out-of-fold ---")
    print(df_dis.head(10).to_string(index=False))

    # ==========================================
    # 9. ФІНАЛЬНЕ ОЦІНЮВАННЯ НА TEST SET
    # ==========================================
    print("\n9. Фінальне оцінювання обраної моделі на test set...")
    best_final_model = grid_rbf.best_estimator_
    best_final_model.fit(X_train, y_train)
    y_pred_test = best_final_model.predict(X_test)

    final_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test),
        'Recall': recall_score(y_test, y_pred_test),
        'F1-score': f1_score(y_test, y_pred_test)
    }

    print("\n--- ТАБЛИЦЯ 4: Фінальні результати обраної моделі (RBF SVM) на test set ---")
    for k, v in final_metrics.items():
        print(f"{k} = {v:.4f}")

if __name__ == '__main__':
    main()