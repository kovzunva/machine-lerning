import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =====================================================================
# 1. СТРУКТУРНІ ФУНКЦІЇ БАГАТОКРИТЕРІАЛЬНОГО ВИБОРУ
# =====================================================================

def check_constraints(row: Dict[str, Any], constraints: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
    """
    Перевіряє жорсткі інженерні обмеження.
    Значення на межі (дорівнює min або max) вважається повністю допустимим.
    """
    violations = []
    for metric, limits in constraints.items():
        if metric not in row or pd.isna(row[metric]):
            violations.append(f"{metric}: missing value")
            continue
        try:
            val = float(row[metric])
        except (ValueError, TypeError):
            violations.append(f"{metric}: non-numeric value")
            continue

        if "min" in limits and val < float(limits["min"]):
            violations.append(f"{metric}: {val:.4f} < min {limits['min']}")
        if "max" in limits and val > float(limits["max"]):
            violations.append(f"{metric}: {val:.4f} > max {limits['max']}")

    return (len(violations) == 0), violations


def normalize_value(value: float, spec: Dict[str, Any]) -> float:
    """
    Нормалізує значення до шкали [0.0, 1.0] на основі орієнтирів worst та best.
    Обрізає результат функцією clip. Заборонено вгадувати напрям за назвою.
    """
    direction = spec.get("direction")
    worst = float(spec["worst"])
    best = float(spec["best"])
    val = float(value)

    if worst == best:
        return 1.0 if val >= best else 0.0

    if direction == "maximize":
        normalized = (val - worst) / (best - worst)
    elif direction == "minimize":
        normalized = (worst - val) / (worst - best)
    else:
        raise ValueError(f"Unknown optimization direction: {direction}")

    return float(np.clip(normalized, 0.0, 1.0))


def validate_config(config: Dict[str, Any]) -> None:
    """Перевіряє коректність структури конфігурації та суму ваг."""
    if "criteria" not in config or not config["criteria"]:
        raise ValueError("Config must specify 'criteria'")
    total_weight = sum(float(c["weight"]) for c in config["criteria"].values())
    if not np.isclose(total_weight, 1.0, atol=1e-4):
        raise ValueError(f"Sum of criteria weights must be 1.0, got {total_weight:.4f}")


def rank_models(benchmark_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розподіляє моделі на допустимі та відхилені, обчислює нормалізовані
    корисності та зважений підсумковий бал з детермінованим розв'язанням ties.
    """
    validate_config(config)
    constraints = config.get("constraints", {})
    criteria = config["criteria"]

    eligible_rows = []
    rejected_rows = []

    for _, row in benchmark_df.iterrows():
        row_dict = row.to_dict()
        is_eligible, violations = check_constraints(row_dict, constraints)
        if is_eligible:
            norm_scores = {}
            weighted_sum = 0.0
            for crit_name, spec in criteria.items():
                if crit_name not in row_dict or pd.isna(row_dict[crit_name]):
                    raise ValueError(f"Missing required metric '{crit_name}' for model '{row_dict.get('model_name')}'")
                z = normalize_value(row_dict[crit_name], spec)
                norm_scores[f"norm_{crit_name}"] = round(z, 4)
                weighted_sum += z * float(spec["weight"])

            res_row = copy.deepcopy(row_dict)
            res_row.update(norm_scores)
            res_row["score"] = round(weighted_sum, 4)
            eligible_rows.append(res_row)
        else:
            rej_row = {
                "model_name": row_dict["model_name"],
                "violations": "; ".join(violations)
            }
            rejected_rows.append(rej_row)

    df_eligible = pd.DataFrame(eligible_rows)
    df_rejected = pd.DataFrame(rejected_rows)

    if not df_eligible.empty:
        # Детерміноване розв'язання нічиїх: спочатку за балом (спадання), потім за назвою (алфавіт)
        df_eligible.sort_values(by=["score", "model_name"], ascending=[False, True], inplace=True)
        df_eligible.reset_index(drop=True, inplace=True)
        df_eligible["rank"] = df_eligible.index + 1

    return df_eligible, df_rejected


# =====================================================================
# 2. ПРОФІЛЮВАННЯ ТА ЄДИНЕ ОЦІНЮВАННЯ
# =====================================================================

def profile_prediction(model: Any, batch: np.ndarray, warmups: int = 7, repeats: int = 45) -> np.ndarray:
    """Вимірює серію тривалостей прогнозування для фіксованого пакета."""
    for _ in range(warmups):
        if hasattr(model, "predict_proba"):
            model.predict_proba(batch)
        else:
            model.predict(batch)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        if hasattr(model, "predict_proba"):
            model.predict_proba(batch)
        else:
            model.predict(batch)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        samples.append(elapsed_ms)
    return np.asarray(samples)


def evaluate_and_profile_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidates: Dict[str, Pipeline],
    cv_splits: int = 5,
    batch_size: int = 64
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Pipeline]]:
    """Проводить чесне 5-fold CV та апаратне профілювання для всіх кандидатів."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    bench_records = []
    latency_distributions = {}
    trained_pipelines = {}

    eval_batch = X_train[:batch_size]

    for name, pipe in candidates.items():
        print(f"  -> Оцінювання кандидата: {name}...")
        fold_f1, fold_rec, fold_roc, fold_bal = [], [], [], []

        for train_idx, val_idx in skf.split(X_train, y_train):
            clone_pipe = copy.deepcopy(pipe)
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            clone_pipe.fit(X_tr, y_tr)
            y_pred = clone_pipe.predict(X_val)
            y_proba = clone_pipe.predict_proba(X_val)[:, 1] if hasattr(clone_pipe, "predict_proba") else y_pred

            fold_f1.append(f1_score(y_val, y_pred, zero_division=0))
            fold_rec.append(recall_score(y_val, y_pred, zero_division=0))
            fold_roc.append(roc_auc_score(y_val, y_proba))
            fold_bal.append(balanced_accuracy_score(y_val, y_pred))

        # Навчання на всій навчальній вибірці для серіалізації та профілювання
        final_pipe = copy.deepcopy(pipe)
        t_fit_start = time.perf_counter_ns()
        final_pipe.fit(X_train, y_train)
        fit_time_ms = (time.perf_counter_ns() - t_fit_start) / 1_000_000.0
        trained_pipelines[name] = final_pipe

        # Серіалізація у тимчасовий буфер для точного визначення розміру
        temp_file = f"_temp_{name}.joblib"
        joblib.dump(final_pipe, temp_file, compress=3)
        size_bytes = os.path.getsize(temp_file)
        model_size_mb = size_bytes / (1024.0 ** 2)
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Профілювання латентності
        lats = profile_prediction(final_pipe, eval_batch, warmups=7, repeats=45)
        latency_distributions[name] = lats

        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        throughput = float(batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0

        bench_records.append({
            "model_name": name,
            "cv_f1_mean": round(float(np.mean(fold_f1)), 4),
            "cv_f1_std": round(float(np.std(fold_f1, ddof=1)), 4),
            "cv_recall_mean": round(float(np.mean(fold_rec)), 4),
            "cv_roc_auc_mean": round(float(np.mean(fold_roc)), 4),
            "cv_balanced_acc_mean": round(float(np.mean(fold_bal)), 4),
            "fit_time_ms": round(fit_time_ms, 2),
            "latency_p50_ms": round(p50, 3),
            "latency_p95_ms": round(p95, 3),
            "throughput_samples_s": round(throughput, 1),
            "model_size_mb": round(model_size_mb, 5)
        })

    return pd.DataFrame(bench_records), latency_distributions, trained_pipelines


# =====================================================================
# 3. АНАЛІЗ ЧУТЛИВОСТІ ТА ПОБУДОВА ГРАФІКІВ
# =====================================================================

def run_sensitivity_analysis(benchmark_df: pd.DataFrame, base_config: Dict[str, Any], pivot_crit: str = "cv_roc_auc_mean") -> pd.DataFrame:
    """Проводить аналіз чутливості вибору шляхом варіювання ваги головного критерію."""
    weights_range = np.linspace(0.10, 0.70, 25)
    records = []

    for w in weights_range:
        cfg = copy.deepcopy(base_config)
        crit_keys = [k for k in cfg["criteria"].keys() if k != pivot_crit]
        cfg["criteria"][pivot_crit]["weight"] = float(w)

        remaining_weight = 1.0 - w
        orig_other_sum = sum(float(base_config["criteria"][k]["weight"]) for k in crit_keys)
        for k in crit_keys:
            if orig_other_sum > 0:
                cfg["criteria"][k]["weight"] = float(base_config["criteria"][k]["weight"]) * (remaining_weight / orig_other_sum)
            else:
                cfg["criteria"][k]["weight"] = remaining_weight / len(crit_keys)

        el_df, _ = rank_models(benchmark_df, cfg)
        if not el_df.empty:
            winner = el_df.iloc[0]["model_name"]
            row = {"pivot_weight": round(w, 3), "winner": winner}
            for _, r in el_df.iterrows():
                row[f"score_{r['model_name']}"] = r["score"]
            records.append(row)

    return pd.DataFrame(records)


def generate_all_plots(benchmark_df: pd.DataFrame, latencies: Dict[str, np.ndarray], sens_df: pd.DataFrame) -> None:
    """Генерує обов'язкові графіки у високій якості (DPI=300)."""
    # Рис. 1: Розподіл повторних вимірювань затримки
    plt.figure(figsize=(9, 4.8))
    data_to_plot = [latencies[m] for m in benchmark_df["model_name"]]
    plt.boxplot(data_to_plot, tick_labels=benchmark_df["model_name"], patch_artist=True,
                boxprops=dict(facecolor="#ccebc5", color="#2b5c8f"),
                medianprops=dict(color="#d95f02", linewidth=2))
    plt.ylabel("Час прогнозування пакета (batch=64), мс")
    plt.title("Розподіл повторних вимірювань затримки (45 запусків)")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("pr5_fig1_latency_distributions.png", dpi=300)
    plt.close()

    # Рис. 2: Компроміс якість проти затримки
    plt.figure(figsize=(8.5, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (_, r) in enumerate(benchmark_df.iterrows()):
        size_factor = max(50, r["model_size_mb"] * 1200)
        plt.scatter(r["latency_p95_ms"], r["cv_f1_mean"], s=size_factor,
                    color=colors[i % len(colors)], alpha=0.75, label=f"{r['model_name']} ({r['model_size_mb']:.3f} МБ)")
        plt.annotate(r["model_name"], (r["latency_p95_ms"], r["cv_f1_mean"]),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)
    plt.xlabel("Затримка p95 (мс, менше — краще)")
    plt.ylabel("Середнє CV F1-score (більше — краще)")
    plt.title("Інженерний компроміс: Якість проти Затримки (розмір точки ~ розмір моделі)")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("pr5_fig2_tradeoff_quality_latency.png", dpi=300)
    plt.close()

    # Рис. 3: Розмір моделей
    plt.figure(figsize=(8.5, 4.2))
    sorted_df = benchmark_df.sort_values(by="model_size_mb")
    bars = plt.barh(sorted_df["model_name"], sorted_df["model_size_mb"], color="#41b6c4", edgecolor="#225ea8")
    for b in bars:
        w = b.get_width()
        plt.text(w + 0.005, b.get_y() + b.get_height() / 2, f"{w:.4f} МБ", va="center", fontsize=9, fontweight="bold")
    plt.xlabel("Розмір серіалізованого pipeline у пам'яті (МБ)")
    plt.title("Порівняння обсягу серіалізованих pipeline")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("pr5_fig3_model_sizes.png", dpi=300)
    plt.close()

    # Рис. 4: Аналіз чутливості
    if not sens_df.empty:
        plt.figure(figsize=(9, 4.8))
        score_cols = [c for c in sens_df.columns if c.startswith("score_")]
        for col in score_cols:
            m_name = col.replace("score_", "")
            plt.plot(sens_df["pivot_weight"], sens_df[col], marker="o", markersize=4, label=m_name)
        plt.xlabel("Вага критерію CV ROC-AUC")
        plt.ylabel("Підсумковий зважений бал")
        plt.title("Чутливість рейтингу моделей до зміни пріоритету ROC-AUC")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pr5_fig4_sensitivity_curve.png", dpi=300)
        plt.close()

# 

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Немає даних_\n"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"
    ]
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(row[h]) for h in headers) + " |"
        lines.append(line)
    return "\n".join(lines)

# =====================================================================
# 4. ГОЛОВНИЙ ЦИКЛ ВИКОНАННЯ
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Автоматизований вибір ML-моделі за інженерними критеріями")
    parser.add_argument("--config", type=str, default=None, help="Шлях до JSON-файлу конфігурації")
    args = parser.parse_args()

    print("================================================================================")
    print("ПРАКТИЧНА РОБОТА №5: АВТОНОМНИЙ БАГАТОКРИТЕРІАЛЬНИЙ ВИБІР ML-МОДЕЛІ")
    print("================================================================================")

    # 1. Завантаження даних із приведенням міток (1=Malignant, 0=Benign)
    raw = load_breast_cancer()
    X = raw.data
    y = (raw.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Дані зафіксовано: Train = {X_train.shape[0]} об'єктів, Test = {X_test.shape[0]} об'єктів.")

    # 2. Визначення 4 фіксованих кандидатних конфігурацій (Pipeline)
    candidates = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, random_state=RANDOM_STATE))
        ]),
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2,
                                           max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1))
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                               min_samples_leaf=3, random_state=RANDOM_STATE))
        ])
    }

    # 3. CV-оцінювання та профілювання
    print("\n[КРОК 1] Проведення 5-fold CV та апаратного профілювання кандидатів...")
    benchmark_df, latencies, trained_models = evaluate_and_profile_candidates(
        X_train, y_train, candidates, cv_splits=5, batch_size=64
    )
    benchmark_df.to_csv("benchmark_results.csv", index=False)
    print("\nФАКТИЧНА МАТРИЦЯ РЕЗУЛЬТАТІВ (benchmark_results.csv):")
    print(benchmark_df.to_string(index=False))

    # 4. Обробка сценаріїв
    scenario_configs = []
    if args.config:
        scenario_configs.append(args.config)
    else:
        cfg_rt = os.path.join("configs", "realtime_service.json")
        cfg_bt = os.path.join("configs", "batch_screening.json")
        if os.path.exists(cfg_rt) and os.path.exists(cfg_bt):
            scenario_configs.extend([cfg_rt, cfg_bt])
        else:
            print("Попередження: папка configs не знайдена, використовуються вбудовані сценарії.")

    summary_outputs = {}

    for cfg_path in scenario_configs:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        sc_name = cfg["scenario"]
        print(f"\n================================================================================")
        print(f"[КРОК 2] РАНЖУВАННЯ ЗА СЦЕНАРІЄМ: {sc_name}")
        print(f"Опис: {cfg.get('description', '')}")
        print(f"================================================================================")

        df_eligible, df_rejected = rank_models(benchmark_df, cfg)

        df_eligible.to_csv(f"ranking_{sc_name}.csv", index=False)
        df_rejected.to_csv(f"rejected_{sc_name}.csv", index=False)

        print("\nВІДХИЛЕНІ МОДЕЛІ (порушення жорстких обмежень):")
        print(df_rejected.to_string(index=False) if not df_rejected.empty else "Жодна модель не відхилена.")

        print("\nРЕЙТИНГ ДОПУСТИМИХ МОДЕЛЕЙ:")
        cols_to_show = ["rank", "model_name", "score"] + [c for c in df_eligible.columns if c.startswith("norm_")]
        print(df_eligible[cols_to_show].to_string(index=False) if not df_eligible.empty else "Усі моделі відхилено.")

        if not df_eligible.empty:
            primary_name = df_eligible.iloc[0]["model_name"]
            backup_name = df_eligible.iloc[1]["model_name"] if len(df_eligible) > 1 else None

            # Збереження моделі
            joblib.dump(trained_models[primary_name], f"selected_{sc_name}.joblib", compress=3)

            sel_summary = {
                "scenario": sc_name,
                "version": "1.0.0",
                "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "primary_model": primary_name,
                "backup_model": backup_name,
                "primary_score": float(df_eligible.iloc[0]["score"]),
                "eligible_count": len(df_eligible),
                "rejected_count": len(df_rejected),
                "rejected_details": df_rejected.to_dict(orient="records"),
                "ranking": df_eligible[["rank", "model_name", "score"]].to_dict(orient="records")
            }
            with open(f"selection_{sc_name}.json", "w", encoding="utf-8") as f:
                json.dump(sel_summary, f, indent=4, ensure_ascii=False)

            # Markdown-звіт
            with open(f"MODEL_SELECTION_{sc_name}.md", "w", encoding="utf-8") as f:
                f.write(f"# Інженерний аудит вибору моделі: {sc_name}\n\n")
                f.write(f"- **Основна модель:** {primary_name} (бал: {df_eligible.iloc[0]['score']:.4f})\n")
                f.write(f"- **Резервна модель:** {backup_name}\n")
                f.write(f"- **Кількість допустимих:** {len(df_eligible)} / {len(benchmark_df)}\n\n")
                f.write("## Рейтинг допустимих\n\n" + dataframe_to_markdown(df_eligible[cols_to_show]) + "\n\n")
                f.write("## Відхилені кандидати\n\n" + dataframe_to_markdown(df_rejected) + "\n")

            summary_outputs[sc_name] = (primary_name, backup_name)

    # 5. Аналіз чутливості для пакетного сценарію
    print("\n[КРОК 3] Проведення аналізу чутливості до зміни пріоритетів...")
    batch_cfg_path = os.path.join("configs", "batch_screening.json")
    if os.path.exists(batch_cfg_path):
        with open(batch_cfg_path, "r", encoding="utf-8") as f:
            b_cfg = json.load(f)
        sens_df = run_sensitivity_analysis(benchmark_df, b_cfg, pivot_crit="cv_roc_auc_mean")
        sens_df.to_csv("sensitivity_analysis.csv", index=False)
        print("Аналіз чутливості зафіксовано в sensitivity_analysis.csv.")
    else:
        sens_df = pd.DataFrame()

    # 6. Побудова графіків
    print("\n[КРОК 4] Генерація аналітичних графіків...")
    generate_all_plots(benchmark_df, latencies, sens_df)
    print("Графіки збережено (pr5_fig1 - pr5_fig4).")

    # 7. Фінальне одноразове оцінювання на ізольованому test set
    print("\n================================================================================")
    print("[КРОК 5] ОДНОРАЗОВА ФІНАЛЬНА ПЕРЕВІРКА НА TEST SET (n = 114)")
    print("================================================================================")
    test_eval_records = []
    for sc_name, (prim, _) in summary_outputs.items():
        pipe = trained_models[prim]
        y_test_pred = pipe.predict(X_test)
        y_test_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_test_pred

        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_roc = roc_auc_score(y_test, y_test_proba)
        test_bal = balanced_accuracy_score(y_test, y_test_pred)

        test_eval_records.append({
            "scenario": sc_name,
            "selected_model": prim,
            "test_f1": round(test_f1, 4),
            "test_recall": round(test_rec, 4),
            "test_roc_auc": round(test_roc, 4),
            "test_balanced_acc": round(test_bal, 4)
        })

    df_test_eval = pd.DataFrame(test_eval_records)
    print(df_test_eval.to_string(index=False))
    df_test_eval.to_csv("test_set_evaluation.csv", index=False)
    print("\nУсі артефакти успішно згенеровано.")


if __name__ == "__main__":
    main()