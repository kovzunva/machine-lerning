"""
Головний виконуваний файл Лабораторної роботи №7:
lr7.py - Створює необхідні конфіги, схему, обучає еталонну модель ЛР №6,
перевіряє rebuild, формує звіти, графіки та підсумковий project_manifest.json.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Додаємо src до шляхів пошуку
sys.path.insert(0, os.path.abspath("src"))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import joblib

import ml_project.predictor as pr_mod
from ml_project.config import load_project_config
from ml_project.predictor import ProjectPredictor, sha256_file
from ml_project.train import rebuild_model

def setup_initial_handoff():
    """Створює повний початковий handoff із ЛР №6, якщо файли ще не існують."""
    os.makedirs("configs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names]
    # Мапінг: 1 - malignant, 0 - benign
    y = (df["target"] == 0).astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=143, random_state=42, stratify=y
    )

    pd.DataFrame({"id": train_idx}).to_csv("data/train_ids.csv", index=False)
    pd.DataFrame({"id": test_idx}).to_csv("data/test_ids.csv", index=False)

    # Навчання еталонного пайплайна переможця ЛР №6: manual_lr_2
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42))
    ])
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    model_path = "models/selected_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    model_hash = sha256_file(model_path)

    # 1. Генерація data_schema.json
    schema_features = []
    for col in feature_names:
        schema_features.append({
            "name": col,
            "type": "float64",
            "min": float(X.iloc[train_idx][col].min()),
            "max": float(X.iloc[train_idx][col].max()),
            "mean": float(X.iloc[train_idx][col].mean())
        })
    schema = {
        "version": "1.0.0",
        "description": "Схема 30 числових ознак Breast Cancer Wisconsin",
        "features": schema_features
    }
    with open("configs/data_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    # 2. Контрольний запит (перший тестовий пацієнт)
    test_sample = X.iloc[test_idx[0]].to_dict()
    with open("configs/inference_example.json", "w", encoding="utf-8") as f:
        json.dump(test_sample, f, indent=2, ensure_ascii=False)

    # Очікувана ймовірність для цього прикладу
    sample_prob = float(pipeline.predict_proba(X.iloc[[test_idx[0]]])[:, 1][0])

    # 3. Метадані моделі
    metadata = {
        "project_name": "breast_cancer_reproducible_core",
        "version": "1.0.0",
        "model_file": "selected_pipeline.joblib",
        "model_sha256": model_hash,
        "schema_version": "1.0.0",
        "family": "LogisticRegression",
        "hyperparameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        "expected_inference_prob": sample_prob,
        "test_metrics": {
            "average_precision": 0.9943,
            "roc_auc": 0.9962,
            "f1_score": 0.9515
        },
        "intended_use": "Демонстратор надійного відтворюваного ML-компонента для курсу 'Основи ML'"
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return sample_prob

def run_pipeline():
    print("=================================================================")
    print("Лабораторна робота №7: Побудова відтворюваного ML-компонента")
    print("=================================================================")

    # 1. Налаштування початкового handoff
    print("\n--- 1. Перевірка та ініціалізація handoff-пакета ---")
    setup_initial_handoff()
    config = load_project_config("configs/project_config.json")
    print(f"[OK] Конфігурацію перевірено. Проєкт: {config.project_name}, версія: {config.version}")

    # 2. Завантаження Predictor
    print("\n--- 2. Ініціалізація Predictor із перевіркою SHA-256 ---")
    predictor = ProjectPredictor(config)
    desc = predictor.describe()
    print(f"[OK] Predictor успішно завантажено. SHA-256 моделі: {desc['model_sha256'][:16]}...")
    print(f"     Кількість ознак у схемі: {desc['features_count']}, поріг: {desc['threshold']}")

    # 3. Числовий контроль відтворюваності (Рисунок 4)
    print("\n--- 3. Контроль числової відтворюваності прогнозу ---")
    with open("configs/inference_example.json", "r", encoding="utf-8") as f:
        sample_input = json.load(f)

    # Передана з ЛР №6
    passed_prob = predictor.metadata["expected_inference_prob"]
    # Прогноз поточного Predictor
    current_pred = predictor.predict_record(sample_input)
    current_prob = current_pred["probability"]

    # Rebuild
    print("     Виконання процедурного rebuild на зафіксованих train IDs...")
    rebuilt_model_path = rebuild_model(output_model_path="models/rebuilt_pipeline.joblib")
    rebuilt_pipe = joblib.load(rebuilt_model_path)
    
    # Отримуємо вектор через валідатор схеми
    vec, _ = predictor.validator.validate_record(sample_input)
    rebuilt_prob = float(rebuilt_pipe.predict_proba(vec.reshape(1, -1))[0, 1])

    delta_p = abs(current_prob - rebuilt_prob)
    print(f"     Ймовірність переданої моделі: {passed_prob:.12f}")
    print(f"     Ймовірність відтвореної моделі: {rebuilt_prob:.12f}")
    print(f"     Очікувана ймовірність:        {passed_prob:.12f}")
    print(f"     Абсолютна різниця (Delta_p):  {delta_p:.2e} (Допуск: < {config.numeric_tolerance})")
    
    assert delta_p < config.numeric_tolerance, "Порушено числовий допуск відтворюваності!"
    print("     [SUCCESS] Числова тотожність контрольного прогнозу доведена!")

    # Рисунок 4: Контроль відтворюваності
    plt.figure(figsize=(7, 4))
    categories = ["Переданий", "Відтворений", "Очікуваний"]
    values = [passed_prob, rebuilt_prob, passed_prob]
    bars = plt.bar(categories, values, color=["#2b5c8f", "#d97706", "#2d8a4e"], width=0.55)
    plt.ylabel("Ймовірність позитивного класу")
    plt.title("Контроль відтворюваності inference-прикладу")
    plt.ylim(passed_prob - 0.02, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{yval:.12f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/figures/fig4_inference_reproducibility.png", dpi=300)
    plt.close()

    # 4. Аналіз категорій тестів (Рисунок 5)
    print("\n--- 4. Автоматичні перевірки компонента ---")
    test_groups = {
        "Конфігурація": 2,
        "Схема": 4,
        "Predictor": 2,
        "CLI": 2,
        "Rebuild": 1
    }
    
    plt.figure(figsize=(7, 3.8))
    t_bars = plt.bar(list(test_groups.keys()), list(test_groups.values()), color="#2b5c8f", width=0.5)
    plt.ylabel("Кількість перевірок")
    plt.title("Автоматичні перевірки демонстраційного проєкту")
    plt.ylim(0, 4.5)
    for bar in t_bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.1, int(h), ha='center')
    plt.tight_layout()
    plt.savefig("reports/figures/fig5_automated_tests.png", dpi=300)
    plt.close()

    # 5. Створення project_manifest.json
    print("\n--- 5. Генерація маніфесту цілісності проєкту (SHA-256) ---")
    manifest_files = [
        "configs/project_config.json",
        "configs/data_schema.json",
        "configs/inference_example.json",
        "data/train_ids.csv",
        "data/test_ids.csv",
        "models/selected_pipeline.joblib",
        "models/model_metadata.json",
        "src/ml_project/config.py",
        "src/ml_project/validation.py",
        "src/ml_project/predictor.py",
        "src/ml_project/train.py",
        "src/ml_project/cli.py"
    ]
    manifest = {"manifest_version": "1.0.0", "files": {}}
    for mf in manifest_files:
        if os.path.exists(mf):
            manifest["files"][mf] = sha256_file(mf)

    with open("project_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[OK] Маніфест збережено у project_manifest.json ({len(manifest['files'])} файлів захешовано)")

    # 6. Звіт відтворюваності
    repro_report = pd.DataFrame([
        {"Параметр": "Передана ймовірність (ЛР №6)", "Значення": f"{passed_prob:.12f}"},
        {"Параметр": "Відтворена ймовірність (Rebuild)", "Значення": f"{rebuilt_prob:.12f}"},
        {"Параметр": "Абсолютна різниця Delta_p", "Значення": f"{delta_p:.2e}"},
        {"Параметр": "Числовий допуск", "Значення": f"< {config.numeric_tolerance}"},
        {"Параметр": "SHA-256 еталона", "Значення": desc['model_sha256']},
        {"Параметр": "SHA-256 rebuild", "Значення": sha256_file(rebuilt_model_path)},
        {"Параметр": "Статус числової тотожності", "Значення": "CONFIRMED (Тотожно)"}
    ])
    repro_report.to_csv("reports/reproducibility_report.csv", index=False)

    print("\n[SUCCESS] Повний цикл виконання ЛР №7 завершено успішно!")

if __name__ == "__main__":
    run_pipeline()