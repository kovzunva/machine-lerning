import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib

def rebuild_model(project_root: str = ".", output_model_path: str = None) -> str:
    """
    Процедурне відтворення моделі manual_lr_2 за зафіксованими train IDs
    без повторного пошуку гіперпараметрів чи вибору моделі.
    """
    train_ids_file = os.path.join(project_root, "data/train_ids.csv")
    if not os.path.exists(train_ids_file):
        raise FileNotFoundError(f"Файл зафіксованих train IDs не знайдено: {train_ids_file}")

    train_ids = pd.read_csv(train_ids_file)["id"].values

    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    features = [c for c in df.columns if c != "target"]
    X = df[features]
    # Мапінг: 1 - malignant, 0 - benign
    y = (df["target"] == 0).astype(int)

    X_train = X.iloc[train_ids]
    y_train = y.iloc[train_ids]

    # Специфікація переможця ЛР №6: manual_lr_2 (C=1.0, lbfgs, max_iter=1000)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    if output_model_path is None:
        output_model_path = os.path.join(project_root, "models/rebuilt_pipeline.joblib")

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(pipeline, output_model_path, compress=3)
    return output_model_path