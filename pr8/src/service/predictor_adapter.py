import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd

def sha256_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

class ServicePredictor:
    def __init__(self, config_dict: dict, project_root: str = "."):
        self.config = config_dict
        self.project_root = project_root
        
        self.model_path = os.path.join(project_root, self.config["model_path"])
        self.metadata_path = os.path.join(project_root, self.config["metadata_path"])
        self.schema_path = os.path.join(project_root, self.config["schema_path"])
        
        self.is_ready = False
        self.model = None
        self.metadata = {}
        self.schema = {}
        self.expected_features = []
        self.bounds = {}
        
        self.load_artifacts()

    def load_artifacts(self):
        if not os.path.exists(self.model_path):
            return
        if not os.path.exists(self.metadata_path):
            return
        if not os.path.exists(self.schema_path):
            return

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        actual_hash = sha256_file(self.model_path)
        expected_hash = self.metadata.get("model_sha256")
        if actual_hash != expected_hash:
            return  # Блокуємо готовність через порушення SHA-256

        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
            
        self.expected_features = [feat["name"] for feat in self.schema["features"]]
        self.bounds = {feat["name"]: (feat["min"], feat["max"]) for feat in self.schema["features"]}

        self.model = joblib.load(self.model_path)
        self.is_ready = True

    def validate_features_dict(self, feats: dict) -> tuple[pd.DataFrame, list]:
        expected_set = set(self.expected_features)
        actual_set = set(feats.keys())

        missing = expected_set - actual_set
        if missing:
            raise ValueError(f"Відсутні обов'язкові ознаки: {sorted(list(missing))}")

        extra = actual_set - expected_set
        if extra:
            raise ValueError(f"Виявлено невідомі зайві ознаки: {sorted(list(extra))}")

        row = {}
        warnings = []
        for name in self.expected_features:
            val = feats[name]
            if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
                raise ValueError(f"Ознака '{name}' має нечисловий тип: {type(val).__name__}")
            f_val = float(val)
            if np.isnan(f_val) or np.isinf(f_val):
                raise ValueError(f"Ознака '{name}' містить недопустиме значення NaN або Inf")
                
            low, high = self.bounds[name]
            if f_val < low or f_val > high:
                warnings.append(f"Ознака '{name}' ({f_val:.4f}) виходить за межі навчального діапазону [{low:.4f}, {high:.4f}]")
            row[name] = [f_val]

        return pd.DataFrame(row), warnings

    def predict_one(self, feats: dict) -> tuple[float, int, str, list]:
        df_row, warnings = self.validate_features_dict(feats)
        prob = float(self.model.predict_proba(df_row)[0, 1])
        threshold = self.config["threshold"]
        pred_class = 1 if prob >= threshold else 0
        class_name = self.config["positive_class"] if pred_class == 1 else "benign"
        return prob, pred_class, class_name, warnings