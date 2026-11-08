import os
import hashlib
import json
import joblib
import numpy as np

from ml_project.config import ProjectConfig
from ml_project.validation import SchemaValidator, InputValidationError

def sha256_file(filepath: str) -> str:
    """Обчислює SHA-256 хеш файлу."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

class IntegrityError(Exception):
    """Помилка цілісності файлів артефактів."""
    pass

class ProjectPredictor:
    def __init__(self, config: ProjectConfig, project_root: str = "."):
        self.config = config
        self.project_root = project_root
        
        self.model_path = os.path.join(project_root, config.model_path)
        self.schema_path = os.path.join(project_root, config.schema_path)
        self.metadata_path = os.path.join(project_root, config.metadata_path)

        # 1. Завантаження метаданих та валідація цілісності артефакту
        if not os.path.exists(self.metadata_path):
            raise IntegrityError(f"Файл метаданих не знайдено: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        actual_model_hash = sha256_file(self.model_path)
        expected_model_hash = self.metadata.get("model_sha256")
        if actual_model_hash != expected_model_hash:
            raise IntegrityError(
                f"Критична помилка цілісності моделі! Фактичний SHA-256 ({actual_model_hash}) не збігається з очікуваним ({expected_model_hash})"
            )

        # 2. Завантаження валідатора схеми
        self.validator = SchemaValidator(self.schema_path)

        # 3. Десеріалізація перевіреної моделі
        self.pipeline = joblib.load(self.model_path)

    def describe(self) -> dict:
        """Повертає повний опис робочого компонента."""
        return {
            "project_name": self.config.project_name,
            "version": self.config.version,
            "model_family": self.metadata.get("family", "LogisticRegression"),
            "model_sha256": self.metadata.get("model_sha256"),
            "schema_version": self.validator.version,
            "features_count": len(self.validator.expected_feature_names),
            "threshold": self.config.threshold,
            "positive_class": self.config.positive_class_label,
            "max_batch_size": self.config.max_batch_size
        }

    def predict_record(self, record: dict) -> dict:
        """Формує прогноз для одного словника-запису."""
        features_df, warnings = self.validator.validate_record(record)
        
        prob = float(self.pipeline.predict_proba(features_df)[0, 1])
        pred_label = 1 if prob >= self.config.threshold else 0
        predicted_class_name = self.config.positive_class_label if pred_label == 1 else "benign"

        return {
            "probability": prob,
            "predicted_class": pred_label,
            "predicted_class_name": predicted_class_name,
            "threshold": self.config.threshold,
            "positive_class": self.config.positive_class_label,
            "model_version": self.config.version,
            "warnings": warnings
        }

    def predict_batch(self, records: list[dict]) -> dict:
        """Формує прогнози для пакета записів."""
        matrix, all_warnings = self.validator.validate_batch(records, self.config.max_batch_size)
        probs = self.pipeline.predict_proba(matrix)[:, 1]
        
        results = []
        for i in range(len(probs)):
            p = float(probs[i])
            pred_l = 1 if p >= self.config.threshold else 0
            results.append({
                "index": i,
                "probability": p,
                "predicted_class": pred_l,
                "predicted_class_name": self.config.positive_class_label if pred_l == 1 else "benign",
                "warnings": all_warnings[i]
            })

        return {
            "total_samples": len(results),
            "model_version": self.config.version,
            "threshold": self.config.threshold,
            "predictions": results
        }