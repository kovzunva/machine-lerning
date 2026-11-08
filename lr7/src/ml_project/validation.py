import json
import numpy as np
import pandas as pd

class InputValidationError(Exception):
    """Помилка валідації вхідних даних проти схеми контракту."""
    pass

class SchemaValidator:
    def __init__(self, schema_path: str):
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.version = self.schema.get("version", "1.0.0")
        self.features_spec = self.schema["features"]
        self.expected_feature_names = [f["name"] for f in self.features_spec]
        self.bounds = {f["name"]: (f["min"], f["max"]) for f in self.features_spec}

    def validate_record(self, record: dict) -> tuple[pd.DataFrame, list[str]]:
        expected_set = set(self.expected_feature_names)
        actual_set = set(record.keys())

        missing = expected_set - actual_set
        if missing:
            raise InputValidationError(f"Відсутні обов'язкові ознаки: {sorted(list(missing))}")

        extra = actual_set - expected_set
        if extra:
            raise InputValidationError(f"Виявлено невідомі зайві ознаки: {sorted(list(extra))}")

        ordered_values = {}
        warnings = []

        for name in self.expected_feature_names:
            val = record[name]
            if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
                raise InputValidationError(f"Ознака '{name}' має нечисловий тип: {type(val).__name__}")
            
            f_val = float(val)
            if np.isnan(f_val) or np.isinf(f_val):
                raise InputValidationError(f"Ознака '{name}' містить неприпустиме значення NaN або Inf")

            low, high = self.bounds[name]
            if f_val < low or f_val > high:
                warnings.append(f"Ознака '{name}' зі значенням {f_val:.4f} виходить за межі навчального діапазону [{low:.4f}, {high:.4f}]")

            ordered_values[name] = [f_val]

        df_record = pd.DataFrame(ordered_values)
        return df_record, warnings

    def validate_batch(self, records: list[dict], max_batch_size: int = 200) -> tuple[pd.DataFrame, list[list[str]]]:
        if not isinstance(records, list) or len(records) == 0:
            raise InputValidationError("Пакет даних повинен бути непорожнім списком словників")
            
        if len(records) > max_batch_size:
            raise InputValidationError(f"Розмір пакета ({len(records)}) перевищує дозволений ліміт ({max_batch_size})")

        frames = []
        all_warnings = []
        for i, rec in enumerate(records):
            try:
                df_rec, warns = self.validate_record(rec)
                frames.append(df_rec)
                all_warnings.append(warns)
            except InputValidationError as e:
                raise InputValidationError(f"Помилка у рядку {i}: {e}")

        return pd.concat(frames, ignore_index=True), all_warnings