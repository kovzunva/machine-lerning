import os
import json
from dataclasses import dataclass

class ConfigurationError(Exception):
    """Помилка завантаження або валідації конфігурації проєкту."""
    pass

@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    version: str
    model_path: str
    schema_path: str
    metadata_path: str
    train_ids_path: str
    test_ids_path: str
    threshold: float
    positive_class_label: str
    max_batch_size: int
    numeric_tolerance: float

def load_project_config(path: str, project_root: str = ".") -> ProjectConfig:
    """Завантажує та валідує конфігурацію, захищаючи від path traversal."""
    if not os.path.exists(path):
        raise ConfigurationError(f"Конфігураційний файл не знайдено: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Помилка парсингу JSON: {e}")
            
    required_keys = [
        "project_name", "version", "model_path", "schema_path", 
        "metadata_path", "train_ids_path", "test_ids_path",
        "threshold", "positive_class_label", "max_batch_size", "numeric_tolerance"
    ]
    for key in required_keys:
        if key not in data:
            raise ConfigurationError(f"Відсутній обов'язковий ключ у конфігурації: {key}")
            
    # Захист від path traversal: забороняємо вихід через '..'
    path_keys = ["model_path", "schema_path", "metadata_path", "train_ids_path", "test_ids_path"]
    for pk in path_keys:
        p_val = data[pk]
        normalized = os.path.normpath(os.path.join(project_root, p_val))
        root_abs = os.path.abspath(project_root)
        if not os.path.abspath(normalized).startswith(root_abs):
            raise ConfigurationError(f"Виявлено спробу Path Traversal у шляху {pk}: {p_val}")
            
    if not (0.0 < data["threshold"] < 1.0):
        raise ConfigurationError(f"Некоректний класифікаційний поріг: {data['threshold']}")
        
    if data["max_batch_size"] <= 0:
        raise ConfigurationError(f"max_batch_size повинен бути додатним: {data['max_batch_size']}")

    return ProjectConfig(**data)