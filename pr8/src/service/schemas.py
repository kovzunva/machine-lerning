from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Словник із 30 числових ознак біоптату")

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
                    "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776,
                    "mean concavity": 0.3001, "mean concave points": 0.1471, "mean symmetry": 0.2419,
                    "mean fractal dimension": 0.07871, "radius error": 1.095, "texture error": 0.9053,
                    "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006399,
                    "compactness error": 0.04904, "concavity error": 0.05373, "concave points error": 0.01587,
                    "symmetry error": 0.03003, "fractal dimension error": 0.006193, "worst radius": 25.38,
                    "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
                    "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119,
                    "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
                }
            }
        }
    }

class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest] = Field(..., description="Пакет спостережень")

class PredictionResponse(BaseModel):
    request_id: str
    probability: float
    predicted_class: int
    predicted_class_name: str
    threshold: float
    positive_class: str
    model_version: str
    schema_version: str
    warnings: List[str] = []

class BatchPredictionResponse(BaseModel):
    request_id: str
    total_samples: int
    model_version: str
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    model_sha256: str
    schema_version: str

class ErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    details: Optional[Any] = None