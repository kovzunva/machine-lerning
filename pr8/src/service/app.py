import os
import time
import uuid
import json
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from service.schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ReadyResponse, ErrorResponse
)
from service.predictor_adapter import ServicePredictor

# Завантаження конфігурації
config_path = os.environ.get("OPERATIONAL_CONFIG_PATH", "configs/operational_config.json")
project_root = os.environ.get("PROJECT_ROOT", ".")

with open(config_path, "r", encoding="utf-8") as f:
    op_config = json.load(f)

app = FastAPI(
    title=op_config["service_name"],
    version=op_config["service_version"],
    description="REST API для інференсу моделі Breast Cancer (ПР №8)"
)

predictor = ServicePredictor(op_config, project_root=project_root)

# Лічильники метрик у пам'яті (In-memory Observability)
METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "errors_total": 0,
    "warnings_total": 0,
    "latencies_ms": []
}

@app.middleware("http")
async def add_context_and_metrics(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    
    response.headers["X-Request-ID"] = req_id
    
    if request.url.path in ["/predict", "/predict-batch"]:
        METRICS["requests_total"] += 1
        METRICS["latencies_ms"].append(duration_ms)
        if response.status_code == 200:
            METRICS["success_total"] += 1
        else:
            METRICS["errors_total"] += 1
            
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    METRICS["errors_total"] += 1
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            request_id=req_id,
            error_code="REQUEST_VALIDATION_ERROR",
            message="Помилка REST-контракту запиту або валідації Pydantic",
            details=exc.errors()
        ).model_dump()
    )

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service=op_config["service_name"],
        version=op_config["service_version"]
    )

@app.get("/ready", response_model=ReadyResponse)
def ready():
    if not predictor.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "message": "Модель не завантажена або пошкоджено SHA-256"}
        )
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        model_sha256=predictor.metadata.get("model_sha256", "unknown"),
        schema_version=predictor.schema.get("version", "1.0.0")
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    req_id = request.state.request_id
    if not predictor.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                request_id=req_id,
                error_code="SERVICE_NOT_READY",
                message="Сервіс не готовий до інференсу"
            ).model_dump()
        )

    try:
        prob, pred_cls, cls_name, warns = predictor.predict_one(payload.features)
        if warns:
            METRICS["warnings_total"] += len(warns)
        return PredictionResponse(
            request_id=req_id,
            probability=prob,
            predicted_class=pred_cls,
            predicted_class_name=cls_name,
            threshold=op_config["threshold"],
            positive_class=op_config["positive_class"],
            model_version=op_config["service_version"],
            schema_version=predictor.schema.get("version", "1.0.0"),
            warnings=warns
        )
    except ValueError as ve:
        METRICS["errors_total"] += 1
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                request_id=req_id,
                error_code="INPUT_SCHEMA_VIOLATION",
                message=str(ve)
            ).model_dump()
        )
    except Exception as exc:
        METRICS["errors_total"] += 1
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=req_id,
                error_code="INTERNAL_ERROR",
                message="Непередбачена внутрішня помилка сервера під час інференсу"
            ).model_dump()
        )

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request):
    req_id = request.state.request_id
    if not predictor.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(request_id=req_id, error_code="SERVICE_NOT_READY", message="Сервіс не готовий").model_dump()
        )

    max_batch = op_config["max_batch_size"]
    if len(payload.items) > max_batch:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                request_id=req_id,
                error_code="BATCH_SIZE_EXCEEDED",
                message=f"Розмір пакета ({len(payload.items)}) перевищує дозволений ліміт ({max_batch})"
            ).model_dump()
        )

    responses = []
    for item in payload.items:
        try:
            prob, pred_cls, cls_name, warns = predictor.predict_one(item.features)
            responses.append(PredictionResponse(
                request_id=req_id,
                probability=prob,
                predicted_class=pred_cls,
                predicted_class_name=cls_name,
                threshold=op_config["threshold"],
                positive_class=op_config["positive_class"],
                model_version=op_config["service_version"],
                schema_version=predictor.schema.get("version", "1.0.0"),
                warnings=warns
            ))
        except ValueError as ve:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(request_id=req_id, error_code="INPUT_SCHEMA_VIOLATION", message=str(ve)).model_dump()
            )

    return BatchPredictionResponse(
        request_id=req_id,
        total_samples=len(responses),
        model_version=op_config["service_version"],
        predictions=responses
    )

@app.get("/metrics")
def get_metrics():
    lats = METRICS["latencies_ms"]
    return {
        "requests_total": METRICS["requests_total"],
        "success_total": METRICS["success_total"],
        "errors_total": METRICS["errors_total"],
        "warnings_total": METRICS["warnings_total"],
        "latency_p50_ms": float(np.percentile(lats, 50)) if lats else 0.0,
        "latency_p95_ms": float(np.percentile(lats, 95)) if lats else 0.0,
        "latency_p99_ms": float(np.percentile(lats, 99)) if lats else 0.0
    }