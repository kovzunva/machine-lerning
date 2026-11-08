"""
Модуль чистих функцій для критичного аудиту ML-конвеєрів.
Відповідає за виявлення витоків, валідацію схем і розрахунок метрик.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score

def detect_split_overlap(train_ids, test_ids) -> set:
    """Перевіряє перетин між train та test вибірками."""
    return set(train_ids).intersection(set(test_ids))

def feature_target_correlations(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Обчислює модуль кореляції кожної ознаки з цільовою змінною."""
    records = []
    y_series = pd.Series(y, index=X.index)
    for col in X.columns:
        corr = float(X[col].corr(y_series))
        records.append({
            "feature": col,
            "correlation": corr,
            "abs_correlation": abs(corr)
        })
    df_corr = pd.DataFrame(records).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return df_corr

def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Незалежно обчислює базові та робастні метрики для бінарної класифікації."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        probs = (df - df.min()) / (df.max() - df.min() + 1e-12)
    else:
        probs = model.predict(X).astype(float)

    preds = (probs >= threshold).astype(int)
    
    ap = float(average_precision_score(y, probs))
    roc_auc = float(roc_auc_score(y, probs))
    bal_acc = float(balanced_accuracy_score(y, preds))
    brier = float(np.mean((probs - y) ** 2))
    
    return {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "brier_score": brier
    }

def validate_record(record: dict, schema: dict) -> tuple[bool, str, list[str]]:
    """
    Перевіряє вхідний запис на відповідність контракту даних.
    Повертає: (is_valid, status, warnings_or_errors)
    """
    warnings = []
    required_cols = list(schema.keys())
    
    # 1. Відсутні поля
    missing = set(required_cols) - set(record.keys())
    if missing:
        return False, "rejected", [f"Missing required feature: {col}" for col in missing]
        
    # 2. Зайві поля
    extra = set(record.keys()) - set(required_cols)
    if extra:
        return False, "rejected", [f"Unexpected feature: {col}" for col in extra]
        
    # 3. Типи та валідність чисел
    for col, spec in schema.items():
        val = record[col]
        if not isinstance(val, (int, float, np.number)):
            return False, "rejected", [f"Field {col} has non-numeric type: {type(val)}"]
        if np.isnan(val) or np.isinf(val):
            return False, "rejected", [f"Field {col} has NaN or Inf value"]
            
        # 4. Out-of-range checks
        if val < spec["min"] or val > spec["max"]:
            warnings.append(f"Value for {col} ({val}) is outside train bounds [{spec['min']:.2f}, {spec['max']:.2f}]")
            
    if warnings:
        return True, "accepted_with_warning", warnings
    return True, "accepted", []

def verify_claim(claim_id: str, claim_text: str, evidence: dict, threshold_fn) -> dict:
    """Верифікує твердження за наданими доказами."""
    verdict = threshold_fn(evidence)
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "evidence_summary": str(evidence),
        "verdict": verdict
    }

def build_finding(finding_id: str, problem: str, evidence: str, severity: str, recommendation: str, status: str) -> dict:
    """Формує структурований запис про дефект для реєстру знахідок."""
    return {
        "id": finding_id,
        "problem": problem,
        "evidence": evidence,
        "severity": severity,
        "recommendation": recommendation,
        "status": status
    }