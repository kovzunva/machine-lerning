# Інженерний аудит вибору моделі: realtime_service

- **Основна модель:** LogisticRegression (бал: 0.8538)
- **Резервна модель:** GradientBoosting
- **Кількість допустимих:** 3 / 4

## Рейтинг допустимих

| rank | model_name | score | norm_cv_f1_mean | norm_cv_recall_mean | norm_latency_p95_ms | norm_model_size_mb | norm_cv_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LogisticRegression | 0.8538 | 0.8769 | 0.735 | 0.9216 | 0.9782 | 0.5978 |
| 2 | GradientBoosting | 0.7709 | 0.88 | 0.7771 | 0.9411 | 0.2733 | 0.7267 |
| 3 | DecisionTree | 0.5544 | 0.3562 | 0.1471 | 0.9882 | 0.9831 | 0.3244 |

## Відхилені кандидати

| model_name | violations |
| --- | --- |
| RandomForest | model_size_mb: 0.0556 > max 0.05 |
