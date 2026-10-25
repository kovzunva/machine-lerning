# Інженерний аудит вибору моделі: batch_screening

- **Основна модель:** LogisticRegression (бал: 0.7883)
- **Резервна модель:** GradientBoosting
- **Кількість допустимих:** 3 / 4

## Рейтинг допустимих

| rank | model_name | score | norm_cv_roc_auc_mean | norm_cv_recall_mean | norm_cv_f1_mean | norm_cv_f1_std | norm_fit_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LogisticRegression | 0.7883 | 0.9677 | 0.5568 | 0.8 | 0.4829 | 1.0 |
| 2 | GradientBoosting | 0.7631 | 0.9202 | 0.6189 | 0.805 | 0.6486 | 0.3179 |
| 3 | RandomForest | 0.6499 | 0.8929 | 0.4337 | 0.6163 | 0.1114 | 0.6763 |

## Відхилені кандидати

| model_name | violations |
| --- | --- |
| DecisionTree | cv_recall_mean: 0.8706 < min 0.9; cv_f1_mean: 0.8963 < min 0.9 |
