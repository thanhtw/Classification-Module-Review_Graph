"""
Backward compatibility module: imports from models_ml.py

New models with OneVsRestClassifier:
- run_linear_svm: LinearSVC for fast, efficient classification
- run_naive_bayes: MultinomialNB for probabilistic text classification
- run_logistic_regression: LogisticRegression with threshold tuning
"""

from src.models.models_ml import (
    run_linear_svm,
    run_naive_bayes,
    run_logistic_regression,
)

__all__ = [
    "run_linear_svm",
    "run_naive_bayes",
    "run_logistic_regression",
]


