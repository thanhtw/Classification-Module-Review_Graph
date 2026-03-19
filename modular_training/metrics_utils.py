from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Jaccard similarity for multilabel classification."""
    vals = []
    for t, p in zip(y_true, y_pred):
        union = np.logical_or(t, p).sum()
        vals.append(1.0 if union == 0 else np.logical_and(t, p).sum() / union)
    return float(np.mean(vals))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute multilabel classification metrics.
    
    Metrics include:
    - accuracy_micro/macro: Per-label accuracy averaged over labels
    - precision_micro/macro: Per-label precision averaged over labels
    - recall_micro/macro: Per-label recall averaged over labels
    - f1_micro/macro: Per-label F1 averaged over labels
    - hamming_loss: Fraction of labels that are incorrectly predicted
    - subset_accuracy: Exact match ratio (all labels must match)
    - hamming_score: Jaccard similarity (intersection/union)
    """
    return {
        "accuracy_micro": float(accuracy_score(y_true, y_pred)),
        "accuracy_macro": float(accuracy_score(y_true, y_pred, normalize=True)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_score": float(hamming_score(y_true, y_pred)),
    }
