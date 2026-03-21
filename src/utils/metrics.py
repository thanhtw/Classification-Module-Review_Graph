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


def tune_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Tune one threshold per label by maximizing validation F1."""
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true={y_true.shape}, y_prob={y_prob.shape}")
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")

    if threshold_grid is None:
        threshold_grid = np.arange(0.10, 0.91, 0.05)

    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=float)

    for i in range(n_labels):
        yi_true = y_true[:, i]
        yi_prob = y_prob[:, i]

        best_t = 0.5
        best_f1 = -1.0
        for t in threshold_grid:
            yi_pred = (yi_prob >= t).astype(int)
            f1 = f1_score(yi_true, yi_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        thresholds[i] = best_t

    return thresholds


def apply_per_label_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-label thresholds to probability matrix."""
    if y_prob.ndim != 2:
        raise ValueError(f"Expected 2D y_prob, got {y_prob.shape}")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1:
        raise ValueError(f"Expected 1D thresholds, got {thresholds.shape}")
    if y_prob.shape[1] != thresholds.shape[0]:
        raise ValueError(
            f"Threshold count ({thresholds.shape[0]}) != label count ({y_prob.shape[1]})"
        )
    return (y_prob >= thresholds.reshape(1, -1)).astype(int)


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
    # Micro accuracy for multilabel: accuracy over all label decisions.
    accuracy_micro = float(accuracy_score(y_true.ravel(), y_pred.ravel()))

    # Macro accuracy for multilabel: mean of per-label accuracies.
    per_label_acc = (y_true == y_pred).mean(axis=0)
    accuracy_macro = float(np.mean(per_label_acc))

    return {
        "accuracy_micro": accuracy_micro,
        "accuracy_macro": accuracy_macro,
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
