"""
Per-label metrics analysis and confusion matrix generation.
Handles class imbalance and provides detailed F1 breakdowns.
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-label metrics for multi-label classification.
    
    Args:
        y_true: True labels (n_samples, n_labels)
        y_pred: Predicted labels (n_samples, n_labels)
        label_names: Names of labels (e.g., ['relevance', 'constructiveness', 'concreteness'])
        
    Returns:
        Dict with per-label metrics (precision, recall, f1, support)
    """
    
    per_label_metrics = {}
    
    for label_idx, label_name in enumerate(label_names):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        
        # Calculate metrics
        precision = precision_score(y_true_label, y_pred_label, zero_division=0)
        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
        
        # Support and class distribution
        n_positive = int(y_true_label.sum())
        n_total = len(y_true_label)
        class_imbalance = (n_positive / n_total) * 100
        
        per_label_metrics[label_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(n_positive),
            "total_samples": int(n_total),
            "positive_rate": round(class_imbalance, 2),
        }
    
    return per_label_metrics


def generate_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_dir: str,
    model_name: str,
    fold: int,
) -> Dict[str, np.ndarray]:
    """
    Generate and save confusion matrices for each label.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of labels
        output_dir: Directory to save visualizations
        model_name: Name of model
        fold: Fold number
        
    Returns:
        Dict of confusion matrices
    """
    
    os.makedirs(output_dir, exist_ok=True)
    cf_matrices = {}
    
    fig, axes = plt.subplots(1, len(label_names), figsize=(15, 4))
    if len(label_names) == 1:
        axes = [axes]
    
    for label_idx, label_name in enumerate(label_names):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        
        cm = confusion_matrix(y_true_label, y_pred_label)
        cf_matrices[label_name] = cm
        
        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[label_idx],
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        axes[label_idx].set_title(f"{label_name.capitalize()}")
        axes[label_idx].set_ylabel("True")
        axes[label_idx].set_xlabel("Predicted")
    
    plt.suptitle(f"Confusion Matrices - {model_name} (Fold {fold})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f"confusion_matrices_{model_name}_fold{fold}.png")
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    return cf_matrices


def generate_per_label_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    model_name: str,
    fold: int,
    output_dir: str,
) -> Dict:
    """
    Generate comprehensive per-label report.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    per_label_metrics = compute_per_label_metrics(y_true, y_pred, label_names)
    cf_matrices = generate_confusion_matrices(
        y_true, y_pred, label_names, output_dir, model_name, fold
    )
    
    # Save report as JSON
    report = {
        "model": model_name,
        "fold": int(fold),
        "label_metrics": per_label_metrics,
    }
    
    report_path = os.path.join(output_dir, f"per_label_report_{model_name}_fold{fold}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def print_per_label_metrics(per_label_metrics: Dict[str, Dict], model_name: str):
    """Pretty print per-label metrics."""
    print(f"\n{'='*80}")
    print(f"PER-LABEL METRICS - {model_name}")
    print(f"{'='*80}")
    
    df_metrics = pd.DataFrame(per_label_metrics).T
    print(df_metrics.to_string())
    print()


if __name__ == "__main__":
    print("Per-label metrics module ready to use in analysis pipeline.")
