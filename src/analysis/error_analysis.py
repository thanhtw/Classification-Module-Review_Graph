"""
Error analysis: Identify and categorize misclassifications.
Provides qualitative insights into model weaknesses.
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict


def analyze_misclassifications(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    model_name: str,
    fold: int,
) -> Dict:
    """
    Analyze misclassified examples by type.
    
    Args:
        texts: List of text samples
        y_true: True labels (n_samples, n_labels)
        y_pred: Predicted labels (n_samples, n_labels)
        label_names: Names of labels
        model_name: Name of model
        fold: Fold number
        
    Returns:
        Analysis dict with error categories
    """
    
    analysis = {
        "model": model_name,
        "fold": int(fold),
        "errors_by_type": {},
        "samples": [],
    }
    
    # Find misclassifications
    misclassified_indices = []
    for i in range(len(texts)):
        if not np.array_equal(y_true[i], y_pred[i]):
            misclassified_indices.append(i)
    
    analysis["total_samples"] = len(texts)
    analysis["total_misclassified"] = len(misclassified_indices)
    analysis["error_rate"] = round(len(misclassified_indices) / len(texts), 4)
    
    # Categorize errors
    for idx in misclassified_indices[:100]:  # Limit to 100 examples
        text = texts[idx]
        true_labels = y_true[idx]
        pred_labels = y_pred[idx]
        
        # Determine error type
        error_types = []
        for label_idx, label_name in enumerate(label_names):
            if true_labels[label_idx] != pred_labels[label_idx]:
                if true_labels[label_idx] == 1 and pred_labels[label_idx] == 0:
                    error_types.append(f"False_Negative_{label_name}")
                else:
                    error_types.append(f"False_Positive_{label_name}")
        
        sample_record = {
            "text": text[:200],  # Truncate for brevity
            "true_labels": {label_names[i]: bool(true_labels[i]) for i in range(len(label_names))},
            "pred_labels": {label_names[i]: bool(pred_labels[i]) for i in range(len(label_names))},
            "error_types": error_types,
        }
        
        analysis["samples"].append(sample_record)
        
        # Track error type frequency
        for error_type in error_types:
            if error_type not in analysis["errors_by_type"]:
                analysis["errors_by_type"][error_type] = 0
            analysis["errors_by_type"][error_type] += 1
    
    # Sort error types by frequency
    analysis["errors_by_type"] = dict(
        sorted(analysis["errors_by_type"].items(), key=lambda x: x[1], reverse=True)
    )
    
    return analysis


def categorize_errors_by_label(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict[str, Dict]:
    """
    Categorize errors specifically by which label failed.
    Useful for understanding per-label model weaknesses.
    """
    
    error_categories = {label: {"false_positives": [], "false_negatives": []} for label in label_names}
    
    for i, text in enumerate(texts):
        for label_idx, label_name in enumerate(label_names):
            true_val = y_true[i, label_idx]
            pred_val = y_pred[i, label_idx]
            
            if true_val != pred_val:
                error_info = {
                    "text": text[:200],
                    "sample_idx": int(i),
                }
                
                if true_val == 1 and pred_val == 0:
                    error_categories[label_name]["false_negatives"].append(error_info)
                else:
                    error_categories[label_name]["false_positives"].append(error_info)
    
    return error_categories


def generate_error_summary(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    model_name: str,
    fold: int,
    output_dir: str,
) -> Dict:
    """
    Generate full error analysis and save reports.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Main error analysis
    error_analysis = analyze_misclassifications(
        texts, y_true, y_pred, label_names, model_name, fold
    )
    
    # Per-label categorization
    error_by_label = categorize_errors_by_label(texts, y_true, y_pred, label_names)
    
    # Add summary
    error_analysis["error_summary_by_label"] = {}
    for label_name in label_names:
        error_analysis["error_summary_by_label"][label_name] = {
            "false_positives": len(error_by_label[label_name]["false_positives"]),
            "false_negatives": len(error_by_label[label_name]["false_negatives"]),
        }
    
    # Save comprehensive report
    report_path = os.path.join(output_dir, f"error_analysis_{model_name}_fold{fold}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)
    
    # Save per-label errors
    per_label_path = os.path.join(output_dir, f"errors_by_label_{model_name}_fold{fold}.json")
    with open(per_label_path, "w", encoding="utf-8") as f:
        # Convert for JSON serialization
        error_by_label_serializable = {
            label: {
                "false_positives": fp[:5],  # Limit to 5 examples
                "false_negatives": fn[:5],
            }
            for label, data in error_by_label.items()
            for fp, fn in [(data["false_positives"], data["false_negatives"])]
        }
        json.dump(error_by_label_serializable, f, indent=2, ensure_ascii=False)
    
    return error_analysis


def print_error_analysis(error_analysis: Dict, model_name: str):
    """Pretty print error analysis."""
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS - {model_name}")
    print(f"{'='*80}")
    print(f"Total Samples: {error_analysis['total_samples']}")
    print(f"Misclassified: {error_analysis['total_misclassified']}")
    print(f"Error Rate: {error_analysis['error_rate']*100:.2f}%")
    print(f"\nTop Error Types:")
    for error_type, count in list(error_analysis["errors_by_type"].items())[:5]:
        print(f"  {error_type}: {count}")
    print()


if __name__ == "__main__":
    print("Error analysis module ready to use in analysis pipeline.")
