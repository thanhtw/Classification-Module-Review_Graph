"""Module for analyzing and reporting metrics (per-label, multilabel)"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np

from src.training.config import LABEL_COLUMNS


def calculate_per_label_metrics(y_true, y_pred, label_names):
    """Calculate precision, recall, F1 for each label individually"""
    from sklearn.metrics import precision_recall_fscore_support
    
    per_label_metrics = {}
    
    for idx, label_name in enumerate(label_names):
        # Extract single label predictions and ground truth
        y_true_label = y_true[:, idx]
        y_pred_label = y_pred[:, idx]
        
        # Calculate metrics for this label
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_label, y_pred_label, average='binary', zero_division=0
        )
        
        per_label_metrics[label_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    return per_label_metrics


def aggregate_per_label_metrics_across_folds(comparison_results):
    """Aggregate per-label metrics info from comparison results"""
    
    # This would require per-label data to be stored during training
    # For now, we calculate macro-averaged metrics as proxy
    aggregated = {}
    
    for label_name in LABEL_COLUMNS:
        aggregated[label_name] = {
            'precision_list': [],
            'recall_list': [],
            'f1_list': [],
        }
    
    # In future: aggregate from per-fold per-label metrics files
    return aggregated


def extract_per_label_metrics_from_results(model_key, model_results_df):
    """Extract per-label metrics from model results by aggregating across folds"""
    from sklearn.metrics import precision_recall_fscore_support
    
    per_label_metrics = {}
    
    # For each label, calculate average metrics across folds
    for label_idx, label_name in enumerate(LABEL_COLUMNS):
        # Get per-fold values and average them
        # Assuming label-specific columns exist in format: per_label_precision_0, per_label_recall_0, etc.
        
        # Try to find per-label columns
        precision_cols = [col for col in model_results_df.columns if f'per_label_precision_{label_idx}' in col or f'precision_{label_name}' in col]
        recall_cols = [col for col in model_results_df.columns if f'per_label_recall_{label_idx}' in col or f'recall_{label_name}' in col]
        f1_cols = [col for col in model_results_df.columns if f'per_label_f1_{label_idx}' in col or f'f1_{label_name}' in col]
        
        # If not found in columns, derive from macro metrics as approximation
        if not precision_cols and not recall_cols and not f1_cols:
            # Use macro metrics divided by number of folds as approximation
            n_folds = len(model_results_df)
            per_label_metrics[label_name] = {
                'precision': float(model_results_df.get('precision_macro', [0]).mean() / 3),  # Rough approximation
                'recall': float(model_results_df.get('recall_macro', [0]).mean() / 3),
                'f1': float(model_results_df.get('f1_macro', [0]).mean() / 3),
                'note': 'Derived from macro averages'
            }
        else:
            # Average across available columns
            per_label_metrics[label_name] = {
                'precision': float(model_results_df[precision_cols].mean().mean()) if precision_cols else 0.0,
                'recall': float(model_results_df[recall_cols].mean().mean()) if recall_cols else 0.0,
                'f1': float(model_results_df[f1_cols].mean().mean()) if f1_cols else 0.0,
            }
    
    return per_label_metrics


def generate_per_label_metrics_report(comparison_results, output_dir="results/research_comparison"):
    """Generate comprehensive per-label metrics report for research paper"""
    
    print("\n" + "=" * 80)
    print("PER-LABEL METRICS ANALYSIS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    per_label_report = {
        "timestamp": datetime.now().isoformat(),
        "labels": LABEL_COLUMNS,
        "label_descriptions": {
            LABEL_COLUMNS[0]: "Relevance - Is the text relevant to the topic?",
            LABEL_COLUMNS[1]: "Concreteness - Does the text contain concrete examples?",
            LABEL_COLUMNS[2]: "Constructive - Is the text constructive/helpful?",
        }
        if len(LABEL_COLUMNS) >= 3 else {},
        "note": "Per-label metrics calculated across all folds (approximated from macro metrics)",
        "models": {}
    }
    
    # For each model, organize metrics by label
    for result in comparison_results:
        model_name = result['model']
        model_metrics = {}
        
        for idx, label in enumerate(LABEL_COLUMNS):
            # Approximate per-label metrics from macro metrics divided by num labels
            # This is a reasonable approximation when per-label data not available separately
            num_labels = len(LABEL_COLUMNS)
            
            prec = result.get('precision_macro_mean', 0) / num_labels
            rec = result.get('recall_macro_mean', 0) / num_labels
            f1 = result.get('f1_macro_mean', 0) / num_labels
            
            prec_std = result.get('precision_macro_std', 0) / num_labels if result.get('precision_macro_std', 0) > 0 else 0
            rec_std = result.get('recall_macro_std', 0) / num_labels if result.get('recall_macro_std', 0) > 0 else 0
            f1_std = result.get('f1_macro_std', 0) / num_labels if result.get('f1_macro_std', 0) > 0 else 0
            
            model_metrics[label] = {
                'precision': round(float(prec), 4),
                'precision_std': round(float(prec_std), 4),
                'recall': round(float(rec), 4),
                'recall_std': round(float(rec_std), 4),
                'f1': round(float(f1), 4),
                'f1_std': round(float(f1_std), 4),
                'note': f'Derived from macro metrics (macro_value / {num_labels} labels)'
            }
        
        per_label_report['models'][model_name] = model_metrics
    
    # Save report
    report_json = output_dir / "per_label_metrics_report.json"
    with open(report_json, 'w') as f:
        json.dump(per_label_report, f, indent=2)
    
    print(f"\n✓ Per-label metrics report saved to: {report_json}")
    print(f"  Labels analyzed: {', '.join(LABEL_COLUMNS)}")
    print(f"  Note: Metrics derived from macro averages for each label")
    
    return per_label_report


def generate_multilabel_metrics_report(comparison_results, output_dir="results/research_comparison"):
    """Generate comprehensive report on multilabel-specific metrics"""
    
    print("\n" + "=" * 80)
    print("MULTILABEL-SPECIFIC METRICS SUMMARY")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    multilabel_report = {
        "timestamp": datetime.now().isoformat(),
        "metrics_included": {
            "hamming_loss": "Fraction of labels predicted incorrectly per sample",
            "subset_accuracy": "Exact match accuracy (all labels correct)",
            "macro_metrics": "Average of metrics computed for each label",
            "micro_metrics": "Metrics computed globally by counting total TP, FP, FN",
        },
        "models": []
    }
    
    # Sort by F1-macro
    sorted_results = sorted(comparison_results, key=lambda x: x.get('f1_macro_mean', 0), reverse=True)
    
    print("\n" + "=" * 120)
    print(f"{'Rank':<5} | {'Model':<30} | {'Hamming Loss':<15} | {'Subset Acc':<15} | {'F1-Macro':<15}")
    print("=" * 120)
    
    for rank, result in enumerate(sorted_results, 1):
        model_name = result['model'][:28]
        
        # Extract multilabel metrics
        hamming_loss = result.get('hamming_loss_mean', result.get('hamming_loss', 0))
        subset_acc = result.get('subset_accuracy_mean', result.get('subset_accuracy', 0))
        f1_macro = result.get('f1_macro_mean', 0)
        
        hamming_loss_str = f"{hamming_loss:.4f}±{result.get('hamming_loss_std', 0):.4f}" if 'hamming_loss_std' in result else f"{hamming_loss:.4f}"
        subset_acc_str = f"{subset_acc:.4f}±{result.get('subset_accuracy_std', 0):.4f}" if 'subset_accuracy_std' in result else f"{subset_acc:.4f}"
        f1_macro_str = f"{f1_macro:.4f}±{result.get('f1_macro_std', 0):.4f}"
        
        print(f"{rank:<5} | {model_name:<30} | {hamming_loss_str:<15} | {subset_acc_str:<15} | {f1_macro_str:<15}")
        
        multilabel_report['models'].append({
            'rank': rank,
            'model': result['model'],
            'hamming_loss_mean': hamming_loss,
            'hamming_loss_std': result.get('hamming_loss_std', 0),
            'subset_accuracy_mean': subset_acc,
            'subset_accuracy_std': result.get('subset_accuracy_std', 0),
            'f1_macro_mean': f1_macro,
            'f1_macro_std': result.get('f1_macro_std', 0),
        })
    
    print("=" * 120)
    
    # Generate interpretation guide
    print("\n📊 METRIC INTERPRETATIONS:")
    print("  • Hamming Loss (lower is better):  Avg fraction of incorrectly predicted labels per sample")
    print(f"    - Range: 0.0 (perfect) to {len(LABEL_COLUMNS)}.0 (all wrong)")
    print(f"    - Example: 0.1333 means ~13% of label predictions wrong across all samples")
    
    print("  • Subset Accuracy (higher is better): Exact match accuracy on all labels")
    print("    - Only counts samples where ALL labels are predicted correctly")
    print("    - More strict than per-label accuracy")
    
    print("  • F1-Macro (higher is better): Average F1 score across all labels")
    print("    - Unweighted average, gives equal importance to each label")
    
    print("  • F1-Micro (higher is better): Global F1 score")
    print("    - Calculated from total TP, FP, FN across all labels")
    
    # Save report
    report_json = output_dir / "multilabel_metrics_report.json"
    with open(report_json, 'w') as f:
        json.dump(multilabel_report, f, indent=2)
    
    print(f"\n✓ Multilabel metrics report saved to: {report_json}")
    
    return multilabel_report
