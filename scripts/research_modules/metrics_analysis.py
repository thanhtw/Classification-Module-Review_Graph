"""Module for analyzing and reporting metrics (per-label, multilabel)"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


def generate_fold_level_per_label_report(
    artifact_root="results/modular_multimodel/model_artifacts",
    output_dir="results/research_comparison",
    model_keys=None,
    model_display_names=None,
    filename_prefix="llm_per_label",
):
    """Generate fold-level and aggregated per-label reports from saved artifacts."""
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd

    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_keys is None:
        model_keys = ["llm_zero_shot", "llm_few_shot"]

    if model_display_names is None:
        model_display_names = {
            "llm_zero_shot": "gpt-5.2-codex (LLM, Zero-shot)",
            "llm_few_shot": "gpt-5.2-codex (LLM, Few-shot k=10)",
        }

    fold_rows = []
    summary_rows = []

    for model_key in model_keys:
        model_dir = artifact_root / model_key
        if not model_dir.exists():
            continue

        for fold_dir in sorted(model_dir.glob("fold_*")):
            pred_path = fold_dir / "predictions.npy"
            label_path = fold_dir / "labels.npy"
            if not pred_path.exists() or not label_path.exists():
                continue

            y_pred = np.load(pred_path)
            y_true = np.load(label_path)
            if y_pred.ndim != 2 or y_true.ndim != 2 or y_pred.shape != y_true.shape:
                continue

            fold_num = int(fold_dir.name.split("_")[-1])
            for label_idx, label_name in enumerate(LABEL_COLUMNS):
                y_true_label = y_true[:, label_idx]
                y_pred_label = y_pred[:, label_idx]
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true_label, y_pred_label, average="binary", zero_division=0
                )
                fold_rows.append(
                    {
                        "model_key": model_key,
                        "model": model_display_names.get(model_key, model_key),
                        "fold": fold_num,
                        "label": label_name,
                        "support_positive": int(y_true_label.sum()),
                        "predicted_positive": int(y_pred_label.sum()),
                        "true_positive": int(np.logical_and(y_true_label == 1, y_pred_label == 1).sum()),
                        "precision": round(float(precision), 4),
                        "recall": round(float(recall), 4),
                        "f1": round(float(f1), 4),
                        "artifact_dir": str(fold_dir),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return {
            "fold_level_csv": "",
            "summary_csv": "",
            "rows": 0,
            "models": list(model_keys),
        }

    fold_df = fold_df.sort_values(["model_key", "fold", "label"]).reset_index(drop=True)

    grouped = (
        fold_df.groupby(["model_key", "model", "label"], as_index=False)
        .agg(
            folds=("fold", "count"),
            support_positive_total=("support_positive", "sum"),
            predicted_positive_total=("predicted_positive", "sum"),
            true_positive_total=("true_positive", "sum"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .fillna(0.0)
    )

    for _, row in grouped.iterrows():
        summary_rows.append(
            {
                "model_key": row["model_key"],
                "model": row["model"],
                "label": row["label"],
                "folds": int(row["folds"]),
                "support_positive_total": int(row["support_positive_total"]),
                "predicted_positive_total": int(row["predicted_positive_total"]),
                "true_positive_total": int(row["true_positive_total"]),
                "precision_mean": round(float(row["precision_mean"]), 4),
                "precision_std": round(float(row["precision_std"]), 4),
                "recall_mean": round(float(row["recall_mean"]), 4),
                "recall_std": round(float(row["recall_std"]), 4),
                "f1_mean": round(float(row["f1_mean"]), 4),
                "f1_std": round(float(row["f1_std"]), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["model_key", "label"]).reset_index(drop=True)

    per_label_report = {
        "timestamp": datetime.now().isoformat(),
        "labels": LABEL_COLUMNS,
        "label_descriptions": {
            LABEL_COLUMNS[0]: "Relevance - Is the text relevant to the topic?",
            LABEL_COLUMNS[1]: "Concreteness - Does the text contain concrete examples?",
            LABEL_COLUMNS[2]: "Constructive - Is the text constructive/helpful?",
        }
        if len(LABEL_COLUMNS) >= 3 else {},
        "selection": "Average across all available folds for selected models",
        "note": (
            "Per-label metrics are averaged across all available folds for the selected models. "
            f"See {filename_prefix}_all_folds.csv for fold-level details."
        ),
        "models": {},
    }

    for row in summary_rows:
        model_name = row["model"]
        model_metrics = per_label_report["models"].setdefault(
            model_name,
            {
                "model_key": row["model_key"],
                "folds": int(row["folds"]),
            },
        )
        model_metrics[row["label"]] = {
            "precision": row["precision_mean"],
            "precision_std": row["precision_std"],
            "recall": row["recall_mean"],
            "recall_std": row["recall_std"],
            "f1": row["f1_mean"],
            "f1_std": row["f1_std"],
            "support_positive_total": row["support_positive_total"],
            "predicted_positive_total": row["predicted_positive_total"],
            "true_positive_total": row["true_positive_total"],
            "note": "Averaged across all available folds",
        }

    fold_csv = output_dir / f"{filename_prefix}_all_folds.csv"
    summary_csv = output_dir / f"{filename_prefix}_summary.csv"
    report_json = output_dir / f"{filename_prefix}_report.json"
    report_txt = output_dir / f"{filename_prefix}_report.txt"
    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with open(report_json, "w") as f:
        json.dump(per_label_report, f, indent=2)

    lines = [
        "=" * 120,
        "PER-LABEL PERFORMANCE REPORT",
        "=" * 120,
        "",
        f"Generated: {per_label_report['timestamp']}",
        f"Labels: {', '.join(LABEL_COLUMNS)}",
        per_label_report["note"],
        "",
    ]

    for label in LABEL_COLUMNS:
        label_desc = per_label_report.get("label_descriptions", {}).get(label, label)
        lines.append(f"{label.upper()} ({label_desc})")
        lines.append("-" * 120)
        lines.append(
            f"{'Rank':<5} | {'Model':<50} | {'Folds':<6} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
        )
        lines.append("-" * 120)

        ranked_models = []
        for model_name, model_metrics in per_label_report["models"].items():
            metric_row = model_metrics.get(label, {})
            ranked_models.append(
                {
                    "model": model_name,
                    "folds": int(model_metrics.get("folds", 0)),
                    "precision": float(metric_row.get("precision", 0.0)),
                    "recall": float(metric_row.get("recall", 0.0)),
                    "f1": float(metric_row.get("f1", 0.0)),
                }
            )

        ranked_models.sort(key=lambda item: (item["f1"], item["recall"], item["precision"]), reverse=True)

        for rank, item in enumerate(ranked_models, 1):
            lines.append(
                f"{rank:<5} | {item['model'][:50]:<50} | {item['folds']:<6} | "
                f"{item['precision']:<10.4f} | {item['recall']:<10.4f} | {item['f1']:<10.4f}"
            )
        lines.append("")

    lines.append("MODEL SUMMARY (F1 BY LABEL)")
    lines.append("-" * 120)
    lines.append(
        f"{'Model':<50} | {'Folds':<6} | "
        + " | ".join(f"{label[:15]:<15}" for label in LABEL_COLUMNS)
    )
    lines.append("-" * 120)

    for model_name, model_metrics in per_label_report["models"].items():
        label_f1_values = " | ".join(
            f"{float(model_metrics.get(label, {}).get('f1', 0.0)):<15.4f}" for label in LABEL_COLUMNS
        )
        lines.append(
            f"{model_name[:50]:<50} | {int(model_metrics.get('folds', 0)):<6} | {label_f1_values}"
        )

    report_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "fold_level_csv": str(fold_csv),
        "summary_csv": str(summary_csv),
        "report_json": str(report_json),
        "report_txt": str(report_txt),
        "rows": len(fold_df),
        "models": sorted(fold_df["model_key"].unique().tolist()),
    }


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
    """Generate per-label metrics report using selected best fold per model."""
    
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
        "note": "Per-label metrics are computed from predictions/labels of selected best fold when artifacts exist.",
        "models": {}
    }
    
    # For each model, organize metrics by label
    for result in comparison_results:
        model_name = result['model']
        model_metrics = {}
        
        artifact_dir = result.get("artifact_dir", "")
        used_artifact = False
        if artifact_dir:
            pred_path = Path(str(artifact_dir)) / "predictions.npy"
            label_path = Path(str(artifact_dir)) / "labels.npy"
            if pred_path.exists() and label_path.exists():
                y_pred = np.load(pred_path)
                y_true = np.load(label_path)
                if y_pred.ndim == 2 and y_true.ndim == 2 and y_pred.shape == y_true.shape:
                    label_metrics = calculate_per_label_metrics(y_true, y_pred, LABEL_COLUMNS)
                    for label in LABEL_COLUMNS:
                        item = label_metrics.get(label, {})
                        model_metrics[label] = {
                            'precision': round(float(item.get('precision', 0.0)), 4),
                            'precision_std': 0.0,
                            'recall': round(float(item.get('recall', 0.0)), 4),
                            'recall_std': 0.0,
                            'f1': round(float(item.get('f1', 0.0)), 4),
                            'f1_std': 0.0,
                            'note': 'Computed from selected best-fold predictions/labels'
                        }
                    used_artifact = True

        if not used_artifact:
            for label in LABEL_COLUMNS:
                model_metrics[label] = {
                    'precision': round(float(result.get('precision_macro_mean', 0.0)), 4),
                    'precision_std': round(float(result.get('precision_macro_std', 0.0)), 4),
                    'recall': round(float(result.get('recall_macro_mean', 0.0)), 4),
                    'recall_std': round(float(result.get('recall_macro_std', 0.0)), 4),
                    'f1': round(float(result.get('f1_macro_mean', 0.0)), 4),
                    'f1_std': round(float(result.get('f1_macro_std', 0.0)), 4),
                    'note': 'Fallback from macro metrics (best-fold aggregate row)'
                }

        if 'selected_fold' in result:
            model_metrics['selected_fold'] = int(result.get('selected_fold', 0))
        model_metrics['artifact_dir'] = str(artifact_dir) if artifact_dir else ''
        
        per_label_report['models'][model_name] = model_metrics
    
    # Save report
    report_json = output_dir / "per_label_metrics_report.json"
    with open(report_json, 'w') as f:
        json.dump(per_label_report, f, indent=2)

    report_txt = output_dir / "per_label_metrics_report.txt"
    lines = [
        "=" * 120,
        "PER-LABEL PERFORMANCE REPORT",
        "=" * 120,
        "",
        f"Generated: {per_label_report['timestamp']}",
        f"Labels: {', '.join(LABEL_COLUMNS)}",
        per_label_report["note"],
        "",
    ]

    for label in LABEL_COLUMNS:
        label_desc = per_label_report.get("label_descriptions", {}).get(label, label)
        lines.append(f"{label.upper()} ({label_desc})")
        lines.append("-" * 120)
        lines.append(
            f"{'Rank':<5} | {'Model':<50} | {'Fold':<6} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
        )
        lines.append("-" * 120)

        ranked_models = []
        for model_name, model_metrics in per_label_report["models"].items():
            metric_row = model_metrics.get(label, {})
            ranked_models.append(
                {
                    "model": model_name,
                    "selected_fold": int(model_metrics.get("selected_fold", 0)),
                    "precision": float(metric_row.get("precision", 0.0)),
                    "recall": float(metric_row.get("recall", 0.0)),
                    "f1": float(metric_row.get("f1", 0.0)),
                }
            )

        ranked_models.sort(key=lambda item: (item["f1"], item["recall"], item["precision"]), reverse=True)

        for rank, item in enumerate(ranked_models, 1):
            lines.append(
                f"{rank:<5} | {item['model'][:50]:<50} | {item['selected_fold']:<6} | "
                f"{item['precision']:<10.4f} | {item['recall']:<10.4f} | {item['f1']:<10.4f}"
            )
        lines.append("")

    lines.append("MODEL SUMMARY (F1 BY LABEL)")
    lines.append("-" * 120)
    lines.append(
        f"{'Model':<50} | {'Fold':<6} | "
        + " | ".join(f"{label[:15]:<15}" for label in LABEL_COLUMNS)
    )
    lines.append("-" * 120)

    for model_name, model_metrics in per_label_report["models"].items():
        label_f1_values = " | ".join(
            f"{float(model_metrics.get(label, {}).get('f1', 0.0)):<15.4f}" for label in LABEL_COLUMNS
        )
        lines.append(
            f"{model_name[:50]:<50} | {int(model_metrics.get('selected_fold', 0)):<6} | {label_f1_values}"
        )

    report_txt.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"\n✓ Per-label metrics report saved to: {report_json}")
    print(f"✓ Per-label text report saved to: {report_txt}")
    print(f"  Labels analyzed: {', '.join(LABEL_COLUMNS)}")
    print("  Note: Best-fold artifact predictions are used when available")
    
    return per_label_report


def generate_multilabel_metrics_report(comparison_results, output_dir="results/research_comparison"):
    """Generate comprehensive report on multilabel-specific metrics (best fold per model)."""
    
    print("\n" + "=" * 80)
    print("MULTILABEL-SPECIFIC METRICS SUMMARY")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    multilabel_report = {
        "timestamp": datetime.now().isoformat(),
        "selection": "Best fold per model",
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
            'selected_fold': int(result.get('selected_fold', 0)),
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
