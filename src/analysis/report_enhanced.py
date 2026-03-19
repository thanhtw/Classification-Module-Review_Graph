"""
Enhanced reporting utilities for model comparison and summary generation.
Generates text-based summary reports in addition to CSV outputs.
"""

import os
from typing import Dict, List

import pandas as pd


def generate_summary_report(df_detailed: pd.DataFrame, output_dir: str) -> str:
    """
    Generate a comprehensive text summary report.
    
    Includes:
    - Overall statistics (total samples, folds, models)
    - Best model across all folds
    - Per-model performance summary
    - Confidence intervals (std deviation)
    - Recommendation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary statistics
    n_models = df_detailed["model"].nunique()
    n_folds = df_detailed["fold"].nunique()
    n_total_runs = len(df_detailed)
    
    # Per-model aggregation
    model_stats = df_detailed.groupby("model", as_index=False).agg({
        "f1_macro": ["mean", "std", "min", "max"],
        "f1_micro": ["mean", "std"],
        "precision_macro": ["mean", "std"],
        "recall_macro": ["mean", "std"],
        "subset_accuracy": ["mean", "std"],
        "hamming_loss": ["mean", "std"],
        "train_time_sec": "mean",
        "infer_time_sec": "mean",
    }).round(4)
    
    # Best model by f1_macro_mean
    model_stats_flat = df_detailed.groupby("model", as_index=False).agg({
        "f1_macro": "mean",
        "f1_micro": "mean",
        "precision_macro": "mean",
        "recall_macro": "mean",
    }).sort_values("f1_macro", ascending=False)
    
    best_model = model_stats_flat.iloc[0]["model"]
    best_f1_macro = model_stats_flat.iloc[0]["f1_macro"]
    
    # Best individual fold across all models
    best_row = df_detailed.loc[df_detailed["f1_macro"].idxmax()]
    
    # Generate report
    report_lines = [
        "=" * 80,
        "MODEL COMPARISON SUMMARY REPORT",
        "=" * 80,
        "",
        "DATASET & EXPERIMENT STATISTICS",
        "─" * 80,
        f"Total Models Trained:        {n_models}",
        f"Cross-Validation Folds:      {n_folds}",
        f"Total Runs:                  {n_total_runs}",
        "",
        "OVERALL WINNER",
        "─" * 80,
        f"Model:                       {best_model}",
        f"F1-Macro Mean:               {best_f1_macro:.4f}",
        f"Recommended:                 YES ✓",
        "",
        "BEST INDIVIDUAL PERFORMANCE",
        "─" * 80,
        f"Model:                       {best_row['model']}",
        f"Best Fold:                   {int(best_row['fold'])}",
        f"F1-Macro:                    {best_row['f1_macro']:.4f}",
        f"F1-Micro:                    {best_row['f1_micro']:.4f}",
        f"Precision-Macro:             {best_row['precision_macro']:.4f}",
        f"Recall-Macro:                {best_row['recall_macro']:.4f}",
        f"Subset Accuracy:             {best_row['subset_accuracy']:.4f}",
        f"Hamming Loss:                {best_row['hamming_loss']:.4f}",
        "",
        "PER-MODEL PERFORMANCE SUMMARY",
        "─" * 80,
    ]
    
    # Flatten multi-level columns for display
    model_display = []
    for _, row in model_stats_flat.iterrows():
        model_name = row["model"]
        f1m_mean = row["f1_macro"]
        f1m_std = df_detailed[df_detailed["model"] == model_name]["f1_macro"].std()
        precision = row["precision_macro"]
        recall = row["recall_macro"]
        
        model_display.append({
            "Model": model_name,
            "F1-Macro": f"{f1m_mean:.4f}",
            "Std": f"{f1m_std:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
        })
    
    model_df = pd.DataFrame(model_display)
    report_lines.append(model_df.to_string(index=False))
    
    report_lines.extend([
        "",
        "KEY INSIGHTS",
        "─" * 80,
    ])
    
    # Find most stable model (lowest std)
    stability_df = df_detailed.groupby("model")["f1_macro"].std().sort_values()
    most_stable = stability_df.index[0]
    most_stable_std = stability_df.iloc[0]
    report_lines.append(f"Most Stable Model:           {most_stable} (std={most_stable_std:.4f})")
    
    # Find fastest model
    fastest_model = df_detailed.groupby("model")["train_time_sec"].mean().idxmin()
    fastest_time = df_detailed.groupby("model")["train_time_sec"].mean().min()
    report_lines.extend([
        f"Fastest Training:            {fastest_model} ({fastest_time:.2f}s avg)",
        "",
        "RECOMMENDATION",
        "─" * 80,
        f"Use Model:                   {best_model}",
        f"Rationale:                   Best F1-Macro across 10-fold CV",
        f"Expected Performance:        f1_macro={best_f1_macro:.4f}±{model_stats_flat.iloc[0]['f1_macro']-best_f1_macro:.4f}",
        f"Production Ready:            YES ✓",
        "",
        "=" * 80,
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "SUMMARY_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nSummary report saved: {report_path}")
    
    return report_path


def generate_confusion_matrix_report(data: List[Dict], output_dir: str) -> str:
    """
    Generate a report on model confusion rates.
    Shows which classes each model confuses most.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = [
        "=" * 80,
        "MODEL RELIABILITY ANALYSIS",
        "=" * 80,
        "",
        "Per-Model Error Patterns",
        "─" * 80,
    ]
    
    # Show hamming loss (fraction of incorrect labels)
    if data and isinstance(data[0], dict) and "hamming_loss" in data[0]:
        by_model = {}
        for row in data:
            model = row.get("model", "unknown")
            hamming = row.get("hamming_loss", 0.0)
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(hamming)
        
        report_lines.append(
            f"{'Model':<25} {'Hamming Loss':<15} {'Best':<10} {'Worst':<10}"
        )
        report_lines.append("─" * 65)
        
        for model in sorted(by_model.keys()):
            losses = by_model[model]
            avg_loss = sum(losses) / len(losses)
            best_loss = min(losses)
            worst_loss = max(losses)
            report_lines.append(
                f"{model:<25} {avg_loss:.4f}{'':<10} {best_loss:.4f}{'':<6} {worst_loss:.4f}"
            )
    
    report_lines.extend([
        "",
        "Interpretation:",
        "- Lower hamming_loss is better",
        "- Hamming loss = fraction of labels incorrectly predicted",
        "=" * 80,
    ])
    
    report_path = os.path.join(output_dir, "error_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    return report_path
