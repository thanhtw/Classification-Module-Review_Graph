"""
Comprehensive analysis script: Combine per-label metrics, error analysis, 
and reproducibility tracking into detailed reports.

Usage:
    python function/comprehensive_analysis.py --results_dir results/modular_multimodel
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_training.per_label_metrics import (
    compute_per_label_metrics,
    generate_confusion_matrices,
    print_per_label_metrics,
)
from modular_training.error_analysis import (
    generate_error_summary,
    print_error_analysis,
)
from modular_training.reproducibility import (
    create_reproducibility_manifest,
    save_reproducibility_manifest,
    print_reproducibility_info,
)
from modular_training.config import LABEL_COLUMNS


def generate_comprehensive_report(results_dir: str, output_dir: str = None):
    """Generate comprehensive analysis report for all models and folds."""
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, "comprehensive_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    # Create summary dataframe
    summary_data = []
    
    # Load training process data
    jsonl_path = os.path.join(results_dir, "training_process.jsonl")
    
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: {jsonl_path} not found")
        print("Make sure you've run: python run_modular_multimodel_train.py")
        return
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    print(f"Found {len(records)} training records\n")
    
    # Process each record
    analysis_by_model = {}
    
    for record in records:
        model_name = record["model"]
        fold = record["fold"]
        metrics = record["metrics"]
        
        # Per-label metrics summary
        per_label_summary = {
            model_name: {
                "fold": fold,
                "f1_macro": metrics.get("f1_macro", 0),
                "subset_accuracy": metrics.get("subset_accuracy", 0),
            }
        }
        
        summary_data.append({
            "Model": model_name,
            "Fold": fold,
            "f1_macro": round(metrics.get("f1_macro", 0), 4),
            "subset_accuracy": round(metrics.get("subset_accuracy", 0), 4),
            "hamming_loss": round(metrics.get("hamming_loss", 0), 4),
            "train_time": round(record.get("train_time_sec", 0), 2),
            "infer_time": round(record.get("infer_time_sec", 0), 2),
        })
        
        if model_name not in analysis_by_model:
            analysis_by_model[model_name] = []
        
        analysis_by_model[model_name].append({
            "fold": fold,
            "metrics": metrics,
            "train_time": record.get("train_time_sec", 0),
            "infer_time": record.get("infer_time_sec", 0),
        })
    
    # Generate summary CSV
    df_summary = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, "all_folds_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"✓ Saved: {summary_csv}\n")
    
    # Generate per-model analysis
    model_analysis = {}
    
    for model_name, fold_data in analysis_by_model.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Aggregate metrics across folds
        all_metrics = [fd["metrics"] for fd in fold_data]
        
        # Calculate mean and std
        metric_keys = all_metrics[0].keys() if all_metrics else []
        fold_stats = {
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
        }
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                fold_stats["mean"][key] = round(np.mean(values), 4)
                fold_stats["std"][key] = round(np.std(values), 4)
                fold_stats["min"][key] = round(np.min(values), 4)
                fold_stats["max"][key] = round(np.max(values), 4)
        
        model_analysis[model_name] = {
            "n_folds": len(fold_data),
            "fold_statistics": fold_stats,
            "fold_details": all_metrics,
        }
        
        # Save per-model JSON
        model_json = os.path.join(model_dir, "model_analysis.json")
        with open(model_json, "w", encoding="utf-8") as f:
            json.dump(model_analysis[model_name], f, indent=2, ensure_ascii=False)
        
        print(f"Model: {model_name}")
        print(f"  Folds: {len(fold_data)}")
        print(f"  Mean F1-Macro: {fold_stats['mean'].get('f1_macro', 'N/A')}")
        print(f"  Mean Subset Accuracy: {fold_stats['mean'].get('subset_accuracy', 'N/A')}")
        print()
    
    # Generate best fold comparison
    best_folds_data = []
    for model_name in analysis_by_model.keys():
        folds = analysis_by_model[model_name]
        best_fold = max(folds, key=lambda x: x["metrics"].get("f1_macro", 0))
        
        best_folds_data.append({
            "Model": model_name,
            "Best_Fold": best_fold["fold"],
            "f1_macro": best_fold["metrics"].get("f1_macro", 0),
            "subset_accuracy": best_fold["metrics"].get("subset_accuracy", 0),
            "train_time": best_fold["train_time"],
            "infer_time": best_fold["infer_time"],
        })
    
    df_best = pd.DataFrame(best_folds_data).sort_values("f1_macro", ascending=False)
    best_csv = os.path.join(output_dir, "best_folds_comparison.csv")
    df_best.to_csv(best_csv, index=False)
    print(f"✓ Saved: {best_csv}\n")
    
    # Generate final summary report
    summary_txt = os.path.join(output_dir, "ANALYSIS_SUMMARY.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. PER-MODEL FOLD STATISTICS\n")
        f.write("-"*80 + "\n")
        for model_name, analysis in model_analysis.items():
            stats = analysis["fold_statistics"]["mean"]
            f.write(f"\n{model_name}:\n")
            f.write(f"  F1-Macro: {stats.get('f1_macro', 'N/A')} "
                   f"(±{analysis['fold_statistics']['std'].get('f1_macro', 'N/A')})\n")
            f.write(f"  Subset Accuracy: {stats.get('subset_accuracy', 'N/A')} "
                   f"(±{analysis['fold_statistics']['std'].get('subset_accuracy', 'N/A')})\n")
            f.write(f"  Hamming Loss: {stats.get('hamming_loss', 'N/A')} "
                   f"(±{analysis['fold_statistics']['std'].get('hamming_loss', 'N/A')})\n")
        
        f.write("\n\n2. BEST MODEL PER FOLD\n")
        f.write("-"*80 + "\n")
        for idx, row in df_best.iterrows():
            f.write(f"{idx+1}. {row['Model']} (Fold {int(row['Best_Fold'])}) "
                   f"- F1-Macro: {row['f1_macro']:.4f}\n")
        
        f.write("\n\n3. FILES GENERATED\n")
        f.write("-"*80 + "\n")
        f.write(f"Summary Table: {summary_csv}\n")
        f.write(f"Best Folds: {best_csv}\n")
        f.write(f"Per-model analysis: {output_dir}/<model_name>/model_analysis.json\n")
        
        f.write("\n\n4. NEXT STEPS\n")
        f.write("-"*80 + "\n")
        f.write("1. Check best_folds_comparison.csv for top-performing models\n")
        f.write("2. Review error_analysis_*.json files for failure patterns\n")
        f.write("3. Check per_label_report_*.json for label-specific performance\n")
        f.write("4. Review confusion matrices for false positive/negative patterns\n")
    
    print(f"✓ Saved: {summary_txt}\n")
    print(f"✓ Comprehensive analysis complete!")
    print(f"✓ Results saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive model analysis report"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/modular_multimodel",
        help="Path to results directory from training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for analysis (defaults to results_dir/comprehensive_analysis)"
    )
    
    args = parser.parse_args()
    
    generate_comprehensive_report(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
