"""
Extract and compare the best fold for each model from cross-validation results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def extract_best_folds(results_dir: str = "results/modular_multimodel") -> pd.DataFrame:
    """
    Extract the best fold for each model based on f1_macro metric.
    
    Args:
        results_dir: Path to results directory containing training_process.jsonl
        
    Returns:
        DataFrame with best fold results for each model
    """
    
    jsonl_path = os.path.join(results_dir, "training_process.jsonl")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")
    
    # Read all records
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        raise ValueError("No records found in training process file")
    
    # Group by model and find best fold
    model_best_folds = {}
    
    for record in records:
        model = record["model"]
        fold = record["fold"]
        metrics = record["metrics"]
        f1_macro = metrics.get("f1_macro", 0)
        
        if model not in model_best_folds:
            model_best_folds[model] = {
                "record": record,
                "f1_macro": f1_macro,
                "fold": fold,
            }
        else:
            # Keep record with higher f1_macro
            if f1_macro > model_best_folds[model]["f1_macro"]:
                model_best_folds[model] = {
                    "record": record,
                    "f1_macro": f1_macro,
                    "fold": fold,
                }
    
    # Convert to DataFrame for easy viewing
    comparison_data = []
    for model, data in model_best_folds.items():
        record = data["record"]
        metrics = record["metrics"]
        
        comparison_data.append({
            "Model": model,
            "Best_Fold": record["fold"],
            "f1_macro": round(metrics.get("f1_macro", 0), 4),
            "subset_accuracy": round(metrics.get("subset_accuracy", 0), 4),
            "hamming_loss": round(metrics.get("hamming_loss", 0), 4),
            "jaccard_score": round(metrics.get("jaccard_score", 0), 4),
            "micro_f1": round(metrics.get("micro_f1", 0), 4),
            "macro_f1": round(metrics.get("macro_f1", 0), 4),
            "train_time_sec": round(record.get("train_time_sec", 0), 2),
            "infer_time_sec": round(record.get("infer_time_sec", 0), 2),
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by f1_macro descending (best models first)
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    
    return df


def export_best_folds_comparison(results_dir: str = "results/modular_multimodel", 
                                  output_dir: str = None):
    """
    Extract best folds and export comparison to CSV and JSON.
    
    Args:
        results_dir: Path to results directory
        output_dir: Path to save comparison results (defaults to results_dir)
    """
    
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract best folds
    df_best = extract_best_folds(results_dir)
    
    print("\n" + "="*80)
    print("BEST FOLD COMPARISON - ALL MODELS")
    print("="*80)
    print(df_best.to_string(index=False))
    print("="*80 + "\n")
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "best_folds_comparison.csv")
    df_best.to_csv(csv_path, index=False)
    print(f"✓ Exported to CSV: {csv_path}")
    
    # Export to JSON
    json_path = os.path.join(output_dir, "best_folds_comparison.json")
    df_best.to_json(json_path, orient="records", indent=2, force_ascii=False)
    print(f"✓ Exported to JSON: {json_path}")
    
    # Export summary
    summary_path = os.path.join(output_dir, "best_model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("BEST FOLD COMPARISON - MODEL SELECTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Best overall model
        best_model = df_best.iloc[0]
        f.write("TOP MODEL (by f1_macro):\n")
        f.write(f"  Model: {best_model['Model']}\n")
        f.write(f"  Fold: {int(best_model['Best_Fold'])}\n")
        f.write(f"  F1-Macro: {best_model['f1_macro']:.4f}\n")
        f.write(f"  Subset Accuracy: {best_model['subset_accuracy']:.4f}\n")
        f.write(f"  Inference Time: {best_model['infer_time_sec']:.2f}s\n\n")
        
        # Top 3 models
        f.write("TOP 3 MODELS:\n")
        for idx, row in df_best.head(3).iterrows():
            f.write(f"  {idx+1}. {row['Model']} (Fold {int(row['Best_Fold'])}) - f1_macro: {row['f1_macro']:.4f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("FULL COMPARISON TABLE:\n")
        f.write("="*80 + "\n")
        f.write(df_best.to_string(index=False))
        f.write("\n")
    
    print(f"✓ Exported to TXT: {summary_path}")
    
    return df_best


if __name__ == "__main__":
    # Run comparison
    try:
        df = export_best_folds_comparison()
        print("\n✓ Comparison complete!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you've run the training script first:")
        print("  conda run -n ThomasAgent python run_modular_multimodel_train.py --models ...")
