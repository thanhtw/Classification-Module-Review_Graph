#!/usr/bin/env python
"""
Research Paper Comparison: LLM (Few-shot) vs Fine-tuned Transformers
Compare llama-3.1-8b-instant LLM (via Groq API) with BERT and RoBERTa

This module orchestrates comprehensive research paper generation by importing
specialized modules for different aspects of the comparison.
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from src.data.preprocessor import load_and_clean_data
from src.training.config import LABEL_COLUMNS

# Import organized research modules
from research_modules import (
    # Report builders
    generate_dataset_report,
    generate_model_configurations,
    generate_training_process_report,
    generate_research_paper_appendix,
    # Metrics analysis
    generate_per_label_metrics_report,
    generate_multilabel_metrics_report,
    # Visualizations
    generate_smote_visualization,
    generate_confusion_matrix_visualizations,
    generate_per_label_confusion_matrices,
    generate_training_curves,
    generate_comprehensive_heatmaps,
    generate_model_comparison_visualizations,
    # Table generators
    generate_comprehensive_metrics_report,
    generate_detailed_comparison_table,
)


def run_research_comparison(n_folds=10, seed=42):
    """Run comprehensive comparison: ML → DL → Transformers → LLM, then compare all"""
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON: All Models vs LLM (Zero-shot & Few-shot)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - LLM Model: llama-3.1-8b-instant (Groq API, 8B parameters)")
    print(f"  - LLM Approaches: Zero-shot + Few-shot (k=100)")
    print(f"  - ML Models: Linear SVM, Logistic Regression, Naive Bayes")
    print(f"  - DL Models: CNN, LSTM, BiLSTM")
    print(f"  - Transformers: BERT, RoBERTa")
    print(f"  - Folds: {n_folds}")
    print(f"  - Seed: {seed}")
    print(f"  - Task: Multilabel Classification (3 labels)")
    print(f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80 + "\n")
    
    # Models organized by category
    models_to_compare = [
        # Machine Learning Models
        ("Linear SVM", "linear_svm"),
        ("Logistic Regression", "logistic_regression"),
        ("Naive Bayes", "naive_bayes"),
        
        # Deep Learning Models
        ("CNN + Attention", "cnn_attention"),
        ("LSTM", "lstm"),
        ("BiLSTM", "bilstm"),
        
        # Transformer Models
        ("BERT", "bert"),
        ("RoBERTa", "roberta"),
        
        # LLM Models (Groq API)
        ("llama-3.1-8b-instant (LLM, Zero-shot)", "llm_zero_shot"),
        ("llama-3.1-8b-instant (LLM, Few-shot k=100)", "llm_few_shot"),
    ]
    
    comparison_results = []
    failed_models = []
    
    for idx, (model_display_name, model_key) in enumerate(models_to_compare, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(models_to_compare)}] Training: {model_display_name}")
        print(f"{'='*80}\n")
        
        try:
            # Run training via train.py script
            cmd = [
                sys.executable, 
                "scripts/train.py",
                f"--models={model_key}",
                f"--n_folds={n_folds}",
                f"--seed={seed}",
            ]
            
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                print(f"\n✗ Error training {model_display_name}:")
                print(result.stderr)
                failed_models.append(model_display_name)
                continue
            
            # Read results from the generated files
            results_base = project_root / "results" / "modular_multimodel"
            detailed_results = results_base / "model_results_detailed.csv"
            
            if detailed_results.exists():
                df = pd.read_csv(detailed_results)
                
                # Determine the correct column name for model
                model_col = None
                if 'Model' in df.columns:
                    model_col = 'Model'
                elif 'model' in df.columns:
                    model_col = 'model'
                elif 'model_key' in df.columns:
                    model_col = 'model_key'
                
                if model_col is None:
                    print(f"\n✗ Could not find model column in CSV for {model_display_name}")
                    failed_models.append(model_display_name)
                    continue
                
                # Filter for this model
                model_df = df[df[model_col] == model_key]
                
                if len(model_df) > 0:
                    # Calculate average metrics across folds
                    avg_metrics = {
                        'model': model_display_name,
                        'model_key': model_key,
                        'num_folds': len(model_df),
                    }
                    
                    # Extract numeric columns
                    numeric_cols = model_df.select_dtypes(include=['float64', 'int64']).columns
                    numeric_cols = [col for col in numeric_cols if col not in [model_col, 'fold', 'Fold']]
                    
                    for col in numeric_cols:
                        values = model_df[col].dropna().values
                        if len(values) > 0:
                            mean_val = float(values.mean())
                            std_val = float(values.std()) if len(values) > 1 else 0.0
                            avg_metrics[f'{col}_mean'] = mean_val
                            avg_metrics[f'{col}_std'] = std_val
                    
                    comparison_results.append(avg_metrics)
                    
                    f1_macro = avg_metrics.get('f1_macro_mean', avg_metrics.get('f1-macro_mean', 0))
                    f1_std = avg_metrics.get('f1_macro_std', avg_metrics.get('f1-macro_std', 0))
                    print(f"\n✓ {model_display_name} Complete")
                    print(f"  - Folds: {len(model_df)}")
                    print(f"  - F1-macro: {f1_macro:.4f} (±{f1_std:.4f})")
                else:
                    print(f"\n✗ No results found for {model_display_name} in CSV")
                    failed_models.append(model_display_name)
            else:
                print(f"\n✗ Results file not found: {detailed_results}")
                failed_models.append(model_display_name)
                
        except subprocess.TimeoutExpired:
            print(f"\n✗ Timeout training {model_display_name}")
            failed_models.append(model_display_name)
        except Exception as e:
            print(f"\n✗ Error training {model_display_name}: {e}")
            failed_models.append(model_display_name)
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive comparison report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80 + "\n")
    
    if comparison_results:
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        results_dir = Path("results/research_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = results_dir / "all_models_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to: {csv_path}\n")
        
        # Display comprehensive comparison table
        print("=" * 150)
        print(f"{'Rank':<5} | {'Model':<30} | {'F1-Macro':<15} | {'F1-Micro':<15} | {'Prec-Macro':<15} | {'Prec-Micro':<15} | {'Recall-Macro':<15} | {'Recall-Micro':<15}")
        print("=" * 150)
        
        # Sort by F1-macro descending
        sorted_results = sorted(comparison_results, key=lambda x: x.get('f1_macro_mean', 0), reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            model_name = result['model'][:28]
            f1_macro = f"{result.get('f1_macro_mean', 0):.4f}±{result.get('f1_macro_std', 0):.4f}"
            f1_micro = f"{result.get('f1_micro_mean', 0):.4f}±{result.get('f1_micro_std', 0):.4f}"
            prec_macro = f"{result.get('precision_macro_mean', 0):.4f}±{result.get('precision_macro_std', 0):.4f}"
            prec_micro = f"{result.get('precision_micro_mean', 0):.4f}±{result.get('precision_micro_std', 0):.4f}"
            rec_macro = f"{result.get('recall_macro_mean', 0):.4f}±{result.get('recall_macro_std', 0):.4f}"
            rec_micro = f"{result.get('recall_micro_mean', 0):.4f}±{result.get('recall_micro_std', 0):.4f}"
            print(f"{rank:<5} | {model_name:<30} | {f1_macro:<15} | {f1_micro:<15} | {prec_macro:<15} | {prec_micro:<15} | {rec_macro:<15} | {rec_micro:<15}")
        
        print("=" * 150)
        
        # Model category summary
        print("\n" + "=" * 80)
        print("SUMMARY BY CATEGORY")
        print("=" * 80 + "\n")
        
        categories = {
            'Machine Learning': ['Linear SVM', 'Logistic Regression', 'Naive Bayes'],
            'Deep Learning': ['CNN + Attention', 'LSTM', 'BiLSTM'],
            'Transformers': ['BERT', 'RoBERTa'],
            'LLM (Groq API)': ['llama-3.1-8b-instant (LLM, Zero-shot)', 'llama-3.1-8b-instant (LLM, Few-shot k=100)'],
        }
        
        for category, model_names in categories.items():
            category_results = [r for r in sorted_results if any(name in r['model'] for name in model_names)]
            if category_results:
                best = category_results[0]
                print(f"{category}:")
                print(f"  Best: {best['model']}")
                print(f"  F1-macro: {best.get('f1_macro_mean', 0):.4f} ± {best.get('f1_macro_std', 0):.4f}")
                print(f"  F1-micro: {best.get('f1_micro_mean', 0):.4f} ± {best.get('f1_micro_std', 0):.4f}")
                print(f"  Precision-macro: {best.get('precision_macro_mean', 0):.4f}")
                print(f"  Precision-micro: {best.get('precision_micro_mean', 0):.4f}")
                print(f"  Recall-macro: {best.get('recall_macro_mean', 0):.4f}")
                print(f"  Recall-micro: {best.get('recall_micro_mean', 0):.4f}")
                print()
        
        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS & REPORTS FOR ALL MODELS")
        print("=" * 80 + "\n")
        
        smote_vis = generate_smote_visualization(results_dir)
        generate_model_comparison_visualizations(comparison_results, results_dir)
        cm_vis = generate_confusion_matrix_visualizations(results_dir)
        per_label_cm_dir = generate_per_label_confusion_matrices(results_dir)
        curves_vis = generate_training_curves(results_dir)
        comp_csv = generate_comprehensive_metrics_report(comparison_results, results_dir)
        heatmap_vis = generate_comprehensive_heatmaps(comparison_results, results_dir)
        complete_csv, key_metrics_csv, latex_file = generate_detailed_comparison_table(comparison_results, results_dir)
        per_label_report = generate_per_label_metrics_report(comparison_results, results_dir)
        multilabel_report = generate_multilabel_metrics_report(comparison_results, results_dir)
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATION & REPORT FILES GENERATED")
        print("=" * 80)
        
        print("\n📊 COMPREHENSIVE VISUALIZATIONS (Publication-Ready 300 DPI):")
        if smote_vis:
            print(f"  ✓ {smote_vis}")
        print(f"  ✓ {results_dir}/model_f1_comparison.png")
        print(f"  ✓ {results_dir}/model_multilabel_metrics.png")
        if cm_vis:
            print(f"  ✓ {cm_vis}")
        if per_label_cm_dir:
            print(f"  ✓ Per-label confusion matrices: {per_label_cm_dir}/")
        if curves_vis:
            print(f"  ✓ {curves_vis}")
        if heatmap_vis:
            print(f"  ✓ {heatmap_vis}")
        
        print("\n📋 COMPREHENSIVE DATA TABLES & REPORTS:")
        print(f"  ✓ {comp_csv}")
        print(f"  ✓ {results_dir}/per_model_metrics/")
        print(f"  ✓ {complete_csv}")
        print(f"  ✓ {key_metrics_csv}")
        print(f"  ✓ {latex_file}")
        
        print("\n✨ Reports generated successfully!")
        print("=" * 80)
    else:
        print("\n✗ No results to compare!")
        if failed_models:
            print(f"\nFailed models: {', '.join(failed_models)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research paper comparison: LLM vs Fine-tuned Transformers")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Generate comprehensive research paper documentation
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE RESEARCH PAPER DOCUMENTATION")
    print("=" * 80)
    
    output_dir = "results/research_comparison"
    generate_research_paper_appendix(output_dir=output_dir)
    
    # Run model comparison
    run_research_comparison(n_folds=args.n_folds, seed=args.seed)
