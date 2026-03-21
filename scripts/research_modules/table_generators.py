"""Module for generating comprehensive data tables and reports"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def generate_comprehensive_metrics_report(comparison_results, output_dir="results/research_comparison"):
    """Generate comprehensive metrics report for ALL models including per-label metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating comprehensive metrics report for all models...")
    
    # Find all available metric columns
    all_metrics = set()
    for result in comparison_results:
        all_metrics.update(result.keys())
    
    # Remove non-metric columns
    non_metrics = {'model', 'model_key', 'num_folds', 'model_name'}
    all_metrics = sorted([m for m in all_metrics if m not in non_metrics])
    
    print(f"\n  ✓ Found {len(all_metrics)} unique metrics across all models")
    print(f"  ✓ Processing {len(comparison_results)} models")
    
    # Create comprehensive CSV with ALL metrics
    comprehensive_df = pd.DataFrame(comparison_results)
    
    # Organize columns: model name first, then all metrics
    model_cols = ['model', 'model_key', 'num_folds']
    metric_cols = [col for col in comprehensive_df.columns if col not in model_cols]
    
    comprehensive_df = comprehensive_df[model_cols + metric_cols]
    
    # Save comprehensive results
    comp_csv = output_dir / "all_models_all_metrics.csv"
    comprehensive_df.to_csv(comp_csv, index=False)
    print(f"  ✓ Comprehensive metrics saved to: {comp_csv}")
    
    # Generate per-model detailed reports
    per_model_dir = output_dir / "per_model_metrics"
    per_model_dir.mkdir(exist_ok=True)
    
    for idx, result in enumerate(comparison_results, 1):
        model_name = result.get('model', f'model_{idx}').replace('/', '_').replace(' ', '_')
        model_report_path = per_model_dir / f"{model_name}_metrics.txt"
        
        with open(model_report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"MODEL #{idx}: {result.get('model', 'Unknown')}\n")
            f.write("=" * 100 + "\n\n")
            
            # Extract all metrics
            f.write("METRICS SUMMARY:\n")
            f.write("-" * 100 + "\n")
            
            metric_dict = {k: v for k, v in result.items() 
                          if k not in non_metrics and isinstance(v, (int, float))}
            
            # Group metrics by type
            metric_groups = {
                'Accuracy Metrics': [k for k in metric_dict if 'accuracy' in k.lower()],
                'F1-Score Metrics': [k for k in metric_dict if 'f1' in k.lower()],
                'Precision Metrics': [k for k in metric_dict if 'precision' in k.lower()],
                'Recall Metrics': [k for k in metric_dict if 'recall' in k.lower()],
                'Hamming Loss': [k for k in metric_dict if 'hamming' in k.lower()],
                'Subset Accuracy': [k for k in metric_dict if 'subset' in k.lower()],
                'Other Metrics': [k for k in metric_dict if k not in 
                                 [x for group in [
                                     [k for k in metric_dict if 'accuracy' in k.lower()],
                                     [k for k in metric_dict if 'f1' in k.lower()],
                                     [k for k in metric_dict if 'precision' in k.lower()],
                                     [k for k in metric_dict if 'recall' in k.lower()],
                                     [k for k in metric_dict if 'hamming' in k.lower()],
                                     [k for k in metric_dict if 'subset' in k.lower()],
                                 ] for x in group]]
            }
            
            for group_name, metrics in metric_groups.items():
                if metrics:
                    f.write(f"\n{group_name}:\n")
                    for metric in sorted(metrics):
                        value = metric_dict[metric]
                        f.write(f"  • {metric:<40} = {value:>10.6f}\n")
            
            f.write("\n" + "-" * 100 + "\n")
            f.write(f"Total Folds: {result.get('num_folds', 'N/A')}\n")
            f.write(f"Model Key: {result.get('model_key', 'N/A')}\n")
    
    print(f"  ✓ Per-model reports generated in: {per_model_dir}")
    
    # Create summary statistics file
    summary_file = output_dir / "metrics_summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("COMPREHENSIVE METRICS SUMMARY - ALL MODELS\n")
        f.write("=" * 120 + "\n\n")
        
        # Statistics for each metric
        f.write("METRIC STATISTICS (Mean ± Std across all models):\n")
        f.write("-" * 120 + "\n")
        
        for col in metric_cols:
            if comprehensive_df[col].dtype in ['float64', 'int64']:
                try:
                    mean_val = comprehensive_df[col].mean()
                    std_val = comprehensive_df[col].std()
                    min_val = comprehensive_df[col].min()
                    max_val = comprehensive_df[col].max()
                    
                    f.write(f"\n{col}:\n")
                    f.write(f"  Mean:     {mean_val:>10.6f}\n")
                    f.write(f"  Std:      {std_val:>10.6f}\n")
                    f.write(f"  Min:      {min_val:>10.6f}\n")
                    f.write(f"  Max:      {max_val:>10.6f}\n")
                except:
                    pass
    
    print(f"  ✓ Summary statistics saved to: {summary_file}")
    
    return comp_csv


def generate_detailed_comparison_table(comparison_results, output_dir="results/research_comparison"):
    """Generate detailed comparison tables for ALL models with ALL metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Generating comprehensive comparison tables for all models/metrics...")
    
    # Create master DataFrame
    df_all = pd.DataFrame(comparison_results)
    
    # Save complete dataset
    complete_csv = output_dir / "all_models_complete_data.csv"
    df_all.to_csv(complete_csv, index=False)
    print(f"  ✓ Complete data saved to: {complete_csv}")
    
    # Identify numeric and categorical columns
    numeric_cols = df_all.select_dtypes(include=['float64', 'int64']).columns.tolist()
    model_cols = [col for col in df_all.columns if col not in numeric_cols]
    
    # Create organized comparison table (key metrics subset)
    key_metrics = ['f1_macro_mean', 'f1_macro_std', 'f1_micro_mean', 'f1_micro_std',
                  'precision_macro_mean', 'precision_micro_mean', 
                  'recall_macro_mean', 'recall_micro_mean',
                  'accuracy_mean', 'accuracy_std',
                  'hamming_loss_mean', 'hamming_loss_std',
                  'subset_accuracy_mean', 'subset_accuracy_std']
    
    table_data = []
    for idx, result in enumerate(comparison_results, 1):
        row = {'Rank': idx, 'Model': result.get('model', 'Unknown')[:40]}
        
        # Add metrics if available
        for metric in key_metrics:
            if metric in result:
                row[metric] = result[metric]
        
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    
    # Save key metrics table
    key_metrics_csv = output_dir / "key_models_metrics_table.csv"
    df_table.to_csv(key_metrics_csv, index=False)
    print(f"  ✓ Key metrics table saved to: {key_metrics_csv}")
    
    # Generate LaTeX table for all metrics
    latex_file = output_dir / "comprehensive_model_comparison.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\tiny\n")
        f.write("\\begin{tabular}{|l|")
        f.write("r" * len([c for c in df_table.columns if c != 'Model']))
        f.write("|}\n")
        f.write("\\hline\n")
        
        # Header
        f.write("Model")
        for col in df_table.columns:
            if col not in ['Rank', 'Model']:
                f.write(f" & {col[:20]}")
        f.write(" \\\\\n")
        f.write("\\hline\n")
        
        # Data rows
        for _, row in df_table.iterrows():
            f.write(str(row['Model'])[:35])
            for col in df_table.columns:
                if col not in ['Rank', 'Model']:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        f.write(f" & {val:.4f}")
                    else:
                        f.write(f" & {val}")
            f.write(" \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comprehensive Model Comparison: All Models and Key Metrics}\n")
        f.write("\\end{table}\n")
    
    print(f"  ✓ LaTeX table saved to: {latex_file}")
    
    # Generate per-category summary tables
    categories = {
        'Machine Learning': ['Linear SVM', 'Logistic Regression', 'Naive Bayes'],
        'Deep Learning': ['CNN', 'LSTM', 'BiLSTM'],
        'Transformers': ['BERT', 'RoBERTa'],
        'LLM': ['llama', 'LLM'],
    }
    
    for category, keywords in categories.items():
        cat_results = [r for r in comparison_results 
                      if any(kw.lower() in r.get('model', '').lower() for kw in keywords)]
        
        if cat_results:
            cat_df = pd.DataFrame(cat_results)
            cat_csv = output_dir / f"category_{category.lower().replace(' ', '_')}_models.csv"
            cat_df.to_csv(cat_csv, index=False)
            print(f"  ✓ {category} models table saved to: {cat_csv}")
    
    print(f"\n  ✓ All comprehensive tables generated successfully")
    
    return complete_csv, key_metrics_csv, latex_file
