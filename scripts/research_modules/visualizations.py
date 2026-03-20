"""Module for generating visualizations (SMOTE, confusion matrices, training curves, heatmaps)"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.config import LABEL_COLUMNS


def load_smote_analysis(results_dir="results/modular_multimodel/global_train_data_analysis"):
    """Load SMOTE analysis data from results directory"""
    smote_file = Path(results_dir) / "train_smote_analysis_summary.json"
    
    if smote_file.exists():
        with open(smote_file, 'r') as f:
            return json.load(f)
    return None


def generate_smote_visualization(output_dir="results/research_comparison"):
    """Generate SMOTE class distribution visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    smote_data = load_smote_analysis()
    if not smote_data:
        print("⚠ SMOTE analysis file not found")
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SMOTE Impact on Dataset Balance', fontsize=16, fontweight='bold')
    
    labels = LABEL_COLUMNS
    before_pos = smote_data.get('label_pos_before', [])
    before_neg = smote_data.get('label_neg_before', [])
    after_pos = smote_data.get('label_pos_after', [])
    after_neg = smote_data.get('label_neg_after', [])
    
    # Plot for each label
    for idx, label in enumerate(labels):
        ax_before = axes[0, idx]
        ax_after = axes[1, idx]
        
        # Before SMOTE
        before_x = [f'{label}\n(Neg)', f'{label}\n(Pos)']
        before_y = [before_neg[idx], before_pos[idx]]
        colors_before = ['#FF6B6B', '#4ECDC4']
        ax_before.bar(before_x, before_y, color=colors_before, edgecolor='black', linewidth=1.5)
        ax_before.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax_before.set_title(f'{label} - Before SMOTE', fontsize=12, fontweight='bold')
        ax_before.set_ylim(0, max(after_pos + after_neg) * 1.1)
        # Add value labels on bars
        for i, v in enumerate(before_y):
            ax_before.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        ax_before.grid(axis='y', alpha=0.3, linestyle='--')
        
        # After SMOTE
        after_x = [f'{label}\n(Neg)', f'{label}\n(Pos)']
        after_y = [after_neg[idx], after_pos[idx]]
        colors_after = ['#95E1D3', '#F38181']
        ax_after.bar(after_x, after_y, color=colors_after, edgecolor='black', linewidth=1.5)
        ax_after.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax_after.set_title(f'{label} - After SMOTE (Balanced)', fontsize=12, fontweight='bold')
        ax_after.set_ylim(0, max(after_pos + after_neg) * 1.1)
        # Add value labels on bars
        for i, v in enumerate(after_y):
            ax_after.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        ax_after.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    smote_vis_path = output_dir / "smote_impact_visualization.png"
    plt.savefig(smote_vis_path, dpi=300, bbox_inches='tight')
    print(f"✓ SMOTE visualization saved to: {smote_vis_path}")
    plt.close()
    
    return smote_vis_path


def generate_confusion_matrix_visualizations(output_dir="results/research_comparison"):
    """Generate confusion matrix heatmaps for ALL models"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Generating confusion matrices for ALL models...")
    
    model_artifacts_dir = Path("results/modular_multimodel/model_artifacts")
    if not model_artifacts_dir.exists():
        print("⚠ Model artifacts directory not found - skipping confusion matrix visualization")
        return None
    
    # Collect all models
    model_dirs = sorted([d for d in model_artifacts_dir.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print("⚠ No model directories found")
        return None
    
    num_models = len(model_dirs)
    cols = 3
    rows = (num_models + cols - 1) // cols  # Ceil division for grid layout
    
    confusion_fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape((rows, cols))
    
    confusion_fig.suptitle(f'Confusion Matrices: All {num_models} Models', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    success_count = 0
    for model_idx, model_dir in enumerate(model_dirs):
        model_name = model_dir.name
        
        try:
            # Try to find predictions files
            pred_files = list(model_dir.glob("*/predictions*.npy"))
            
            if pred_files:
                # Load predictions from first fold
                pred_file = pred_files[0]
                y_pred = np.load(pred_file)
                
                # Try to find true labels
                label_files = list(model_dir.parent.glob("**/y_val*.npy"))
                if not label_files:
                    label_files = list(model_dir.glob("*/y_val*.npy"))
                
                if label_files:
                    y_true = np.load(label_files[0])
                    
                    # For multilabel, use micro-averaged confusion matrix
                    from sklearn.metrics import confusion_matrix
                    y_true_flat = y_true.ravel()
                    y_pred_flat = y_pred.ravel()
                    
                    cm = confusion_matrix(y_true_flat, y_pred_flat)
                    
                    # Plot heatmap
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[model_idx],
                               cbar=True, xticklabels=['Negative', 'Positive'], 
                               yticklabels=['Negative', 'Positive'], vmin=0)
                    axes[model_idx].set_title(f'{model_name}', fontsize=11, fontweight='bold')
                    axes[model_idx].set_ylabel('True Label', fontweight='bold', fontsize=9)
                    axes[model_idx].set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
                    success_count += 1
                    print(f"  ✓ {model_name}")
        except Exception as e:
            print(f"  ⚠ {model_name}: {e}")
    
    # Remove unused subplots
    for idx in range(num_models, len(axes)):
        confusion_fig.delaxes(axes[idx])
    
    if success_count > 0:
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrices_all_models.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrices for {success_count} models saved to: {cm_path}")
        plt.close()
        return cm_path
    else:
        plt.close()
        print("✗ No confusion matrices could be generated")
        return None


def generate_training_curves(output_dir="results/research_comparison"):
    """Generate training curves (F1-Macro/F1-Micro & Loss across epochs) if history data available"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📈 Searching for training history data...")
    
    # Try to find training history files
    history_files = list(Path("results").glob("**/training_history*.json"))
    
    if not history_files:
        print("⚠ No training history files found - skipping training curves")
        return None
    
    # Load and visualize first available history
    try:
        with open(history_files[0], 'r') as f:
            history = json.load(f)
        
        # Create figure with F1-Macro/F1-Micro and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training Curves: Deep Learning Model', fontsize=14, fontweight='bold')
        
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        
        # F1-Macro and F1-Micro plot
        ax1.plot(epochs, history.get('train_f1_macro', []), 'o-', linewidth=2, markersize=4, label='Training F1-Macro', color='#3498DB')
        ax1.plot(epochs, history.get('val_f1_macro', []), 's-', linewidth=2, markersize=4, label='Validation F1-Macro', color='#E74C3C')
        ax1.plot(epochs, history.get('train_f1_micro', []), 'o--', linewidth=2, markersize=4, label='Training F1-Micro', color='#2ECC71', alpha=0.7)
        ax1.plot(epochs, history.get('val_f1_micro', []), 's--', linewidth=2, markersize=4, label='Validation F1-Micro', color='#F39C12', alpha=0.7)
        ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
        ax1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax1.set_title('(a) F1-Macro & F1-Micro', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Loss plot
        ax2.plot(epochs, history.get('train_loss', []), 'o-', linewidth=2, markersize=4, label='Training loss', color='#3498DB')
        ax2.plot(epochs, history.get('val_loss', []), 's-', linewidth=2, markersize=4, label='Validation loss', color='#E74C3C')
        ax2.set_xlabel('Epochs', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_path = output_dir / "training_curves.png"
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to: {curves_path}")
        plt.close()
        
        return curves_path
    except Exception as e:
        print(f"⚠ Could not generate training curves: {e}")
        return None


def generate_comprehensive_heatmaps(comparison_results, output_dir="results/research_comparison"):
    """Generate comprehensive heatmaps showing all models vs all metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔥 Generating comprehensive metric heatmaps...")
    
    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    
    # Extract numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Get model names
    models = df['model'].values
    
    # Select key metrics for heatmap
    key_metrics = [col for col in numeric_df.columns 
                   if any(m in col.lower() for m in ['f1', 'precision', 'recall', 'accuracy', 'hamming', 'subset'])]
    
    if len(key_metrics) > 0:
        # Create heatmap data
        heatmap_df = numeric_df[key_metrics].copy()
        heatmap_df.index = models
        
        # Normalize to 0-1 range for better visualization
        heatmap_normalized = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min() + 1e-10)
        
        # Create large figure for all metrics
        fig, ax = plt.subplots(figsize=(max(16, len(key_metrics)), max(12, len(models))))
        
        sns.heatmap(heatmap_normalized, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Normalized Score'}, linewidths=0.5)
        
        ax.set_title(f'Comprehensive Metric Heatmap: All {len(models)} Models × {len(key_metrics)} Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Models', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_path = output_dir / "comprehensive_metrics_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Comprehensive heatmap saved to: {heatmap_path}")
        plt.close()
        
        return heatmap_path
    
    return None


def generate_model_comparison_visualizations(comparison_results, output_dir="results/research_comparison"):
    """Generate model performance comparison visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not comparison_results:
        print("⚠ No comparison results to visualize")
        return
    
    # Sort by F1-macro
    sorted_results = sorted(comparison_results, key=lambda x: x.get('f1_macro_mean', 0), reverse=True)
    
    # Prepare data
    models = [r['model'][:30] for r in sorted_results]  # Truncate for readability
    f1_macro = [r.get('f1_macro_mean', 0) for r in sorted_results]
    f1_micro = [r.get('f1_micro_mean', 0) for r in sorted_results]
    hamming_loss = [r.get('hamming_loss_mean', 0) for r in sorted_results]
    subset_acc = [r.get('subset_accuracy_mean', 0) for r in sorted_results]
    
    # **Figure 1: F1 Score Comparison**
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_macro, width, label='F1-Macro', color='#3498DB', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, f1_micro, width, label='F1-Micro', color='#2ECC71', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: F1-Macro vs F1-Micro', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig_path = output_dir / "model_f1_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ F1 comparison saved to: {fig_path}")
    plt.close()
    
    # **Figure 2: Hamming Loss vs Subset Accuracy**
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Hamming Loss
    bars_hl = ax1.barh(models, hamming_loss, color='#E74C3C', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Hamming Loss (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Hamming Loss by Model', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars_hl, hamming_loss)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Subset Accuracy
    bars_sa = ax2.barh(models, subset_acc, color='#9B59B6', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Subset Accuracy (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('Subset Accuracy by Model', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars_sa, subset_acc)):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig_path = output_dir / "model_multilabel_metrics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Multilabel metrics saved to: {fig_path}")
    plt.close()
