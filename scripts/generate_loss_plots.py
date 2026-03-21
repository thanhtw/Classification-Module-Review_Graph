#!/usr/bin/env python3
"""
Generate training loss vs epochs plots for each model.
Shows training loss vs validation loss across epochs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Configure matplotlib for publication quality
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100


def load_training_histories(model_name, model_dir):
    """Load all training histories for a model across folds"""
    histories = []
    fold_dirs = sorted([d for d in model_dir.glob("fold_*") if d.is_dir()])
    
    for fold_dir in fold_dirs:
        history_file = fold_dir / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                histories.append(history)
            except Exception as e:
                print(f"  ⚠️  Could not load {fold_dir.name}: {e}")
    
    return histories


def average_histories(histories):
    """Average training histories across folds"""
    if not histories:
        return None
    
    # Get lengths of all histories
    lengths = [len(h.get('train_loss', [])) for h in histories]
    
    if not lengths:
        return None
    
    # Use the minimum length to avoid misalignment
    min_length = min(lengths)
    
    # Average loss values
    train_losses = []
    val_losses = []
    
    for epoch_idx in range(min_length):
        epoch_train_losses = []
        epoch_val_losses = []
        
        for history in histories:
            if epoch_idx < len(history.get('train_loss', [])):
                epoch_train_losses.append(history['train_loss'][epoch_idx])
            if epoch_idx < len(history.get('val_loss', [])):
                epoch_val_losses.append(history['val_loss'][epoch_idx])
        
        if epoch_train_losses:
            train_losses.append(np.mean(epoch_train_losses))
        if epoch_val_losses:
            val_losses.append(np.mean(epoch_val_losses))
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }


def create_loss_plot(model_name, avg_history, output_path):
    """Create loss vs epochs plot"""
    train_losses = avg_history.get('train_loss', [])
    val_losses = avg_history.get('val_loss', [])
    
    if not train_losses or not val_losses:
        return None
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss curves
    ax.plot(epochs, train_losses, 'o-', linewidth=2.5, markersize=5, 
            label='training loss', color='#3498DB')
    ax.plot(epochs, val_losses, 's-', linewidth=2.5, markersize=5, 
            label='validation loss', color='#E67E22')
    
    # Set labels and title
    ax.set_xlabel('epochs', fontsize=12, fontweight='bold')
    ax.set_ylabel('loss', fontsize=12, fontweight='bold')
    ax.set_title('loss vs epochs', fontsize=13, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(fontsize=11, loc='upper right')
    
    # Set background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def generate_loss_plots(output_dir="results/research_comparison"):
    """Generate loss vs epochs plots for all models with training data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loss_output_dir = output_dir / "training_loss_plots"
    loss_output_dir.mkdir(parents=True, exist_ok=True)
    
    model_artifacts_dir = Path("results/modular_multimodel/model_artifacts")
    model_dirs = sorted([d for d in model_artifacts_dir.iterdir() if d.is_dir()])
    
    print(f"\n📈 Generating training loss plots for models with history data...")
    print(f"Output directory: {loss_output_dir}\n")
    
    success_count = 0
    skipped_count = 0
    
    for model_idx, model_dir in enumerate(model_dirs):
        model_name = model_dir.name
        print(f"[{model_idx+1}/{len(model_dirs)}] Processing {model_name}...", end=" ", flush=True)
        
        try:
            # Load training histories from all folds
            histories = load_training_histories(model_name, model_dir)
            
            if not histories:
                print("⊘ (no training history)")
                skipped_count += 1
                continue
            
            # Average histories across folds
            avg_history = average_histories(histories)
            
            if not avg_history:
                print("❌ (could not average histories)")
                continue
            
            # Create and save plot
            plot_file = loss_output_dir / f"loss_plot_{model_name}.png"
            create_loss_plot(model_name, avg_history, plot_file)
            
            print(f"✓ Saved")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}")
            continue
    
    print(f"\n✅ Generated loss plots for {success_count}/{len(model_dirs)} models")
    print(f"⊘ Skipped {skipped_count} models (no training history)")
    print(f"📁 Output saved to: {loss_output_dir}\n")
    
    return loss_output_dir


if __name__ == "__main__":
    output_base = "results/research_comparison"
    
    print("=" * 80)
    print("TRAINING LOSS VISUALIZATION GENERATOR")
    print("=" * 80)
    
    result = generate_loss_plots(output_base)
    
    print("=" * 80)
    print("✅ LOSS PLOT GENERATION COMPLETE")
    print("=" * 80)
