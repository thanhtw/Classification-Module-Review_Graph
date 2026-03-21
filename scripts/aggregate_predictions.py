#!/usr/bin/env python3
"""
Aggregate predictions and labels from all folds for each model.

This script collects predictions and true labels from all folds (stored as .npy files)
and combines them into single arrays for model-wide analysis and confusion matrix generation.

Structure:
    - Scans: results/modular_multimodel/model_artifacts/{model_name}/fold_{i}/predictions.npy
    - Outputs: results/modular_multimodel/model_artifacts/{model_name}/all_folds_predictions.npy
    - Outputs: results/modular_multimodel/model_artifacts/{model_name}/all_folds_labels.npy
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
MODEL_NAMES = [
    'bert', 'roberta', 'lstm', 'bilstm', 'cnn_attention',
    'linear_svm', 'logistic_regression', 'naive_bayes',
    'llm_few_shot', 'llm_zero_shot'
]
ARTIFACTS_ROOT = Path('results/modular_multimodel/model_artifacts')


def load_predictions_and_labels(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and labels from all folds for a model.
    
    Args:
        model_name: Name of the model (e.g., 'bert', 'lstm')
        
    Returns:
        all_predictions: Combined predictions from all folds (n_samples, n_labels)
        all_labels: Combined true labels from all folds (n_samples, n_labels)
    """
    model_dir = ARTIFACTS_ROOT / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find all fold directories
    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in: {model_dir}")
    
    all_preds = []
    all_labels = []
    
    print(f"  Loading from {len(fold_dirs)} folds...", end="")
    
    for fold_dir in fold_dirs:
        pred_path = fold_dir / "predictions.npy"
        label_path = fold_dir / "labels.npy"
        
        if not pred_path.exists() or not label_path.exists():
            print(f"\n    ⚠️  Skipping {fold_dir.name}: missing predictions.npy or labels.npy")
            continue
        
        try:
            preds = np.load(pred_path)
            labels = np.load(label_path)
            all_preds.append(preds)
            all_labels.append(labels)
        except Exception as e:
            print(f"\n    ⚠️  Error loading {fold_dir.name}: {e}")
            continue
    
    if not all_preds:
        raise ValueError(f"No predictions loaded for {model_name}")
    
    # Concatenate all folds
    combined_preds = np.vstack(all_preds)
    combined_labels = np.vstack(all_labels)
    
    print(f" ✓ Loaded {len(all_preds)} folds")
    print(f"     Total samples: {combined_preds.shape[0]}")
    print(f"     Labels shape: {combined_preds.shape}")
    
    return combined_preds, combined_labels


def save_aggregated_predictions(model_name: str, all_preds: np.ndarray, all_labels: np.ndarray) -> None:
    """
    Save aggregated predictions and labels for a model.
    
    Args:
        model_name: Name of the model
        all_preds: Combined predictions array
        all_labels: Combined true labels array
    """
    model_dir = ARTIFACTS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy files
    preds_path = model_dir / "all_folds_predictions.npy"
    labels_path = model_dir / "all_folds_labels.npy"
    
    np.save(str(preds_path), all_preds)
    np.save(str(labels_path), all_labels)
    
    print(f"     Saved {preds_path.name} ({all_preds.nbytes / 1024 / 1024:.2f} MB)")
    print(f"     Saved {labels_path.name} ({all_labels.nbytes / 1024 / 1024:.2f} MB)")


def aggregate_all_models() -> None:
    """Aggregate predictions and labels for all models."""
    print("\n" + "="*80)
    print("AGGREGATING PREDICTIONS AND LABELS FROM ALL FOLDS")
    print("="*80 + "\n")
    
    if not ARTIFACTS_ROOT.exists():
        print(f"❌ Artifacts root not found: {ARTIFACTS_ROOT}")
        print("Run training first with: python scripts/train.py")
        return
    
    success_count = 0
    failed_models = []
    
    for model_name in MODEL_NAMES:
        print(f"Processing {model_name}...", end="")
        
        try:
            all_preds, all_labels = load_predictions_and_labels(model_name)
            save_aggregated_predictions(model_name, all_preds, all_labels)
            success_count += 1
            print()
        except FileNotFoundError as e:
            print(f" ⚠️  {e}")
            failed_models.append(model_name)
        except ValueError as e:
            print(f" ❌ {e}")
            failed_models.append(model_name)
        except Exception as e:
            print(f" ❌ Unexpected error: {e}")
            failed_models.append(model_name)
    
    # Summary
    print("\n" + "="*80)
    print(f"✅ Aggregation complete")
    print("="*80)
    print(f"   Successful: {success_count}/{len(MODEL_NAMES)} models")
    
    if failed_models:
        print(f"   Failed: {failed_models}")
        print("\n   💡 Run training for missing models with:")
        print(f"      python scripts/train.py --models {' '.join(failed_models)}")
    
    print(f"\n📁 All aggregated files saved to: {ARTIFACTS_ROOT}")
    print("   Files: all_folds_predictions.npy, all_folds_labels.npy")
    print("\n💡 Use these for confusion matrix generation:")
    print("   python scripts/generate_confusion_matrices.py")


if __name__ == '__main__':
    aggregate_all_models()
