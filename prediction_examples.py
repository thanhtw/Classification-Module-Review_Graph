#!/usr/bin/env python3
"""
PRACTICAL PYTHON EXAMPLES: PREDICTION FORMATS

This file contains runnable Python code examples showing how to:
1. Load predictions from each model type
2. Inspect prediction formats
3. Perform operations on predictions
4. Generate confusion matrices
"""

import numpy as np
from pathlib import Path

# ============================================================================
# EXAMPLE 1: Load and Inspect ML Model Predictions
# ============================================================================

def example_load_ml_predictions():
    """Load LinearSVM predictions and inspect format."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Machine Learning Model (LinearSVM) Predictions")
    print("="*70)
    
    model_name = 'linear_svm'
    fold_num = 1
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    
    try:
        preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
        labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')
    except FileNotFoundError:
        print(f"Files not found. Run training first:")
        print(f"  python scripts/train.py --models linear_svm --n_folds 2")
        return
    
    print(f"\nPredictions array:")
    print(f"  Shape: {preds.shape}")          # Expected: (240, 3) approx
    print(f"  Dtype: {preds.dtype}")          # Expected: int64
    print(f"  Min value: {preds.min()}")      # Expected: 0
    print(f"  Max value: {preds.max()}")      # Expected: 1
    print(f"  Unique values: {np.unique(preds)}")  # Expected: [0 1]
    
    print(f"\nFirst 10 samples:")
    print(preds[:10])
    
    print(f"\nLabel distribution:")
    for label_idx, label_name in enumerate(['relevance', 'concreteness', 'constructive']):
        pos_count = np.sum(preds[:, label_idx] == 1)
        neg_count = np.sum(preds[:, label_idx] == 0)
        print(f"  {label_name:15} → Positive: {pos_count:3d}, Negative: {neg_count:3d}")
    
    # Compute accuracy
    accuracy = np.mean(preds == labels)
    print(f"\nAccuracy (all labels): {accuracy:.4f}")
    
    # Per-label accuracy
    for label_idx, label_name in enumerate(['relevance', 'concreteness', 'constructive']):
        acc = np.mean(preds[:, label_idx] == labels[:, label_idx])
        print(f"  {label_name} accuracy: {acc:.4f}")


# ============================================================================
# EXAMPLE 2: Load and Inspect Transformer Model Predictions
# ============================================================================

def example_load_transformer_predictions():
    """Load BERT predictions and inspect sigmoid/threshold process."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Transformer Model (BERT) Predictions")
    print("="*70)
    
    model_name = 'bert'
    fold_num = 1
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    
    try:
        preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
        labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')
    except FileNotFoundError:
        print(f"Files not found. Run training first:")
        print(f"  python scripts/train.py --models bert --n_folds 2")
        return
    
    print(f"\nPredictions array (after sigmoid + threshold):")
    print(f"  Shape: {preds.shape}")          # Expected: (240, 3)
    print(f"  Dtype: {preds.dtype}")          # Expected: int64
    print(f"  Unique values: {np.unique(preds)}")  # Expected: [0 1]
    
    print(f"\nFirst 10 samples (after threshold at 0.5):")
    print(preds[:10])
    
    print(f"\nNote: BERT internally:")
    print(f"  1. Encodes text with 12 transformer layers")
    print(f"  2. Gets [CLS] token embedding (768-dim)")
    print(f"  3. Dense layer → logits (3-dim, unbounded)")
    print(f"  4. Sigmoid: sigmoid(logits) ∈ [0, 1]")
    print(f"  5. Threshold: > 0.5 → 1, else → 0")
    print(f"  6. Result stored as binary int64 array")
    
    print(f"\nComparison with true labels:")
    print(f"  Sample predictions: {preds[0]}")
    print(f"  True labels:        {labels[0]}")
    print(f"  Match: {np.array_equal(preds[0], labels[0])}")


# ============================================================================
# EXAMPLE 3: Load and Compare Neural Network Predictions
# ============================================================================

def example_load_nn_predictions():
    """Load LSTM predictions (similar to CNN, BiLSTM)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Neural Network Model (LSTM) Predictions")
    print("="*70)
    
    model_name = 'lstm'
    fold_num = 1
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    
    try:
        preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
        labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')
    except FileNotFoundError:
        print(f"Files not found. Run training first:")
        print(f"  python scripts/train.py --models lstm --n_folds 2")
        return
    
    print(f"\nPredictions array:")
    print(f"  Shape: {preds.shape}")
    print(f"  Dtype: {preds.dtype}")
    
    print(f"\nLSTM pipeline:")
    print(f"  1. Text → Tokenize → Word IDs")
    print(f"  2. Embedding layer: (seq_len, 300)")
    print(f"  3. LSTM layer: Process sequence → hidden states")
    print(f"  4. Mean pooling: Aggregate → (128 or 256-dim)")
    print(f"  5. Dropout + Dense: → logits (3-dim)")
    print(f"  6. Sigmoid + Threshold (same as BERT)")
    print(f"  7. Final binary output: (n_samples, 3)")
    
    print(f"\nFirst 5 samples:")
    print(preds[:5])
    
    print(f"\nBiLSTM would be similar but:")
    print(f"  - Bidirectional processing → 256-dim instead of 128-dim")
    print(f"  - Rest of pipeline identical")


# ============================================================================
# EXAMPLE 4: Load and Inspect LLM Predictions
# ============================================================================

def example_load_llm_predictions():
    """Load LLM predictions (JSON parsed format)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Language Model (LLM) Predictions")
    print("="*70)
    
    model_name = 'llm_few_shot'
    fold_num = 1
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    
    try:
        preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
        labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')
    except FileNotFoundError:
        print(f"Files not found. Run training first:")
        print(f"  python scripts/train.py --models llm_few_shot --n_folds 2")
        return
    
    print(f"\nPredictions array:")
    print(f"  Shape: {preds.shape}")
    print(f"  Dtype: {preds.dtype}")
    
    print(f"\nLLM pipeline:")
    print(f"  1. Build prompt with text + optional few-shot examples")
    print(f"  2. Call Groq API: {{'role': 'user', 'content': prompt}}")
    print(f"  3. LLM generates: JSON with relevance, concreteness, constructive")
    print(f"  4. Parse JSON: Extract values from response")
    print(f"  5. Convert to binary: _safe_int01(value) → 0 or 1")
    print(f"  6. Return numpy array")
    
    print(f"\nFirst 10 samples:")
    print(preds[:10])
    
    print(f"\nEdge case handling:")
    print(f"  - Invalid JSON → [0, 0, 0] (default)")
    print(f"  - Non-integer values → _safe_int01() converts")
    print(f"  - Missing keys → 0 (default)")
    
    print(f"\nComparison with few-shot vs zero-shot:")
    print(f"  Few-shot: Includes examples in prompt → typically higher accuracy")
    print(f"  Zero-shot: No examples → faster but less accurate")


# ============================================================================
# EXAMPLE 5: Compare Predictions Across All Models (One Fold)
# ============================================================================

def example_compare_all_models():
    """Load predictions from all 10 models for the same fold and compare."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Compare Predictions Across All Models")
    print("="*70)
    
    models = [
        'bert', 'roberta', 'lstm', 'bilstm', 'cnn_attention',
        'linear_svm', 'logistic_regression', 'naive_bayes',
        'llm_few_shot', 'llm_zero_shot'
    ]
    
    fold_num = 1
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    
    # Try to load predictions for all models
    predictions = {}
    labels = None
    
    for model_name in models:
        try:
            preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
            labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')
            predictions[model_name] = preds
        except FileNotFoundError:
            pass
    
    if not predictions:
        print("No predictions found. Run training with multiple models:")
        print("  python scripts/train.py --n_folds 2")
        return
    
    print(f"\nLoaded predictions from {len(predictions)}/{len(models)} models\n")
    
    # Compare first sample across all models
    sample_idx = 0
    print(f"First test sample predictions across all models:")
    print(f"True labels: {labels[sample_idx]}\n")
    
    print(f"{'Model':<20} {'Relevance':>8} {'Concrete':>8} {'Construct':>8} {'Match':>8}")
    print("-" * 60)
    
    for model_name in sorted(predictions.keys()):
        pred = predictions[model_name][sample_idx]
        true = labels[sample_idx]
        match = np.array_equal(pred, true)
        match_str = "✓" if match else "✗"
        print(f"{model_name:<20} {pred[0]:>8} {pred[1]:>8} {pred[2]:>8} {match_str:>8}")
    
    # Compute agreement between all models
    print(f"\nAgreement analysis (how often models agree on predictions):")
    
    # Stack all predictions
    all_preds = np.stack([predictions[m] for m in sorted(predictions.keys())], axis=0)
    # (n_models, n_samples, 3)
    
    # For each label, compute agreement
    for label_idx, label_name in enumerate(['relevance', 'concreteness', 'constructive']):
        label_preds = all_preds[:, :, label_idx]  # (n_models, n_samples)
        
        # Count how many models predict each sample
        agreement = np.mean(label_preds == label_preds[0], axis=0)  # Against first model
        
        print(f"\n  {label_name}:")
        print(f"    Models agreeing with first model: {np.mean(agreement):.1%}")
        print(f"    High agreement (≥80%): {np.sum(agreement >= 0.8)}/{len(agreement)}")
        print(f"    Low agreement (<50%): {np.sum(agreement < 0.5)}/{len(agreement)}")


# ============================================================================
# EXAMPLE 6: Generate Confusion Matrix from Predictions
# ============================================================================

def example_confusion_matrix():
    """Generate and display confusion matrix for one model."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Generate Confusion Matrix from Predictions")
    print("="*70)
    
    from sklearn.metrics import confusion_matrix
    
    model_name = 'bert'
    fold_num = 1
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    
    try:
        preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
        labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')
    except FileNotFoundError:
        print(f"Files not found. Run training first:")
        print(f"  python scripts/train.py --models bert --n_folds 2")
        return
    
    print(f"\nModel: {model_name}, Fold: {fold_num}\n")
    
    # Generate confusion matrix for each label
    for label_idx, label_name in enumerate(['relevance', 'concreteness', 'constructive']):
        y_true = labels[:, label_idx]
        y_pred = preds[:, label_idx]
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        tn, fp, fn, tp = cm.ravel()
        
        print(f"{label_name.upper()}:")
        print(f"  Confusion Matrix:")
        print(f"    Predicted:  0      1")
        print(f"  True 0:     {tn:4d}  {fp:4d}  (TN=True Negative, FP=False Positive)")
        print(f"  True 1:     {fn:4d}  {tp:4d}  (FN=False Negative, TP=True Positive)")
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Metrics: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        print()


# ============================================================================
# EXAMPLE 7: Aggregate Predictions from All Folds
# ============================================================================

def example_aggregate_folds():
    """Load and aggregate predictions from all 10 folds."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Aggregate Predictions from All Folds")
    print("="*70)
    
    model_name = 'bert'
    artifacts_path = Path('results/modular_multimodel/model_artifacts')
    model_dir = artifacts_path / model_name
    
    # Find all fold directories
    fold_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        print(f"No folds found. Check {model_dir}")
        return
    
    print(f"Found {len(fold_dirs)} folds for {model_name}\n")
    
    # Load individual folds
    all_folds = []
    for fold_dir in fold_dirs:
        try:
            preds = np.load(fold_dir / 'predictions.npy')
            all_folds.append((fold_dir.name, preds.shape[0]))
        except FileNotFoundError:
            pass
    
    if all_folds:
        print("Fold sizes:")
        total_samples = 0
        for fold_name, fold_size in all_folds:
            print(f"  {fold_name}: {fold_size} samples")
            total_samples += fold_size
        print(f"  Total: {total_samples} samples\n")
    
    # Check for aggregated file
    agg_preds_path = model_dir / 'all_folds_predictions.npy'
    agg_labels_path = model_dir / 'all_folds_labels.npy'
    
    if agg_preds_path.exists() and agg_labels_path.exists():
        agg_preds = np.load(agg_preds_path)
        agg_labels = np.load(agg_labels_path)
        
        print(f"Aggregated file found:")
        print(f"  all_folds_predictions.npy: {agg_preds.shape}")
        print(f"  all_folds_labels.npy: {agg_labels.shape}")
        print(f"  Total samples: {agg_preds.shape[0]}")
    else:
        print(f"Aggregated files not found. Run:")
        print(f"  python scripts/aggregate_predictions.py")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == '__main__':
    print("\n" + "█"*70)
    print("PREDICTION FORMAT EXAMPLES")
    print("█"*70)
    print("\nNote: Run these after training the models with:")
    print("  python scripts/train.py --n_folds 2")
    
    # Run examples (skip if files not found)
    try:
        example_load_ml_predictions()
    except Exception as e:
        print(f"\nSkipping Example 1: {e}")
    
    try:
        example_load_transformer_predictions()
    except Exception as e:
        print(f"\nSkipping Example 2: {e}")
    
    try:
        example_load_nn_predictions()
    except Exception as e:
        print(f"\nSkipping Example 3: {e}")
    
    try:
        example_load_llm_predictions()
    except Exception as e:
        print(f"\nSkipping Example 4: {e}")
    
    try:
        example_compare_all_models()
    except Exception as e:
        print(f"\nSkipping Example 5: {e}")
    
    try:
        example_confusion_matrix()
    except Exception as e:
        print(f"\nSkipping Example 6: {e}")
    
    try:
        example_aggregate_folds()
    except Exception as e:
        print(f"\nSkipping Example 7: {e}")
    
    print("\n" + "█"*70 + "\n")
