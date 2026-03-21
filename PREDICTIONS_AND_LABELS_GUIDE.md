#!/usr/bin/env python3
"""
Quick documentation and usage guide for predictions and labels aggregation.

This file explains the new system for storing and aggregating model predictions
and labels from each training fold.
"""

# ============================================================================
# PREDICTIONS AND LABELS STORAGE SYSTEM
# ============================================================================
#
# After training each model on each fold, predictions and true labels are
# automatically saved in numpy format for later confusion matrix generation.
#
# ============================================================================
# FILE STRUCTURE
# ============================================================================
#
# results/modular_multimodel/model_artifacts/
# ├── bert/
# │   ├── fold_1/
# │   │   ├── pytorch_model.bin
# │   │   ├── config.json
# │   │   ├── predictions.npy       ← Predictions for fold 1
# │   │   ├── labels.npy            ← True labels for fold 1
# │   │   └── metadata.json
# │   ├── fold_2/
# │   │   ├── pytorch_model.bin
# │   │   ├── predictions.npy       ← Predictions for fold 2
# │   │   ├── labels.npy            ← True labels for fold 2
# │   │   └── metadata.json
# │   ├── ...
# │   ├── all_folds_predictions.npy ← Aggregated from all folds
# │   └── all_folds_labels.npy      ← Aggregated from all folds
# │
# ├── lstm/
# ├── bilstm/
# ├── roberta/
# ├── linear_svm/
# ├── logistic_regression/
# ├── naive_bayes/
# ├── cnn_attention/
# ├── llm_few_shot/
# └── llm_zero_shot/
#
# ============================================================================
# WHAT'S NEW (CHANGES MADE)
# ============================================================================
#
# 1. AUTOMATED SAVING (During Training)
#    ✓ models_ml.py       - LinearSVM, NaiveBayes, LogisticRegression
#    ✓ models_transformers.py - BERT, RoBERTa
#    ✓ models_nn.py       - LSTM, BiLSTM, CNN Attention
#    ✓ models_llm.py      - LLM Zero-Shot, LLM Few-Shot
#
#    Each model now saves:
#    - predictions.npy: Binary multilabel predictions (shape: n_samples × 3)
#    - labels.npy:      True multilabel targets (shape: n_samples × 3)
#
# 2. AGGREGATION SCRIPT (After Training)
#    ✓ scripts/aggregate_predictions.py
#
#    Combines predictions and labels from all folds into:
#    - all_folds_predictions.npy: Combined predictions across all folds
#    - all_folds_labels.npy:      Combined true labels across all folds
#
# ============================================================================
# USAGE WORKFLOW
# ============================================================================
#
# Step 1: Train all models (saves per-fold predictions automatically)
#    $ python scripts/train.py --models bert roberta lstm bilstm cnn_attention \
#                               linear_svm logistic_regression naive_bayes \
#                               llm_few_shot llm_zero_shot --n_folds 10
#
#    Output: results/modular_multimodel/model_artifacts/{model}/fold_{i}/predictions.npy
#            results/modular_multimodel/model_artifacts/{model}/fold_{i}/labels.npy
#
# Step 2: Aggregate predictions from all folds
#    $ python scripts/aggregate_predictions.py
#
#    Output: results/modular_multimodel/model_artifacts/{model}/all_folds_predictions.npy
#            results/modular_multimodel/model_artifacts/{model}/all_folds_labels.npy
#
# Step 3: Generate confusion matrices using aggregated predictions
#    $ python scripts/generate_confusion_matrices.py         # Use aggregated data
#
# ============================================================================
# DATA FORMAT
# ============================================================================
#
# Each predictions.npy file shape: (n_test_samples, 3)
# Each labels.npy file shape:      (n_test_samples, 3)
#
# Where 3 columns represent:
# [relevance, concreteness, constructive]
#
# Values: Binary (0 or 1) for each sample-label pair
#
# Example predictions for 10 samples:
# [[0, 1, 0],     # Sample 1: not relevant, concrete, not constructive
#  [1, 1, 1],     # Sample 2: relevant, concrete, constructive
#  [1, 0, 1],     # Sample 3: relevant, not concrete, constructive
#  ...
#  [0, 0, 0]]     # Sample 10: none of the labels apply
#
# Same format applies to labels.npy
#
# ============================================================================
# LOADING DATA PROGRAMMATICALLY
# ============================================================================
#
# import numpy as np
# from pathlib import Path
#
# model_name = 'bert'
# artifacts_root = Path('results/modular_multimodel/model_artifacts')
#
# # Load aggregated predictions and labels
# all_preds = np.load(artifacts_root / model_name / 'all_folds_predictions.npy')
# all_labels = np.load(artifacts_root / model_name / 'all_folds_labels.npy')
#
# print(f"Shape: {all_preds.shape}")          # (2398, 3) for 10-fold CV
# print(f"Unique predictions: {np.unique(all_preds)}")  # [0, 1]
#
# # Or load specific fold
# fold_1_preds = np.load(artifacts_root / model_name / 'fold_1' / 'predictions.npy')
# fold_1_labels = np.load(artifacts_root / model_name / 'fold_1' / 'labels.npy')
#
# ============================================================================
# CONFUSION MATRIX GENERATION
# ============================================================================
#
# Use the aggregated predictions and labels to generate confusion matrices:
#
# from sklearn.metrics import confusion_matrix
# import numpy as np
#
# # Load data
# all_preds = np.load('results/modular_multimodel/model_artifacts/bert/all_folds_predictions.npy')
# all_labels = np.load('results/modular_multimodel/model_artifacts/bert/all_folds_labels.npy')
#
# # Generate confusion matrix for each label
# for label_idx, label_name in enumerate(['relevance', 'concreteness', 'constructive']):
#     cm = confusion_matrix(all_labels[:, label_idx], all_preds[:, label_idx])
#     print(f"{label_name}:")
#     print(cm)
#
# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
# Q: Where are the predictions saved?
# A: Automatically in: results/modular_multimodel/model_artifacts/{model}/fold_{i}/
#
# Q: What if I don't see predictions.npy files?
# A: Check that:
#    1. Training ran successfully (check logs for errors)
#    2. Model save_dir was provided (should be automatic in train.py)
#    3. Permissions allow writing to results/ directory
#
# Q: How do I reload predictions for a specific fold?
# A: preds = np.load('results/modular_multimodel/model_artifacts/bert/fold_1/predictions.npy')
#
# Q: Can I verify the data before generating matrices?
# A: Yes! Check shape, data types, and value ranges:
#    preds = np.load('...')
#    print(f"Shape: {preds.shape}, Dtype: {preds.dtype}, Min: {preds.min()}, Max: {preds.max()}")
#
# ============================================================================
# TECHNICAL DETAILS
# ============================================================================
#
# Files Modified:
# - src/models/models_ml.py          → save predictions.npy + labels.npy
# - src/models/models_transformers.py → save predictions.npy + labels.npy
# - src/models/models_nn.py          → save predictions.npy + labels.npy + updated _train_eval
# - src/models/models_llm.py         → save predictions.npy + labels.npy
#
# Scripts Added:
# - scripts/aggregate_predictions.py  → aggregate across all folds
#
# Integration:
# - Automatic per-fold saving during training (no code changes needed)
# - Optional aggregation script to combine all folds
# - Ready for confusion matrix generation and analysis
#
# ============================================================================
