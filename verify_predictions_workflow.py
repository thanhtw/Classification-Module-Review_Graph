#!/usr/bin/env python3
"""
Quick verification script to test the predictions/labels workflow.

This script demonstrates the complete workflow and can be used to verify
that predictions and labels are being saved correctly.
"""

import numpy as np
from pathlib import Path


def verify_predictions_workflow():
    """Verify that predictions and labels can be loaded and used."""
    
    print("\n" + "="*80)
    print("PREDICTIONS AND LABELS WORKFLOW VERIFICATION")
    print("="*80 + "\n")
    
    artifacts_root = Path('results/modular_multimodel/model_artifacts')
    
    # Check if artifacts root exists
    if not artifacts_root.exists():
        print("❌ Artifacts root not found!")
        print(f"   Expected: {artifacts_root.absolute()}")
        print("\n💡 Run training first:")
        print("   python scripts/train.py --n_folds 10")
        return False
    
    print(f"✅ Artifacts root found: {artifacts_root}\n")
    
    # Check for predictions in bert model (as example)
    bert_fold_1 = artifacts_root / 'bert' / 'fold_1'
    
    if not bert_fold_1.exists():
        print("❌ BERT fold_1 not found!")
        print("   No training results detected yet.")
        print("\n💡 Run training first:")
        print("   python scripts/train.py --models bert --n_folds 10")
        return False
    
    # Check for predictions.npy
    pred_file = bert_fold_1 / 'predictions.npy'
    label_file = bert_fold_1 / 'labels.npy'
    
    if not pred_file.exists() or not label_file.exists():
        print("⚠️  Predictions or labels files not found in fold_1!")
        print(f"    Expected: {pred_file}")
        print(f"    Expected: {label_file}")
        print("\n💡 Check training logs for errors.")
        return False
    
    print(f"✅ Found predictions.npy in BERT fold_1")
    print(f"✅ Found labels.npy in BERT fold_1\n")
    
    # Load and verify data
    try:
        predictions = np.load(str(pred_file))
        labels = np.load(str(label_file))
        
        print(f"📊 Data Verification:")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Predictions dtype: {predictions.dtype}")
        print(f"   Predictions range: [{predictions.min()}, {predictions.max()}]")
        print(f"   Unique values: {np.unique(predictions)}")
        print(f"   Sample predictions (first 5 rows):")
        for i in range(min(5, len(predictions))):
            print(f"      {predictions[i]}")
        
        # Verify format
        if predictions.shape[1] != 3:
            print(f"\n❌ Error: Expected 3 labels, got {predictions.shape[1]}")
            return False
        
        if not np.all(np.isin(predictions, [0, 1])):
            print(f"\n❌ Error: Predictions contain values other than 0 or 1")
            return False
        
        if predictions.shape != labels.shape:
            print(f"\n❌ Error: Predictions and labels have different shapes")
            return False
        
        print(f"\n✅ Data format verified successfully!\n")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return False
    
    # Check for aggregated files
    print("📁 Checking aggregated files...\n")
    
    bert_all_preds = artifacts_root / 'bert' / 'all_folds_predictions.npy'
    bert_all_labels = artifacts_root / 'bert' / 'all_folds_labels.npy'
    
    if bert_all_preds.exists() and bert_all_labels.exists():
        all_preds = np.load(str(bert_all_preds))
        all_labels = np.load(str(bert_all_labels))
        
        print(f"✅ Found aggregated predictions: {all_preds.shape}")
        print(f"✅ Found aggregated labels: {all_labels.shape}")
        print(f"\n   Total samples: {all_preds.shape[0]}")
        print(f"   Expected (10-fold CV): ~2398")
        
        if all_preds.shape[0] != 2398:
            print(f"\n   ⚠️  Warning: Expected 2398 samples, got {all_preds.shape[0]}")
    else:
        print(f"⚠️  Aggregated files not found yet")
        print(f"   Run: python scripts/aggregate_predictions.py")
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE - System is ready!")
    print("="*80)
    print("\nNext steps:")
    print("1. Ensure all 10 models are trained:")
    print("   python scripts/train.py --n_folds 10")
    print("\n2. Aggregate predictions from all folds:")
    print("   python scripts/aggregate_predictions.py")
    print("\n3. Generate confusion matrices:")
    print("   python scripts/generate_confusion_matrices.py")
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == '__main__':
    verify_predictions_workflow()
