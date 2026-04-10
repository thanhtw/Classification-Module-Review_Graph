#!/usr/bin/env python3
"""Standalone test for SMOTE fix - no transformer dependencies"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessor import load_and_clean_data, set_seed, preprocess_for_tfidf
from src.training.config import LABEL_COLUMNS
from src.utils.smote import apply_smote_multilabel

def test_smote_balancing():
    """Test SMOTE fix to verify each label is independently balanced"""
    seed = 42
    set_seed(seed)

    # Load data
    print("📥 Loading data...")
    data = load_and_clean_data("data/cleaned_3label_data.csv")
    texts_all = data["text"].values
    labels_all = data[LABEL_COLUMNS].values

    # Create single fold for quick test
    n_train = int(0.8 * len(texts_all))
    indices = np.arange(len(texts_all))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    train_texts = [texts_all[i] for i in train_idx]
    train_labels = labels_all[train_idx]

    # Vectorize
    print("📊 Vectorizing texts...")
    x = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95).fit_transform(
        [preprocess_for_tfidf(t) for t in train_texts]
    )
    x_before = x.toarray().astype(np.float32)
    y_before = train_labels.astype(np.int64)

    # Apply SMOTE
    print("⚙️  Applying SMOTE...")
    x_after, y_after, stats = apply_smote_multilabel(x_before, y_before, seed=seed)

    # Examine results
    print("\n" + "="*80)
    print("✅ SMOTE BALANCING VERIFICATION")
    print("="*80)
    
    pretty_labels = ["Relevance", "Concreteness", "Constructiveness"]
    
    print("\n📊 BEFORE SMOTE:")
    print("-" * 80)
    for i, label in enumerate(pretty_labels):
        pos = int(y_before[:, i].sum())
        neg = int(len(y_before) - pos)
        ratio = (pos / (pos + neg)) if (pos + neg) > 0 else 0
        print(f"  {label:20} | Pos: {pos:5} | Neg: {neg:5} | Ratio: {ratio:.1%}")

    print("\n📊 AFTER SMOTE (CORRECTED):")
    print("-" * 80)
    
    all_balanced = True
    for i, label in enumerate(pretty_labels):
        pos = int(y_after[:, i].sum())
        neg = int(len(y_after) - pos)
        diff = abs(pos - neg)
        ratio = (pos / (pos + neg)) if (pos + neg) > 0 else 0
        
        # Check if reasonably balanced (within 20% difference is acceptable)
        max_val = max(pos, neg)
        is_balanced = diff <= max_val * 0.2
        all_balanced = all_balanced and is_balanced
        
        status = "✅" if is_balanced else "❌"
        print(f"  {status} {label:20} | Pos: {pos:5} | Neg: {neg:5} | Diff: {diff:5} | Ratio: {ratio:.1%}")

    print("="*80)
    
    if all_balanced:
        print("\n✅ SUCCESS: All labels are now properly balanced after SMOTE!")
        print(f"\n📈 Statistics:")
        print(f"   - Original dataset size: {len(y_before)} samples")
        print(f"   - After SMOTE size: {len(y_after)} samples")
        print(f"   - Added samples: {len(y_after) - len(y_before)}")
        print(f"   - SMOTE method: {stats['method']}")
        print(f"   - Fully balanced each label: {bool(stats['fully_balanced_each_label'])}")
    else:
        print("\n❌ FAILURE: Classes are still not balanced!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_smote_balancing()
    sys.exit(0 if success else 1)
