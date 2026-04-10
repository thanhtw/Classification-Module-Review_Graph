#!/usr/bin/env python3
"""Quick test to verify SMOTE balancing fix without full training"""

import sys
from pathlib import Path
import numpy as np
from src.data.preprocessor import load_and_clean_data, set_seed
from src.analysis.analysis_utils import export_train_smote_analysis
from scripts.research_modules.visualizations import generate_smote_visualization

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_smote_fix():
    seed = 42
    set_seed(seed)

    # Load data
    print("📥 Loading data...")
    from src.training.config import LABEL_COLUMNS
    data = load_and_clean_data("data/cleaned_3label_data.csv")
    texts_all = data["text"].values
    labels_all = data[LABEL_COLUMNS].values

    # Use 80% for training (simulating fold)
    n_train = int(0.8 * len(texts_all))
    train_texts = texts_all[:n_train]
    train_labels = labels_all[:n_train]

    # Run SMOTE analysis
    print("⚙️  Running SMOTE analysis with fixed balancing...")
    output_dir = "results/modular_multimodel/global_train_data_analysis"
    export_train_smote_analysis(
        train_texts=train_texts,
        train_labels=train_labels,
        output_dir=output_dir,
        seed=seed,
        use_smote=True
    )

    # Generate visualization
    print("📊 Generating SMOTE visualization...")
    generate_smote_visualization(output_dir="Support_document")

    # Check if balanced
    import json
    smote_file = Path(output_dir) / "train_smote_analysis_summary.json"
    with open(smote_file) as f:
        stats = json.load(f)

    print("\n✅ SMOTE Analysis Results:")
    print("=" * 60)
    labels = ["Relevance", "Concreteness", "Constructiveness"]
    for i, label in enumerate(labels):
        pos = stats["label_pos_after"][i]
        neg = stats["label_neg_after"][i]
        diff = abs(pos - neg)
        ratio = pos / (pos + neg) if (pos + neg) > 0 else 0
        print(f"{label:20} | Pos: {pos:5} | Neg: {neg:5} | Diff: {diff:5} | Ratio: {ratio:.1%}")

    print("=" * 60)
    if stats.get("fully_balanced_each_label"):
        print("✅ SUCCESS: All labels are perfectly balanced!")
    else:
        diff_after = stats.get("label_abs_diff_after", [])
        if max(diff_after) <= 10:  # Allow small tolerance
            print("✅ SUCCESS: All labels are nearly balanced (diff <= 10)")
        else:
            print("❌ FAIL: Some labels still imbalanced")
            print(f"   Max imbalance: {max(diff_after)}")

if __name__ == "__main__":
    test_smote_fix()
