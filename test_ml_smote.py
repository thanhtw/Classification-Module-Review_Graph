#!/usr/bin/env python3
"""Test SMOTE fix with ML models only, save visualization to Support_document"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.train import parse_args, _make_folds
from src.data.preprocessor import load_and_clean_data, set_seed
from src.training.config import LABEL_COLUMNS
from src.analysis.analysis_utils import export_train_smote_analysis
from scripts.research_modules.visualizations import generate_smote_visualization
from src.models.models_ml import run_linear_svm, run_naive_bayes, run_logistic_regression
import numpy as np

def test_ml_models_with_smote():
    """Test SMOTE fix with ML models only"""
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
    test_idx = indices[n_train:]

    train_texts = [texts_all[i] for i in train_idx]
    train_labels = labels_all[train_idx]
    test_texts = [texts_all[i] for i in test_idx]
    test_labels = labels_all[test_idx]

    # Export SMOTE analysis
    print("⚙️  Running SMOTE analysis...")
    output_dir = "results/modular_multimodel/global_train_data_analysis"
    export_train_smote_analysis(
        train_texts=train_texts,
        train_labels=train_labels,
        output_dir=output_dir,
        seed=seed,
        use_smote=True
    )

    # Generate visualization to Support_document
    print("📊 Generating SMOTE visualization...")
    generate_smote_visualization(output_dir="Support_document")

    # Test ML models
    print("\n🤖 Training ML models with fixed SMOTE...")
    models_to_test = [
        ("linear_svm", run_linear_svm),
        ("naive_bayes", run_naive_bayes),
        ("logistic_regression", run_logistic_regression),
    ]

    for model_name, model_func in models_to_test:
        print(f"\n  Training {model_name}...")
        try:
            results = model_func(
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                seed=seed,
                use_smote=True,
                epochs=1,
            )
            print(f"  ✅ {model_name} completed")
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")

    # Verify balancing
    print("\n" + "="*60)
    print("✅ SMOTE BALANCING VERIFICATION")
    print("="*60)
    import json
    smote_file = Path(output_dir) / "train_smote_analysis_summary.json"
    with open(smote_file) as f:
        stats = json.load(f)

    labels = ["Relevance", "Concreteness", "Constructiveness"]
    for i, label in enumerate(labels):
        pos = stats["label_pos_after"][i]
        neg = stats["label_neg_after"][i]
        diff = abs(pos - neg)
        ratio = pos / (pos + neg) if (pos + neg) > 0 else 0
        status = "✅" if diff <= 10 else "❌"
        print(f"{status} {label:20} | Pos: {pos:5} | Neg: {neg:5} | Diff: {diff:5} | Ratio: {ratio:.1%}")

    print("="*60)
    print(f"\n📂 Visualization saved to: Support_document/smote_impact_visualization.png")

if __name__ == "__main__":
    test_ml_models_with_smote()
