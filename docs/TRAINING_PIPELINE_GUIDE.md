# Training Pipeline Guide: 10-Fold Cross-Validation & Model Comparison

## Overview

The training pipeline (`run_modular_multimodel_train.py`) implements a comprehensive workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Load Data & Setup                                       │
│ - Load 2398 samples with 3 binary labels                        │
│ - Create 10 folds using stratified KFold                        │
│ - Initialize configurations for all models                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ Step 2: For Each Model (10 models × 10 folds = 100 runs)        │
│                                                                   │
│ For model in [bert, roberta, linear_svm, naive_bayes, ...]:    │
│   For fold in [1, 2, ..., 10]:                                  │
│     1. Split data:  train_idx, test_idx                         │
│     2. Preprocess:  tokenization, vectorization (TF-IDF)        │
│     3. Apply SMOTE: on training set only (for imbalance)        │
│     4. Train model: on balanced training set                    │
│     5. Predict:     on test set                                 │
│     6. Compute:     8 metrics (accuracy, precision, recall, F1) │
│     7. Save:        model artifact + metadata + predictions     │
│     8. Record:      row with all metrics                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ Step 3: Select Best Fold Per Model                               │
│                                                                   │
│ For each model:                                                 │
│   1. Sort folds by:  f1_macro (desc)                            │
│                      → f1_micro (desc)                          │
│                      → subset_accuracy (desc)                   │
│   2. Select:         top 1 fold                                 │
│   3. Extract:        best_fold, best_f1_macro, etc.             │
│   4. Result:         best_fold_per_model DataFrame (10 rows)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ Step 4: Compare Best Folds & Generate Final Report              │
│                                                                   │
│ Overall Best Model = argmax(f1_macro_mean) across all models   │
│                                                                   │
│ Outputs:                                                        │
│   1. model_results_detailed.csv (100 rows)                     │
│      - All folds, all models, all metrics                      │
│                                                                   │
│   2. model_comparison_macro_micro.csv                           │
│      - Mean ± Std for each model across 10 folds               │
│      - Sorted by f1_macro_mean (descending)                    │
│      - Shows model stability across folds                       │
│                                                                   │
│   3. best_fold_per_model.csv (10 rows)                         │
│      - Best fold for each model                                │
│      - Sorted by best_f1_macro                                 │
│      - Shows peak individual performance                       │
│                                                                   │
│   4. model_ranking_by_macro_micro_f1.csv                        │
│      - Final ranking of all models                             │
│      - Criteria: f1_macro_mean + f1_micro_mean                │
│      - WINNER: Model ranked #1                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Code Flow

### 1. Load Data & Create 10 Folds

```python
# Load dataset
df = load_and_clean_data("data/cleaned_3label_data.csv")
texts = df["text"].tolist()  # 2398 samples
labels = df[LABEL_COLUMNS].values  # (2398, 3) binary labels

# Create 10 folds
folds = _make_folds(
    n_samples=len(texts),
    n_folds=10,  # 10-fold cross-validation
    test_size=0.2,  # ignored when n_folds >= 2
    seed=42
)
# Result: 10 fold_data dicts, each with train_idx & test_idx
```

### 2. Train All Models × All Folds

```python
rows = []  # Collect all results

for model_name in ["bert", "roberta", "linear_svm", ...]:
    for fold_id in range(1, 11):  # folds 1-10
        
        # 1. Split data for this fold
        train_idx = fold_data["train_idx"]
        test_idx = fold_data["test_idx"]
        
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # 2. Train model
        metrics, train_time, infer_time = run_model(
            train_texts, y_train,
            test_texts, y_test,
            use_smote=True,  # Apply SMOTE on training set
            seed=42 + fold_id + model_id
        )
        
        # 3. Record results
        row = {
            "model": model_name,
            "fold": fold_id,
            "accuracy_micro": metrics["accuracy_micro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_micro": metrics["f1_micro"],
            "hamming_loss": metrics["hamming_loss"],
            "train_time_sec": train_time,
            "infer_time_sec": infer_time,
            "artifact_dir": model_artifact_dir,
        }
        rows.append(row)
```

### 3. Select Best Fold Per Model

```python
# Convert to DataFrame for easy grouping
df_all_results = pd.DataFrame(rows)  # 100 rows (10 models × 10 folds)

# Group by model and select best fold
best_fold_df = (
    df_all_results
    .sort_values(
        ["model", "f1_macro", "f1_micro", "subset_accuracy"],
        ascending=[True, False, False, False]  # Sort by f1_macro desc
    )
    .groupby("model", as_index=False)
    .head(1)  # Keep only the best fold per model
)
# Result: 10 rows (one best fold per model)

# Example output:
# model              | best_fold | best_f1_macro | best_f1_micro | best_subset_accuracy
# ─────────────────────────────────────────────────────────────────────────────────────
# naive_bayes        | 2         | 0.5906        | 0.6512        | 0.4162
# logistic_regression| 2         | 0.5815        | 0.6208        | 0.4103
# bert               | 5         | 0.5400        | 0.5901        | 0.3850
# ...
```

### 4. Generate Comparison Reports

```python
# Report 1: Detailed Results (all 100 runs)
df_all_results.to_csv("model_results_detailed.csv")
# Columns: model, fold, accuracy_micro, precision_macro, recall_macro, f1_macro, ...

# Report 2: Mean ± Std Per Model (10 rows)
comparison_df = (
    df_all_results
    .groupby("model", as_index=False)
    .agg({
        "f1_macro": ["mean", "std"],
        "f1_micro": ["mean", "std"],
        "precision_macro": ["mean", "std"],
        "recall_macro": ["mean", "std"],
        ...
    })
)
comparison_df.to_csv("model_comparison_macro_micro.csv")
# Example:
# model              | f1_macro_mean | f1_macro_std | precision_macro_mean | ...
# ─────────────────────────────────────────────────────────────────────────────
# naive_bayes        | 0.5860        | 0.0065       | 0.6120               | ...
# logistic_regression| 0.5220        | 0.0815       | 0.5340               | ...
# bert               | 0.5100        | 0.0420       | 0.5890               | ...

# Report 3: Best Fold Per Model (10 rows)
best_fold_df.to_csv("best_fold_per_model.csv")
# Shows best individual performance of each model

# Report 4: Final Ranking (10 rows, sorted by f1_macro_mean)
ranking_df = comparison_df[
    ["model", "f1_macro_mean", "f1_micro_mean", 
     "precision_macro_mean", "recall_macro_mean"]
].sort_values("f1_macro_mean", ascending=False)
ranking_df.to_csv("model_ranking_by_macro_micro_f1.csv")

# WINNER: ranking_df.iloc[0]["model"]
```

## Metric Selection for "Best Fold"

**Primary Metric: F1-Macro**
- Balances precision and recall per label
- Averaged across all 3 labels
- Fair treatment of imbalanced labels (constructiveness = 9.79%)

**Tiebreaker 1: F1-Micro**
- Overall precision/recall (all samples & labels equally weighted)

**Tiebreaker 2: Subset Accuracy**
- Exact match ratio (all 3 labels must match)

## Output Files Description

| File | Rows | Purpose |
|------|------|---------|
| `model_results_detailed.csv` | 100 | All folds, all metrics, debug data |
| `model_comparison_macro_micro.csv` | 10 | Average performance± std per model |
| `best_fold_per_model.csv` | 10 | Peak performance for each model |
| `model_ranking_by_macro_micro_f1.csv` | 10 | Final ranking (model #1 is BEST) |
| `training_process.json` | - | Full manifest with metadata |

## Metrics Computed Per Fold

For each fold, we compute **8 core metrics**:

1. **accuracy_micro** - Per-label accuracy averaged
2. **precision_micro** - Per-label precision averaged
3. **recall_micro** - Per-label recall averaged
4. **f1_micro** - Per-label F1 averaged
5. **accuracy_macro** - Per-label accuracy averaged  
6. **precision_macro** - Per-label precision averaged
7. **recall_macro** - Per-label recall averaged
8. **f1_macro** - Per-label F1 averaged
9. **hamming_loss** - Fraction of labels incorrectly predicted
10. **subset_accuracy** - Exact match ratio (all 3 labels correct)
11. **hamming_score** - Jaccard similarity

## Usage

```bash
# Run with default 10 folds, all models
conda run -n ThomasAgent python run_modular_multimodel_train.py

# Run specific models only
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --models linear_svm naive_bayes logistic_regression \
  --n_folds 10

# Run fewer folds for quick testing
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --models bert roberta \
  --n_folds 2

# Custom output directory
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --output_dir results/my_experiment
```

## Reproducibility

- All results saved with seed and hyperparameters
- Model artifacts (weights, vectorizers) saved per fold
- Training process manifest in JSON format
- Can reproduce exact results with same seed

## Model Comparison Interpretation

**Example Output:**
```
model              | f1_macro_mean | f1_macro_std | best_fold | best_f1_macro
───────────────────────────────────────────────────────────────────────────────
1. naive_bayes     | 0.5860        | 0.0065       | 2         | 0.5906
2. logistic_reg    | 0.5220        | 0.0815       | 2         | 0.5815
3. bert            | 0.5100        | 0.0420       | 5         | 0.5400
4. linear_svm      | 0.4393        | 0.0173       | 1         | 0.4567
```

**Interpretation:**
- **Winner**: Naive Bayes (f1_macro_mean = 0.5860)
- **Stability**: Naive Bayes (std = 0.0065) is most stable across 10 folds
- **Best Individual**: Naive Bayes fold 2 (best_f1_macro = 0.5906)
- **Variance**: Logistic Regression has high variance (std = 0.0815) - less reliable
