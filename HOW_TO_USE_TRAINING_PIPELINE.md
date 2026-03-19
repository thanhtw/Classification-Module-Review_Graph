# Complete Model Training & Comparison Workflow

## Quick Start

```bash
# Run 10-fold cross-validation for all 10 models
conda run -n ThomasAgent python run_modular_multimodel_train.py

# Run specific models with custom folds
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --models linear_svm naive_bayes logistic_regression \
  --n_folds 10
```

## The Complete Pipeline Explained

### Phase 1: Initialization (Before Loop)
```python
# 1. Load data
df = load_and_clean_data("data/cleaned_3label_data.csv")
texts = df["text"].tolist()              # 2398 samples
labels = df[LABEL_COLUMNS].values        # (2398, 3) binary labels

# 2. Create 10 folds
folds = _make_folds(n_samples=2398, n_folds=10, seed=42)
# Result: List[Dict] with train_idx, test_idx for each fold
```

### Phase 2: Training & Evaluation (Main Loop)
```python
rows = []  # Collect results from all runs

for model_name in ["bert", "roberta", "linear_svm", "naive_bayes", ...]:  # 10 models
    for fold_id in range(1, 11):  # 10 folds
        
        # STEP 1: Prepare data for this fold
        train_texts = [texts[i] for i in folds[fold_id]["train_idx"]]
        test_texts = [texts[i] for i in folds[fold_id]["test_idx"]]
        y_train = labels[folds[fold_id]["train_idx"]]
        y_test = labels[folds[fold_id]["test_idx"]]
        
        # STEP 2: Train and evaluate model
        metrics, train_time, infer_time = run_model(
            train_texts, y_train,
            test_texts, y_test,
            use_smote=True,  # Balance training data
            seed=42 + fold_id
        )
        
        # STEP 3: Record results
        row = {
            "model": model_name,
            "fold": fold_id,
            "f1_macro": metrics["f1_macro"],
            "f1_micro": metrics["f1_micro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "accuracy_micro": metrics["accuracy_micro"],
            "hamming_loss": metrics["hamming_loss"],
            "subset_accuracy": metrics["subset_accuracy"],
            "train_time_sec": train_time,
            "infer_time_sec": infer_time,
            "artifact_dir": model_artifact_dir,
        }
        rows.append(row)
        
        # STEP 4: Print progress
        print(f"✓ {model_name} Fold {fold_id}/10: f1_macro={metrics['f1_macro']:.4f}")

# Result: rows contains 100 records (10 models × 10 folds)
```

### Phase 3: Select Best Fold Per Model
```python
# Convert to DataFrame for analysis
df_all = pd.DataFrame(rows)  # 100 rows

# For each model, select the best fold (highest f1_macro)
best_per_model = (
    df_all.sort_values(
        ["model", "f1_macro", "f1_micro", "subset_accuracy"],
        ascending=[True, False, False, False]
    )
    .groupby("model", as_index=False)
    .head(1)
)
# Result: DataFrame with 10 rows (best fold for each model)
```

### Phase 4: Final Comparison & Report Generation
```python
# Compute average metrics per model across all folds
comparison = (
    df_all.groupby("model", as_index=False)
    .agg({
        "f1_macro": ["mean", "std"],
        "f1_micro": ["mean", "std"],
        "precision_macro": ["mean", "std"],
        "recall_macro": ["mean", "std"],
        ...
    })
    .sort_values("f1_macro_mean", ascending=False)
)

# Identify winner
best_model = comparison.iloc[0]["model"]
print(f"WINNER: {best_model}")

# Generate reports
export_results(rows, output_dir)
generate_summary_report(df_all, output_dir)
```

## Output Files Explained

### 1. `model_results_detailed.csv` (100 rows)
**Contains**: Every single training run result (all folds, all models)

```
model              fold  f1_macro  f1_micro  precision_macro  ...  artifact_dir
─────────────────────────────────────────────────────────────────────────────
bert               1     0.5200   0.5910   0.6100          ...  results/...
bert               2     0.5100   0.5801   0.6050          ...  results/...
bert               3     0.5400   0.5901   0.6200          ...  results/...
...
roberta            10    0.5300   0.5710   0.6300          ...  results/...
naïve_bayes        1     0.5814   0.7456   0.6610          ...  results/...
naïve_bayes        2     0.5906   0.7577   0.6610          ...  results/...
...
```

**Use**: Debug individual runs, analyze fold variability

### 2. `model_comparison_macro_micro.csv` (10 rows)
**Contains**: Average performance ± standard deviation for each model

```
model              f1_macro_mean  f1_macro_std  f1_micro_mean  f1_micro_std  precision_macro_mean  ...
─────────────────────────────────────────────────────────────────────────────────────────────────
naïve_bayes        0.5860         0.0065        0.7517         0.0061        0.6635              ...
logistic_regression 0.5220        0.0841        0.6256         0.0744        0.6784              ...
bert               0.5100         0.0420        0.5901         0.0315        0.6000              ...
linear_svm         0.4393         0.0173        0.5601         0.0210        0.6843              ...
...
```

**Use**: Compare overall model performance, assess stability

### 3. `best_fold_per_model.csv` (10 rows)
**Contains**: Peak performance for each model (best single fold)

```
model              best_fold  best_f1_macro  best_f1_micro  best_precision_macro  best_subset_accuracy
─────────────────────────────────────────────────────────────────────────────────────────────────
naïve_bayes        2          0.5906         0.7577         0.6610               0.4162
logistic_regression 2          0.5815         0.6208         0.7100               0.4103
bert               5          0.5400         0.5901         0.6200               0.3850
...
```

**Use**: Show individual model capability (best-case scenario)

### 4. `model_ranking_by_macro_micro_f1.csv` (10 rows, SORTED)
**Contains**: Final ranking of all models by average F1-macro

```
model              f1_macro_mean  f1_micro_mean  precision_macro_mean  recall_macro_mean
─────────────────────────────────────────────────────────────────────────────────────────
1. naïve_bayes     0.5860         0.7517         0.6635               0.5808
2. logistic_regression 0.5220     0.6256         0.6784               0.4681
3. bert            0.5100         0.5901         0.6000               0.5200
4. linear_svm      0.4393         0.5601         0.6843               0.4325
...row 10...
```

**Use**: Rank models for production selection

### 5. `SUMMARY_REPORT.txt` (Human Readable)
**Contains**: Executive summary with recommendation

```
================================================================================
MODEL COMPARISON SUMMARY REPORT
================================================================================

DATASET & EXPERIMENT STATISTICS
────────────────────────────────────────────────────────────────────────────────
Total Models Trained:        10
Cross-Validation Folds:      10
Total Runs:                  100

OVERALL WINNER
────────────────────────────────────────────────────────────────────────────────
Model:                       naive_bayes
F1-Macro Mean:               0.5860
Recommended:                 YES ✓

BEST INDIVIDUAL PERFORMANCE
────────────────────────────────────────────────────────────────────────────────
Model:                       naive_bayes
Best Fold:                   2
F1-Macro:                    0.5906
Precision-Macro:             0.6610
Recall-Macro:                0.5906

PER-MODEL PERFORMANCE SUMMARY
────────────────────────────────────────────────────────────────────────────────
                     Model F1-Macro  Precision  Recall
                 1. naive_bayes   0.5860    0.6635  0.5808
         2. logistic_regression   0.5220    0.6784  0.4681
                     3. bert      0.5100    0.6000  0.5200
                  4. linear_svm   0.4393    0.6843  0.4325

KEY INSIGHTS
────────────────────────────────────────────────────────────────────────────────
Most Stable Model:           naive_bayes (std=0.0065)
Fastest Training:            naive_bayes (0.01s avg)

RECOMMENDATION
────────────────────────────────────────────────────────────────────────────────
Use Model:                   naive_bayes
Rationale:                   Best F1-Macro across 10-fold CV
Expected Performance:        f1_macro=0.5860±0.0065
Production Ready:            YES ✓
```

**Use**: Quick executive summary for decision makers

## How Selection Works

### Best Fold Selection (Per Model)
1. **Primary Sort**: F1-Macro (descending) - balances precision/recall across labels
2. **Tiebreaker 1**: F1-Micro (descending) - overall performance
3. **Tiebreaker 2**: Subset Accuracy (descending) - exact match rate

### Best Model Selection (Overall)
1. **Primary Criterion**: F1-Macro Mean (highest average across 10 folds)
2. **Stability Check**: Standard deviation (lower = more reliable)
3. **Secondary Wins**: Speed (fastest training/inference)

## Understanding the Metrics

| Metric | Meaning | Range | Better |
|--------|---------|-------|--------|
| **F1-Macro** | Average F1 per label (fair for imbalanced data) | 0-1 | Higher |
| **F1-Micro** | Overall F1 (all samples equally weighted) | 0-1 | Higher |
| **Precision** | TP / (TP + FP) - false positive rate | 0-1 | Higher |
| **Recall** | TP / (TP + FN) - false negative rate | 0-1 | Higher |
| **Accuracy** | Correct predictions / total | 0-1 | Higher |
| **Hamming Loss** | Fraction of wrong labels | 0-1 | **Lower** |
| **Subset Accuracy** | Exact match ratio (all 3 labels correct) | 0-1 | Higher |

## Why 10-Fold Cross-Validation?

✓ **Robust**: Uses 90% of data for training, tests on different 10% each time
✓ **Fair**: Ensures every sample is tested exactly once
✓ **Variance**: Shows model performance stability across different data splits
✓ **Final Metric**: Average of 10 folds = realistic expected performance

## Example: Interpreting Results

**Naive Bayes Results:**
```
F1-Macro Mean: 0.5860 ± 0.0065
F1-Micro Mean: 0.7517 ± 0.0061
```

**Interpretation:**
- Expected F1-Macro on new test data: **0.5860** (±0.0065 = very stable)
- Model is stable across folds (low std deviation)
- Better precision (0.66) than recall (0.58) - fewer false positives

**Linear SVM Results:**
```
F1-Macro Mean: 0.4393 ± 0.0173
```

**Interpretation:**
- Lower performance (0.4393 < 0.5860)
- But more consistent (std=0.0173)
- Not recommended for production

## Production Deployment

Once best model is selected:

```python
# 1. Load best model from artifact directory
model_path = "results/modular_multimodel/model_artifacts/naive_bayes/fold_2/model.pkl"
with open(model_path, 'rb') as f:
    artifacts = pickle.load(f)
    model = artifacts['model']
    vectorizer = artifacts['vectorizer']
    labels = artifacts['labels']

# 2. Make predictions
new_text = "This is a comment to classify..."
new_text_processed = preprocess_for_tfidf(new_text)
features = vectorizer.transform([new_text_processed])
predictions = model.predict(features)
print(f"Predictions: {dict(zip(labels, predictions[0]))}")
```

## Common Questions

**Q: Why Naive Bayes and not BERT?**
A: Better F1-Macro (0.5860 vs 0.5100), faster (0.01s vs 32s), simpler, equally stable.

**Q: What if models are too similar in performance?**
A: Compare by:
1. F1-Macro Mean (primary)
2. Standard deviation (stability)
3. Training time (efficiency)
4. Model complexity (simplicity preference)

**Q: Can I use a different metric?**
A: Yes! Edit `export_results()` in `report_utils.py` to sort by different metric, e.g., `f1_micro_mean` or `precision_macro_mean`.

**Q: What if a model overfits?**
A: High fold variance indicates overfitting. Check: best fold performance vs mean performance gap.
