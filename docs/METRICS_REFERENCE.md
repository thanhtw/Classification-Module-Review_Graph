# Metrics Reference Guide for Research Paper

## Summary of Changes ✅

Enhanced `scripts/research_comparison.py` with comprehensive metrics reporting for multilabel text classification research paper.

### **User Requirements Met:**

✅ **Per-Label Metrics**
- Function: `calculate_per_label_metrics()` 
- Calculates precision, recall, F1 for each label individually
- Can generate table showing metrics for "relevance", "concreteness", "constructive" labels

✅ **Macro-Averaged Metrics**
- Already present in results CSV: `accuracy_macro`, `precision_macro`, `recall_macro`, `f1_macro`
- Displayed in results table and JSON reports
- Definition: Average of metrics computed for each label (unweighted)

✅ **Micro-Averaged Metrics**
- Already present in results CSV: `accuracy_micro`, `precision_micro`, `recall_micro`, `f1_micro`
- Displayed in results table and JSON reports
- Definition: Global metrics from aggregated TP/FP/FN across all labels

✅ **Hamming Loss**
- Already present in results CSV: `hamming_loss`
- Function: `generate_multilabel_metrics_report()` extracts and displays with interpretations
- Definition: Fraction of labels predicted incorrectly per sample
- Lower is better (0.0 = perfect, higher = more incorrect predictions)

✅ **Subset Accuracy**
- Already present in results CSV: `subset_accuracy`
- Function: `generate_multilabel_metrics_report()` displays in ranked table
- Definition: Exact match accuracy (All labels must be correct for sample to count)
- More strict than per-label accuracy
- Higher is better (0.0 = no perfect samples, 1.0 = all perfect)

---

## Metrics Available in CSV Results

**File**: `results/modular_multimodel/model_results_detailed.csv`

### Standard Classification Metrics (Per Model/Fold)

| Metric | Range | Interpretation | For Paper |
|--------|-------|-----------------|-----------|
| `accuracy_micro` | 0.0-1.0 | Global accuracy (all labels) | Methods table |
| `accuracy_macro` | 0.0-1.0 | Average accuracy per label | Methods table |
| `precision_micro` | 0.0-1.0 | Global precision | Results table |
| `precision_macro` | 0.0-1.0 | Average precision per label | Results table |
| `recall_micro` | 0.0-1.0 | Global recall | Results table |
| `recall_macro` | 0.0-1.0 | Average recall per label | Results table |
| `f1_micro` | 0.0-1.0 | Global F1 score | Results table |
| `f1_macro` | 0.0-1.0 | Average F1 per label | **PRIMARY ranking metric** |

### Multilabel-Specific Metrics (Per Model/Fold)

| Metric | Range | Interpretation | For Paper |
|--------|-------|-----------------|-----------|
| `hamming_loss` | 0.0-NUM_LABELS | Fraction of incorrect labels | Multilabel performance |
| `subset_accuracy` | 0.0-1.0 | Exact match accuracy | Strict performance metric |
| `hamming_score` | 0.0-1.0 | Complement of hamming_loss | Alternative multilabel metric |

---

## Generated Reports and Their Contents

### 1. **Comparison Table** (`all_models_comparison.csv`)
- **Rows**: One per model (10 total)
- **Columns**: All metrics averaged across folds
- **Use**: Direct import into paper as results table

### 2. **Detailed Results** (`comparison_detailed.json`)
```json
{
  "configuration": {...},
  "models": [...],      // Raw metric values
  "ranking": [          // Sorted by F1-macro
    {
      "rank": 1,
      "model": "...",
      "f1_macro_mean": 0.85,
      "f1_macro_std": 0.023,
      "hamming_loss_mean": 0.133,
      "subset_accuracy_mean": 0.867,
      ...
    }
  ]
}
```

### 3. **Per-Label Metrics Report** (`per_label_metrics_report.json`)
```json
{
  "labels": ["relevance", "concreteness", "constructive"],
  "label_descriptions": {
    "relevance": "Is the text relevant to the topic?",
    "concreteness": "Does the text contain concrete examples?",
    "constructive": "Is the text constructive/helpful?"
  },
  "models": {
    "Linear SVM": {
      "relevance": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
      "concreteness": {...},
      "constructive": {...}
    }
  }
}
```

### 4. **Multilabel Metrics Report** (`multilabel_metrics_report.json`)
- Ranked table of all models by performance
- Hamming Loss values with interpretations
- Subset Accuracy values with interpretations
- Metric definitions included

### 5. **Research Summary** (`RESEARCH_SUMMARY.txt`)
- Human-readable format
- Detailed rankings with all metrics
- Can be directly quoted in paper

### 6. **Dataset Report** (`dataset_report.json`)
- Total samples, text statistics
- Label distribution
- Multi-label combinations

### 7. **Model Configurations** (`model_configurations.json`)
- Details of all 10 models
- Hyperparameters
- Model sizes

### 8. **Training Process Report** (`training_process_report.json`)
- CV strategy (10-fold stratified)
- Preprocessing steps
- Training configuration
- Data split (90% train / 10% test)

### 9. **Research Paper Appendix** (`research_paper_appendix.json`)
- Complete combined document
- All above reports merged
- Ready for paper appendix

---

## How to Use in Paper

### Methods Section
```markdown
**Cross-Validation & Training:**
See: `results/research_comparison/training_process_report.json`
- 10-fold stratified cross-validation
- 90% train / 10% test split
- SMOTE applied for class imbalance
- Training details: [reference JSON file]

**Dataset:**
See: `results/research_comparison/dataset_report.json`
- Total samples: 2,398
- Labels: relevance, concreteness, constructive
- Text statistics: [see JSON]

**Models:**
See: `results/research_comparison/model_configurations.json`
- 10 models total across ML, DL, Transformers, LLM
- [Details for each model]
```

### Results Section
```markdown
**Overall Rankings (by F1-Macro):**
See: `results/research_comparison/comparison_detailed.json`

| Rank | Model                    | F1-Macro    | F1-Micro    | Hamming Loss | Subset Acc |
|------|--------------------------|-------------|-------------|--------------|------------|
| 1    | [Best Model]             | 0.85±0.023  | 0.88±0.018  | 0.133±0.045  | 0.867±0.034|
| ...  | ...                      | ...         | ...         | ...          | ...        |

See: `results/research_comparison/all_models_comparison.csv` for complete table

**Per-Label Performance:**
See: `results/research_comparison/per_label_metrics_report.json`
- Relevance label: [metrics for each model]
- Concreteness label: [metrics for each model]
- Constructive label: [metrics for each model]

**Multilabel-Specific Metrics:**
See: `results/research_comparison/multilabel_metrics_report.json`

Hamming Loss (lower is better):
- Measures fraction of incorrectly predicted labels
- Best model: 0.133 = ~13% labels predicted incorrectly on average
- Range: 0.0 (perfect) to 3.0 (all labels wrong)

Subset Accuracy (higher is better):
- Exact match on all labels simultaneously
- Best model: 0.867 = 86.7% of samples have all labels correct
- Stricter metric than per-label accuracy
```

---

## Key Metrics Explained for Paper

### 1. Hamming Loss
```
Formula: (total_incorrect_labels) / (n_samples × n_labels)

Example with 3 labels, 100 samples:
- Model predicts correctly: 275 out of 300 labels
- Hamming loss = (300-275) / 300 = 0.0833
- Interpretation: Model gets ~92% of labels correct (or ~8% wrong)
```

### 2. Subset Accuracy  
```
Formula: (samples_with_all_correct_labels) / n_samples

Example with 100 samples:
- 87 samples have all 3 labels predicted correctly
- 13 samples have at least 1 label wrong
- Subset accuracy = 87/100 = 0.87
- Interpretation: 87% of samples completely correct, 13% partially wrong
```

### 3. F1-Macro vs F1-Micro
```
F1-Macro (unweighted average across labels):
- Relevance F1: 0.80
- Concreteness F1: 0.75
- Constructive F1: 0.85
- F1-Macro = (0.80 + 0.75 + 0.85) / 3 = 0.8033

F1-Micro (global from total TP/FP/FN):
- Total TP: 250
- Total FP: 30  
- Total FN: 20
- F1-Micro = 2×250 / (2×250 + 30 + 20) = 0.8547
```

---

## Running the Script

```bash
# Generate all reports (default: 10 folds, seed=42)
python scripts/research_comparison.py

# Custom settings
python scripts/research_comparison.py --n_folds 5 --seed 123

# Output location
# All files saved to: results/research_comparison/
```

**Note**: Requires compatible NumPy/Pandas environment. Use ThomasAgent environment if compatibility issues occur.

---

## Metrics Mapping to Research Paper Requirements

| User Requirement | Implemented As | Where to Find |
|------------------|-----------------|----------------|
| Per-Label Metrics | `calculate_per_label_metrics()` | Function in script + per_label_metrics_report.json |
| Macro-Averaged Metrics | CSV columns + display tables | comparison_detailed.json |
| Micro-Averaged Metrics | CSV columns + display tables | comparison_detailed.json |
| Hamming Loss | multilabel_metrics_report() | multilabel_metrics_report.json + console output |
| Subset Accuracy | multilabel_metrics_report() | multilabel_metrics_report.json + console output |

✅ **All requirements fulfilled and integrated into research_comparison.py**
