# Predictions and Labels Storage Implementation Summary

## ✅ Changes Completed

### 1. **Modified Model Training Scripts** (Automatic Saving)

All model training functions now automatically save predictions and true labels during training:

**Modified files:**
- ✅ `src/models/models_ml.py` - 3 models:
  - `run_linear_svm()` 
  - `run_naive_bayes()`
  - `run_logistic_regression()`

- ✅ `src/models/models_transformers.py` - 2 models:
  - `run_transformer()` (BERT, RoBERTa)

- ✅ `src/models/models_nn.py` - 3 models:
  - `run_lstm_like()` (LSTM, BiLSTM)
  - `run_cnn_attention()`
  - Modified `_train_eval()` to return predictions and labels

- ✅ `src/models/models_llm.py` - 2 models:
  - `run_llm_zero_few_shot()` (LLM Few-Shot, LLM Zero-Shot)

**What's saved after each fold:**
```
results/modular_multimodel/model_artifacts/{model}/fold_{i}/
├── predictions.npy      ← Model predictions (shape: n_samples × 3)
├── labels.npy          ← True labels (shape: n_samples × 3)
├── model files         ← (existing)
└── metadata.json       ← (existing)
```

### 2. **Created Aggregation Script** (Combine All Folds)

✅ `scripts/aggregate_predictions.py`

Combines predictions and labels from all 10 folds into single files:

```bash
python scripts/aggregate_predictions.py
```

**Output:**
```
results/modular_multimodel/model_artifacts/{model}/
├── all_folds_predictions.npy  ← Combined from all folds
├── all_folds_labels.npy       ← Combined from all folds
└── fold_{i}/...               ← (existing per-fold files)
```

### 3. **Documentation** (Usage Guide)

✅ `PREDICTIONS_AND_LABELS_GUIDE.md`

Comprehensive guide including:
- File structure overview
- Data format explanation
- Usage workflow
- Code examples for loading data
- Troubleshooting tips

---

## 📋 Quick Start

### Step 1: Train All Models
```bash
python scripts/train.py \
  --models bert roberta lstm bilstm cnn_attention \
           linear_svm logistic_regression naive_bayes \
           llm_few_shot llm_zero_shot \
  --n_folds 10
```

**Result:** `predictions.npy` and `labels.npy` saved in each fold directory

### Step 2: Aggregate Predictions (Optional but Recommended)
```bash
python scripts/aggregate_predictions.py
```

**Result:** Combined predictions/labels across all folds for model-wide analysis

### Step 3: Generate Confusion Matrices
```bash
python scripts/generate_confusion_matrices.py
```

**Uses:** Aggregated predictions and labels to draw confusion matrices

---

## 🗂️ Data Storage Structure

```
results/modular_multimodel/model_artifacts/
├── bert/
│   ├── fold_1/
│   │   ├── predictions.npy (2398, 3)  ← Per-fold predictions
│   │   ├── labels.npy (2398, 3)       ← Per-fold labels
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── metadata.json
│   ├── fold_2/ ... fold_10/
│   ├── all_folds_predictions.npy (2398, 3)  ← AGGREGATED
│   └── all_folds_labels.npy (2398, 3)       ← AGGREGATED
├── lstm/ ... bilstm/ ... (similar structure)
├── linear_svm/ ... logistic_regression/ ... naive_bayes/ (20-30 KB .npy files)
├── cnn_attention/ ...
└── llm_few_shot/ ... llm_zero_shot/ ...
```

---

## 💾 Data Format Details

### Shape
- **predictions.npy**: `(n_samples, 3)`
- **labels.npy**: `(n_samples, 3)`

### For 10-fold CV with 2398 test samples total:
- Each fold: `(~240, 3)` 
- All folds aggregated: `(2398, 3)`

### Columns represent:
```
Column 0: Relevance     (0 or 1)
Column 1: Concreteness  (0 or 1)
Column 2: Constructive  (0 or 1)
```

### Example matrix (first 5 rows):
```python
[[0, 1, 0],   # Sample 1: not relevant, concrete, not constructive
 [1, 1, 1],   # Sample 2: relevant, concrete, constructive
 [1, 0, 1],   # Sample 3: relevant, not concrete, constructive
 [0, 0, 0],   # Sample 4: none apply
 [1, 1, 0]]   # Sample 5: relevant, concrete, not constructive
```

---

## 🔍 Verification Checklist

After running training, verify predictions were saved:

```bash
# Check if files exist
ls -lh results/modular_multimodel/model_artifacts/*/fold_1/predictions.npy
ls -lh results/modular_multimodel/model_artifacts/*/fold_1/labels.npy

# Count samples in each fold
python -c "import numpy as np; p = np.load('results/modular_multimodel/model_artifacts/bert/fold_1/predictions.npy'); print(f'Shape: {p.shape}, Values: {np.unique(p)}')"
```

After running aggregation:

```bash
# Check aggregated files
ls -lh results/modular_multimodel/model_artifacts/*/all_folds_predictions.npy
ls -lh results/modular_multimodel/model_artifacts/*/all_folds_labels.npy

# Verify all 2398 samples
python -c "import numpy as np; p = np.load('results/modular_multimodel/model_artifacts/bert/all_folds_predictions.npy'); print(f'Shape: {p.shape}')"
```

---

## 📚 Code Examples

### Load aggregated predictions for one model
```python
import numpy as np
from pathlib import Path

model_name = 'bert'
artifacts_root = Path('results/modular_multimodel/model_artifacts')

all_preds = np.load(artifacts_root / model_name / 'all_folds_predictions.npy')
all_labels = np.load(artifacts_root / model_name / 'all_folds_labels.npy')

print(f"Predictions shape: {all_preds.shape}")  # (2398, 3)
print(f"Labels shape: {all_labels.shape}")      # (2398, 3)
print(f"Unique values: {np.unique(all_preds)}")  # [0, 1]
```

### Generate confusion matrix for one label
```python
from sklearn.metrics import confusion_matrix
import numpy as np

all_preds = np.load('results/modular_multimodel/model_artifacts/bert/all_folds_predictions.npy')
all_labels = np.load('results/modular_multimodel/model_artifacts/bert/all_folds_labels.npy')

# Confusion matrix for 'relevance' label (column 0)
cm = confusion_matrix(all_labels[:, 0], all_preds[:, 0])
print(cm)
# Output:
# [[TN  FP]
#  [FN  TP]]
```

---

## ❓ Troubleshooting

**Q: Where are predictions saved?**  
A: Automatically in `results/modular_multimodel/model_artifacts/{model_name}/fold_{i}/`

**Q: Do I need to modify training code?**  
A: No! Saving is automatic if you run `train.py` with `save_dir` parameter (default)

**Q: Can I manually load fold-specific predictions?**  
A: Yes:
```python
import numpy as np
fold_1_preds = np.load('results/modular_multimodel/model_artifacts/bert/fold_1/predictions.npy')
```

**Q: What if aggregation script fails for one model?**  
A: That model's training might have failed. Check logs in training output or re-run training for that model.

---

## 🎯 Summary

| Task | Status | Location |
|------|--------|----------|
| Save predictions per fold | ✅ Automatic | `fold_{i}/predictions.npy` |
| Save labels per fold | ✅ Automatic | `fold_{i}/labels.npy` |
| Aggregate across folds | ✅ Script | `aggregate_predictions.py` |
| Generate confusion matrices | ✅ Ready | `generate_confusion_matrices.py` |
| Documentation | ✅ Complete | `PREDICTIONS_AND_LABELS_GUIDE.md` |

All systems ready! Run your training and aggregation workflow as described in "Quick Start" above.
