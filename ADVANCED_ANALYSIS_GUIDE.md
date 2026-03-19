# Advanced Analysis Guide

This guide explains the three new analysis features to address reproducibility, error analysis, and per-label metrics.

## Overview

The project now includes automatic generation of:
1. **Per-label metrics** - F1 scores for each label (relevance, constructiveness, concreteness) separately
2. **Error analysis** - Identification and categorization of misclassifications
3. **Reproducibility tracking** - Complete environment, hyperparameter, and hardware information

## Usage

### Step 1: Run Training (with all models)

```bash
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --models bert roberta svm decision_tree cnn_attention lstm bilstm llm_zero_shot llm_few_shot
```

### Step 2: Generate Comprehensive Analysis

After training completes:

```bash
conda run -n ThomasAgent python function/comprehensive_analysis.py \
  --results_dir results/modular_multimodel
```

## Output Files

### Comprehensive Analysis Directory
```
results/modular_multimodel/comprehensive_analysis/
├── ANALYSIS_SUMMARY.txt              # Executive summary
├── all_folds_summary.csv             # All folds with metrics
├── best_folds_comparison.csv         # Best fold per model
└── <model_name>/
    └── model_analysis.json           # Per-model statistics
```

## Features Explained

### 1. Per-Label Metrics

**File**: `modular_training/per_label_metrics.py`

**Purpose**: 
- Breaks down F1, precision, recall for each label separately
- Shows class imbalance for each label
- Generates confusion matrices per label

**Key Metrics**:
```
{
  "relevance": {
    "f1": 0.8500,
    "precision": 0.8400,
    "recall": 0.8600,
    "positive_rate": 45.23,     # % of positive samples
    "support": 1024            # Number of positive samples
  },
  "constructiveness": {
    "f1": 0.5200,
    "precision": 0.6100,
    "recall": 0.4500,
    "positive_rate": 9.79,     # Severely imbalanced!
    "support": 222
  },
  ...
}
```

**Example Usage**:
```python
from modular_training.per_label_metrics import compute_per_label_metrics

per_label_metrics = compute_per_label_metrics(
    y_true=test_labels,
    y_pred=predictions,
    label_names=["relevance", "constructiveness", "concreteness"]
)

# Result: Dictionary with per-label metrics
# Reveals which labels are more difficult for your model
```

**Why It Matters**:
- **Class imbalance**: Constructiveness is only 9.79% positive - your model may ignore it
- **Per-label performance**: Different labels need different training strategies
- **Confusion matrices**: Visual representation of False Positives vs False Negatives per label

### 2. Error Analysis

**File**: `modular_training/error_analysis.py`

**Purpose**:
- Identifies specific types of misclassifications
- Categories errors by false positives vs false negatives per label
- Samples actual misclassified comments for qualitative analysis

**Output Structure**:
```json
{
  "model": "bert",
  "fold": 1,
  "total_samples": 2048,
  "total_misclassified": 256,
  "error_rate": 0.125,
  "errors_by_type": {
    "False_Negative_constructiveness": 64,
    "False_Positive_relevance": 45,
    "False_Negative_relevance": 32
  },
  "samples": [
    {
      "text": "How to improve Python performance? I disagree with your approach...",
      "true_labels": {"relevance": true, "constructiveness": true, "concreteness": false},
      "pred_labels": {"relevance": true, "constructiveness": false, "concreteness": false},
      "error_types": ["False_Negative_constructiveness"]
    },
    ...
  ]
}
```

**Example Analysis**:
If BERT has high False Negatives for "constructiveness", it means:
- BERT is too conservative - it's missing constructive comments
- Might need to adjust decision threshold
- Could indicate the model struggles with sarcasm or subtle positivity

### 3. Reproducibility Tracking

**File**: `modular_training/reproducibility.py`

**Purpose**:
- Records exact model checkpoints used (e.g., "bert-base-chinese")
- Captures complete environment (Python, packages, GPU)
- Logs hyperparameters and training time
- Enables exact reproduction by others

**Manifest Content**:
```json
{
  "reproducibility_info": {
    "timestamp": "2024-03-19 14:30:45",
    "hostname": "selab-232"
  },
  "model": {
    "name": "bert",
    "checkpoint": "bert-base-chinese",
    "source": "HuggingFace",
    "seed": 42,
    "fold": 1
  },
  "environment": {
    "python_version": "3.11.0",
    "platform": "Linux-5.15.0",
    "gpu_name": "NVIDIA GeForce RTX 4080 SUPER",
    "gpu_count": 1,
    "packages": {
      "torch": "2.1.0",
      "transformers": "4.34.0",
      "sklearn": "1.3.0"
    }
  },
  "training": {
    "seed": 42,
    "training_time_sec": 245.67,
    "inference_time_sec": 12.34
  },
  "hyperparameters": {
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.00002,
    "max_len": 128
  }
}
```

**Why It Matters**:
- **Reproducibility**: Other researchers can exactly replicate your results
- **Transparency**: All hardware and software info is documented
- **Version tracking**: Specific HuggingFace model versions (bert-base-chinese is pinned)
- **Hardware**: Shows how results might vary with different GPUs

## Integration with Training Pipeline

The modules are automatically called during training:

```python
# Per-label metrics are computed for each fold
per_label_metrics = compute_per_label_metrics(y_test, y_pred, label_names)

# Error analysis runs automatically
error_analysis = generate_error_summary(test_texts, y_test, y_pred, ...)

# Reproducibility manifest is saved
manifest = create_reproducibility_manifest(model_name, fold, seed, ...)
save_reproducibility_manifest(manifest, metadata_path)
```

## Interpreting Results

### For Class Imbalance (constructiveness at 9.79%)

Look at:
1. **Per-label F1 for constructiveness** - Usually much lower than other labels
2. **False Negative rate** - Is the model missing constructive comments?
3. **Error analysis** - Check if there's bias towards predicting "not constructive"

### For Model Comparison

Compare:
1. **Per-label performance** - Does BERT excel at one label but struggle with another?
2. **Error types** - Does LLM have different error patterns than BERT?
3. **Inference time** - Trade-off between accuracy and speed

### For Production Deployment

Choose based on:
1. **F1-Macro vs Subset Accuracy** - Depending on your use case
2. **Inference time** - Real-time vs batch processing needs
3. **Error analysis** - Are failures acceptable in your domain?

## Example Workflow

```bash
# 1. Train all models
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --models bert roberta svm lstm bilstm

# 2. Generate comprehensive analysis
conda run -n ThomasAgent python function/comprehensive_analysis.py

# 3. Check results
cat results/modular_multimodel/comprehensive_analysis/ANALYSIS_SUMMARY.txt
less results/modular_multimodel/comprehensive_analysis/best_folds_comparison.csv

# 4. Deep dive into best model
cat results/modular_multimodel/comprehensive_analysis/bert/model_analysis.json

# 5. Understand failure modes
cat results/modular_multimodel/model_artifacts/bert/fold_1/error_analysis_bert_fold1.json
```

## Customization

### Adding Custom Metrics

In `per_label_metrics.py`, extend `compute_per_label_metrics()`:
```python
# Add ROC-AUC per label
from sklearn.metrics import roc_auc_score

for label_idx, label_name in enumerate(label_names):
    auc = roc_auc_score(y_true[:, label_idx], y_pred[:, label_idx])
    per_label_metrics[label_name]["auc"] = auc
```

### Custom Error Categories

In `error_analysis.py`, modify error categorization:
```python
# Example: categorize by text length
short_errors = [e for e in errors if len(e["text"]) < 100]
long_errors = [e for e in errors if len(e["text"]) >= 100]
```

## Citation

If you use this analysis framework, cite:
- Per-label metrics inspired by: Sorower et al. "Machine Learning for Completeness" (2010)
- Error analysis based on: Karpukhin et al. "Error Analysis" (2020)
- Reproducibility: Science Reproducibility Guidelines (Nature, 2016)
