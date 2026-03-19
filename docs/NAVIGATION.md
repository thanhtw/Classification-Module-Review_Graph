# Project Navigation Guide

Quick reference for finding code, documentation, and running tasks.

## 🎯 Common Tasks

### I want to...

**Train the model**
```bash
python scripts/train.py --models linear_svm --n_folds 10
```
📁 Location: [`scripts/train.py`](../../scripts/train.py)
📖 Config: [`src/training/config.py`](../../src/training/config.py)

**Make predictions**
```bash
python scripts/inference.py --model_path results/modular_multimodel/model_artifacts/linear_svm/fold_1/
```
📁 Location: [`scripts/inference.py`](../../scripts/inference.py)

**Analyze results**
```bash
python scripts/analyze.py
```
📁 Location: [`scripts/analyze.py`](../../scripts/analyze.py)
📁 Analysis tools: [`src/analysis/`](../../src/analysis/)

**Understand the project structure**
📖 Read: [`STRUCTURE.md`](STRUCTURE.md)
📖 Quick ref: [`README.md`](../../README.md)

**Add a new model**
1. Edit: [`src/models/models_<type>.py`](../../src/models/)
2. Update: [`src/training/config.py`](../../src/training/config.py) (if adding new config)
3. Reference: [`STRUCTURE.md#src-models`](STRUCTURE.md#src-models)

**Debug data issues**
📁 Data preprocessor: [`src/data/preprocessor.py`](../../src/data/preprocessor.py)
📁 Data processing script: [`scripts/data_processing_3label.py`](../../scripts/data_processing_3label.py)

**Debug metric calculations**
📁 Metrics module: [`src/utils/metrics.py`](../../src/utils/metrics.py)
📁 Per-label metrics: [`src/utils/per_label_metrics.py`](../../src/utils/per_label_metrics.py)

**Handle class imbalance**
📁 SMOTE utilities: [`src/utils/smote.py`](../../src/utils/smote.py)
📁 Multilabel SMOTE: [`src/utils/mlsmote.py`](../../src/utils/mlsmote.py)

---

## 📂 Directory Map

| Path | Purpose | Key Files |
|------|---------|-----------|
| `src/training/` | Configuration | `config.py` |
| `src/models/` | Model implementations | `models_*.py` |
| `src/data/` | Data loading | `preprocessor.py` |
| `src/embeddings/` | Text vectorization | `word2vec.py` |
| `src/utils/` | Shared utilities | `metrics.py`, `smote.py`, `reporting.py` |
| `src/analysis/` | Result analysis | `error_analysis.py`, `analysis_utils.py` |
| `src/inference/` | Prediction pipelines | (placeholder) |
| `scripts/` | Executable scripts | `train.py`, `analyze.py`, `inference.py` |
| `data/` | Datasets | `cleaned_3label_data.csv` |
| `embeddings/` | Pretrained vectors | `cc.zh.300.vec.gz` |
| `results/` | Model outputs | `modular_multimodel/` |
| `docs/` | Documentation | `README.md`, `STRUCTURE.md` |

---

## 🔍 Finding Code

### By Functionality

**Models**
- Linear SVM: [`src/models/models_svm.py`](../../src/models/models_svm.py), [`src/models/models_ml.py`](../../src/models/models_ml.py)
- Logistic Regression, Naive Bayes: [`src/models/models_ml.py`](../../src/models/models_ml.py)
- CNN, LSTM, BiLSTM, RNN: [`src/models/models_nn.py`](../../src/models/models_nn.py)
- BERT, RoBERTa: [`src/models/models_transformers.py`](../../src/models/models_transformers.py)
- LLM-based: [`src/models/models_llm.py`](../../src/models/models_llm.py)

**Utilities**
- Metrics: [`src/utils/metrics.py`](../../src/utils/metrics.py)
- SMOTE: [`src/utils/smote.py`](../../src/utils/smote.py)
- Reporting: [`src/utils/reporting.py`](../../src/utils/reporting.py)
- Reproducibility: [`src/utils/reproducibility.py`](../../src/utils/reproducibility.py)

**Data Processing**
- Preprocessing: [`src/data/preprocessor.py`](../../src/data/preprocessor.py)
- Embeddings: [`src/embeddings/word2vec.py`](../../src/embeddings/word2vec.py)

**Analysis**
- Error analysis: [`src/analysis/error_analysis.py`](../../src/analysis/error_analysis.py)
- Reports: [`src/analysis/report_enhanced.py`](../../src/analysis/report_enhanced.py)

---

## 📚 Documentation Map

| Document | Purpose |
|----------|---------|
| [`README.md`](../../README.md) | Project overview and quick start |
| [`STRUCTURE.md`](STRUCTURE.md) | Detailed structure and import patterns |
| [`TRAINING_PIPELINE_GUIDE.md`](TRAINING_PIPELINE_GUIDE.md) | Training methodology and details |
| [`HOW_TO_USE_TRAINING_PIPELINE.md`](HOW_TO_USE_TRAINING_PIPELINE.md) | Step-by-step usage guide |
| [`ADVANCED_ANALYSIS_GUIDE.md`](ADVANCED_ANALYSIS_GUIDE.md) | Advanced analysis techniques |
| [`QUICK_START_ANALYSIS.md`](QUICK_START_ANALYSIS.md) | Quick reference guide |

---

## 💻 Command Reference

```bash
# Training
python scripts/train.py                                           # All models, default settings
python scripts/train.py --models linear_svm naive_bayes          # Specific models
python scripts/train.py --n_folds 5                              # 5-fold CV
python scripts/train.py --no_smote                               # No SMOTE

# Analysis
python scripts/analyze.py                                         # Comprehensive analysis

# Inference
python scripts/inference.py --model_path <path> --text "text"    # Make prediction

# Utilities
python scripts/hyperparameter_summary.py                          # Show all hyperparameters
python scripts/extract_best_folds.py                             # Extract best models
python scripts/data_processing_3label.py                         # Process multilabel data
```

---

## 🔗 Import Cheat Sheet

```python
# Models
from src.models.models_ml import run_linear_svm, run_naive_bayes
from src.models.models_nn import run_lstm_like, run_cnn_attention
from src.models.models_transformers import run_transformer
from src.models.models_llm import run_llm_zero_few_shot

# Config
from src.training.config import LABEL_COLUMNS, CommonConfig

# Data
from src.data.preprocessor import load_and_clean_data, tokenize_text
from src.embeddings.word2vec import Word2VecVectorizer

# Utils
from src.utils.metrics import compute_metrics
from src.utils.smote import apply_smote_multilabel
from src.utils.reproducibility import set_seed

# Analysis
from src.analysis.error_analysis import analyze_errors

# Scripts
from scripts.train import parse_args, main
```

---

## ❓ Troubleshooting

**Import errors?**
→ Ensure you're using `ThomasAgent` conda environment to avoid NumPy 2.x issues
→ Read: [`STRUCTURE.md#import-patterns`](STRUCTURE.md#import-patterns)

**Can't find a file?**
→ Check [`STRUCTURE.md`](STRUCTURE.md) for directory organization
→ Use: `find . -name "*filename*" -type f`

**Don't know where to add new code?**
→ Read: [`STRUCTURE.md#benefits-of-this-structure`](STRUCTURE.md#benefits-of-this-structure)
→ See: [`STRUCTURE.md#when-adding-new-code`](STRUCTURE.md#when-adding-new-code)

**Results look wrong?**
→ Check: [`src/utils/metrics.py`](../../src/utils/metrics.py)
→ Debug: [`src/analysis/error_analysis.py`](../../src/analysis/error_analysis.py)

---

## 📝 Labels

- **relevance**: Is content relevant to query?
- **concreteness**: Does content provide concrete information?
- **constructive**: Is feedback constructive?

All labels are binary (0 or 1) per document → **Multilabel Classification**
