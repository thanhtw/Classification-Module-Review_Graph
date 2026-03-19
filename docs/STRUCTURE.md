# Project Structure Overview

## Directory Organization

This project follows a modern Python package structure for clarity and maintainability.

### Root Level
- `src/` - Main source code package (organized by functionality)
- `scripts/` - Executable scripts (each serves a specific purpose)
- `data/` - Raw and processed datasets
- `embeddings/` - Pre-trained embedding files
- `results/` - Training outputs and model artifacts
- `tests/` - Unit and integration tests
- `docs/` - Project documentation
- `requirements.txt` - Python dependencies
- `README.md` - Project overview and quick start
- `.gitignore` - Git ignore rules
- `.git/` - Git repository

---

## `src/` - Source Code Organization

The source code is organized by **functional domain** rather than by file type (models, tests, utils, etc.). This makes it easier to understand the project's architecture.

### `src/training/`
**Purpose**: Training configuration and orchestration

- `config.py` - Centralized configuration for all models, hyperparameters, and training settings
  - Model configs: `CNNConfig`, `RNNConfig`, `TransformerConfig`, `LLMConfig`
  - Common settings: label columns, available models, default parameters
  - Used by all model implementations

### `src/models/`
**Purpose**: Model implementations for different architectures

- `models_ml.py` - Classical machine learning models
  - Linear SVM (OneVsRest)
  - Logistic Regression (OneVsRest)
  - Gaussian Naive Bayes (OneVsRest)
  - All use Word2Vec embeddings

- `models_nn.py` - Neural network models (PyTorch)
  - CNN with attention mechanism
  - LSTM-based models
  - BiLSTM models
  - RNN models
  - Uses GloVe or FastText embeddings

- `models_transformers.py` - Transformer-based models
  - BERT (distilbert-base-chinese)
  - RoBERTa (roberta-base or roberta-large)
  - Fine-tuning with Hugging Face Trainer

- `models_llm.py` - Large Language Model approaches
  - Zero-shot classification
  - Few-shot learning
  - Uses Qwen2 or LLaMa models

- `models_svm.py` - SVM-specific implementations (extends models_ml.py)

### `src/data/`
**Purpose**: Data loading, preprocessing, and preparation

- `preprocessor.py` (originally `data_utils.py`)
  - Load CSV data and validate structure
  - Preprocess Chinese text (character cleaning, tokenization)
  - Build vocabulary and convert texts to sequences
  - Set random seeds for reproducibility
  - Handle missing values and data validation

### `src/embeddings/`
**Purpose**: Text embedding and vectorization

- `word2vec.py` (originally `embeddings_word2vec.py`)
  - Word2Vec vectorizer implementation
  - Trains on corpus or loads pre-trained vectors
  - Generates document-level embeddings via averaging
  - Handles out-of-vocabulary words gracefully
  - Serializable for model persistence

### `src/utils/`
**Purpose**: Shared utilities and helper functions

- `metrics.py` (originally `metrics_utils.py`)
  - Multilabel evaluation metrics
  - F1-score (macro and micro)
  - Precision, recall, accuracy
  - Hamming loss
  - Subset accuracy

- `reporting.py` (originally `report_utils.py`)
  - Generate training reports
  - Export results to CSV/JSON
  - Format outputs for analysis

- `per_label_metrics.py`
  - Calculate metrics per label/class
  - Useful for evaluating class imbalance issues

- `smote.py` (originally `smote_utils.py`)
  - Apply SMOTE for multilabel classification
  - Handle class imbalance in training data
  - Works with dense arrays (Word2Vec, GloVe)

- `mlsmote.py`
  - Multilabel-specific SMOTE implementation
  - Extends standard SMOTE to multilabel setting

- `reproducibility.py`
  - Set random seeds across libraries (numpy, torch, random)
  - Ensure deterministic training
  - Configure CUDA determinism

### `src/inference/`
**Purpose**: Prediction and inference utilities

Currently a placeholder for inference utilities. To be expanded with:
- Model loading and prediction
- Batch inference support
- Confidence score calculation
- Output formatting

### `src/analysis/`
**Purpose**: Data analysis and model evaluation tools

- `error_analysis.py`
  - Analyze model errors
  - Identify misclassifications
  - Error patterns and statistics

- `analysis_utils.py`
  - Export SMOTE analysis results
  - Generate data distribution reports
  - Statistical analysis of training data

- `report_enhanced.py`
  - Generate detailed analysis reports
  - Per-label performance breakdown
  - Model comparison visualizations

---

## `scripts/` - Executable Scripts

Each script in this directory is a **standalone executable** for a specific task.

- `train.py` (original: `run_modular_multimodel_train.py`)
  - Main training script
  - Usage: `python scripts/train.py [--models model_name] [options]`
  - Orchestrates cross-validation training across multiple models
  - Generates results and model artifacts

- `inference.py` (from `function/`)
  - Perform inference/predictions
  - Load trained models and make predictions on new data
  - Usage: `python scripts/inference.py --model_path <path>`

- `analyze.py` (from `function/comprehensive_analysis.py`)
  - Comprehensive analysis of results
  - Generate analysis reports
  - Usage: `python scripts/analyze.py`

- `hyperparameter_summary.py` (from `model_hyperparameters_summary.py`)
  - Summarize all model hyperparameters
  - Generate hyperparameter reference document
  - Usage: `python scripts/hyperparameter_summary.py`

- `extract_best_folds.py` (from `utils/`)
  - Extract best performing fold for each model
  - Useful for final model selection
  - Usage: `python scripts/extract_best_folds.py`

- `data_processing_3label.py` (from `utils/3label_processing.py`)
  - Process 3-label multilabel data
  - Data format conversion and validation
  - Usage: `python scripts/data_processing_3label.py`

---

## `data/` - Datasets

- `cleaned_3label_data.csv` - Main training dataset
  - Format: `text, label1, label2, label3` (columns: "relevance", "concreteness", "constructive")
  - Multilabel binary classification (each label: 0 or 1)

---

## `embeddings/` - Pre-trained Embeddings

- `cc.zh.300.vec.gz` - Chinese FastText embeddings
  - Dimension: 300
  - Language: Chinese
  - Format: FastText (.vec.gz)
  - Can be used by Word2Vec, GloVe, or neural models

---

## `results/` - Training Outputs

```
results/modular_multimodel/
├── best_fold_per_model.csv              # Best fold for each model
├── model_comparison_macro_micro.csv     # Metrics across models
├── model_ranking_by_macro_micro_f1.csv  # Models ranked by performance
├── model_results_detailed.csv           # Per-fold results
├── SUMMARY_REPORT.txt                   # Human-readable summary
├── training_process.json                # Full training metadata
├── training_process.jsonl               # Training log (line-delimited)
├── global_train_data_analysis/          # Data statistics
│   ├── train_*_distribution.csv
│   ├── train_*_analysis_summary.json
│   └── train_*_characteristics.csv
├── model_artifacts/                     # Saved models and vectorizers
│   ├── linear_svm/fold_1/
│   │   ├── model.pkl
│   │   └── metadata.json
│   ├── logistic_regression/fold_*/
│   ├── naive_bayes/fold_*/
│   ├── cnn_attention/fold_*/
│   ├── lstm/fold_*/
│   ├── bilstm/fold_*/
│   ├── bert/fold_*/
│   ├── roberta/fold_*/
│   └── ...
└── temp/                                # Temporary files (can be removed)
```

---

## Import Patterns

### Standard Imports (Recommended)

```python
# Models
from src.models.models_ml import run_linear_svm, run_naive_bayes
from src.models.models_nn import run_lstm_like, run_cnn_attention
from src.models.models_transformers import run_transformer
from src.models.models_llm import run_llm_zero_few_shot

# Configuration
from src.training.config import LABEL_COLUMNS, CommonConfig, CNNConfig

# Data and Embeddings
from src.data.preprocessor import load_and_clean_data, tokenize_text
from src.embeddings.word2vec import Word2VecVectorizer

# Utilities
from src.utils.metrics import compute_metrics
from src.utils.smote import apply_smote_multilabel
from src.utils.reproducibility import set_seed
from src.utils.reporting import export_results

# Analysis
from src.analysis.error_analysis import analyze_errors
from src.analysis.analysis_utils import export_train_smote_analysis
```

### When Adding New Code

1. **New ML Model**:
   ```python
   # Add to src/models/models_ml.py or create models_<type>.py
   def run_my_model(...): pass
   ```

2. **New Utility Function**:
   ```python
   # Add to src/utils/<category>.py
   def my_utility_function(): pass
   ```

3. **New Analysis Tool**:
   ```python
   # Add to src/analysis/<tool>.py
   def analyze_something(): pass
   ```

4. **New Script**:
   ```python
   # Create scripts/<script>.py
   # Import from src package using absolute imports
   from src.models.models_ml import run_linear_svm
   ```

---

## Benefits of This Structure

✅ **Clarity**: Organized by functionality, not file type
✅ **Maintainability**: Easy to locate and modify related code
✅ **Scalability**: Simple to add new models or utilities
✅ **Testability**: Each module can be tested independently
✅ **Imports**: Consistent import patterns across the project
✅ **Collaboration**: Clear ownership and purpose of each module
✅ **Documentation**: Self-documenting through folder and file names
