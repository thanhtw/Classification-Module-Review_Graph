# Classification Module - Review Graph

A comprehensive multilabel text classification system supporting multiple model architectures including BERT, RoBERTa, CNN, LSTM, BiLSTM, SVM, and LLM-based approaches.

## 📁 Project Structure

```
.
├── docs/                                    # Project documentation
│   ├── README.md                           # Original project readme
│   ├── TRAINING_PIPELINE_GUIDE.md          # Detailed training pipeline documentation
│   ├── HOW_TO_USE_TRAINING_PIPELINE.md     # Usage instructions
│   ├── ADVANCED_ANALYSIS_GUIDE.md          # Advanced analysis techniques
│   └── QUICK_START_ANALYSIS.md             # Quick start guide
│
├── src/                                     # Source code
│   ├── training/                           # Training configuration and orchestration
│   │   ├── __init__.py
│   │   ├── config.py                       # Centralized model and training configurations
│   │   └── trainer.py                      # Main training orchestration (if needed)
│   │
│   ├── models/                             # Model implementations
│   │   ├── __init__.py
│   │   ├── models_ml.py                    # ML models (SVM, Logistic Regression, Naive Bayes)
│   │   ├── models_nn.py                    # Neural network models (CNN, LSTM, BiLSTM, RNN)
│   │   ├── models_transformers.py          # Transformer models (BERT, RoBERTa)
│   │   ├── models_llm.py                   # Large Language Model implementations
│   │   └── models_svm.py                   # SVM-specific implementations
│   │
│   ├── data/                               # Data processing and loading
│   │   ├── __init__.py
│   │   └── preprocessor.py                 # Data preprocessing and utilities
│   │
│   ├── embeddings/                         # Embedding and vectorization
│   │   ├── __init__.py
│   │   └── word2vec.py                     # Word2Vec vectorizer implementation
│   │
│   ├── utils/                              # Shared utilities and helpers
│   │   ├── __init__.py
│   │   ├── metrics.py                      # Evaluation metrics computation
│   │   ├── reporting.py                    # Report generation utilities
│   │   ├── per_label_metrics.py            # Per-label metric calculations
│   │   ├── smote.py                        # SMOTE/class imbalance handling
│   │   ├── mlsmote.py                      # Multilabel SMOTE implementation
│   │   └── reproducibility.py              # Seed and reproducibility utilities
│   │
│   ├── inference/                          # Inference pipeline
│   │   ├── __init__.py
│   │   └── predictor.py                    # Predictor/inference utilities
│   │
│   └── analysis/                           # Analysis and error investigation
│       ├── __init__.py
│       ├── error_analysis.py               # Error analysis tools
│       ├── analysis_utils.py               # Analysis utilities
│       └── report_enhanced.py              # Enhanced reporting
│
├── scripts/                                 # Executable scripts
│   ├── __init__.py
│   ├── train.py                            # Main training script
│   ├── inference.py                        # Inference script
│   ├── analyze.py                          # Analysis script
│   ├── hyperparameter_summary.py           # Hyperparameter summary tool
│   ├── extract_best_folds.py               # Extract best performing folds
│   └── data_processing_3label.py           # 3-label data processing utility
│
├── data/                                    # Data directory
│   └── cleaned_3label_data.csv             # Training data
│
├── embeddings/                              # Pre-trained embeddings storage
│   └── cc.zh.300.vec.gz                    # Chinese FastText/Word2Vec embeddings
│
├── results/                                 # Training results and artifacts
│   └── modular_multimodel/                 # Multimodel training results
│
├── tests/                                   # Unit tests (to be added)
│
├── .gitignore
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n classification python=3.10
conda activate classification
conda install --file requirements.txt
```

### Training

Run the main training pipeline with default settings:

```bash
python scripts/train.py
```

Train specific models:

```bash
python scripts/train.py --models linear_svm naive_bayes logistic_regression
```

With cross-validation folds:

```bash
python scripts/train.py --models bert roberta --n_folds 5
```

### Inference

```bash
python scripts/inference.py --model_path <path_to_model> --text "Your text here"
```

### Analysis

```bash
python scripts/analyze.py --results_dir results/modular_multimodel/
```

## 🏗️ Key Components

### Models (`src/models/`)
- **ML Models**: Linear SVM, Logistic Regression, Naive Bayes (with Word2Vec embeddings)
- **Neural Networks**: CNN, LSTM, BiLSTM, RNN 
- **Transformers**: BERT, RoBERTa with fine-tuning
- **LLMs**: Zero-shot and few-shot learning approaches

### Data Processing (`src/data/`)
- Chinese text preprocessing
- Tokenization and cleaning
- Feature extraction and sequence building
- Data loading and splitting

### Embeddings (`src/embeddings/`)
- Word2Vec and FastText support
- Pre-trained Chinese vectors (cc.zh.300.vec.gz)
- Document-level embedding generation

### Utilities (`src/utils/`)
- Multilabel metrics (F1-macro, F1-micro, precision, recall, hamming loss)
- SMOTE for class imbalance handling
- Reproducibility utilities (seed management)
- Report generation
- Per-label performance analysis

### Analysis Tools (`src/analysis/`)
- Error analysis and investigation tools
- Per-label metric breakdown
- Enhanced reporting capabilities

## 🔧 Configuration

Edit `src/training/config.py` to customize:

- Model hyperparameters
- Training parameters (epochs, batch size, learning rate)
- Data preprocessing options
- Cross-validation splits
- Label columns and names

Example:

```python
from src.training.config import CommonConfig, LABEL_COLUMNS

config = CommonConfig(
    test_size=0.2,
    random_state=42,
    use_smote=True,
)
```

## 📊 Output Structure

Training results are organized as:

```
results/modular_multimodel/
├── best_fold_per_model.csv                # Best fold performance per model
├── model_comparison_macro_micro.csv       # Cross-model comparison metrics
├── model_ranking_by_macro_micro_f1.csv    # Model ranking by F1 scores
├── model_results_detailed.csv             # Detailed per-fold results
├── SUMMARY_REPORT.txt                     # Human-readable summary
├── training_process.json                  # Full training process metadata
├── training_process.jsonl                 # Line-delimited training log
├── global_train_data_analysis/            # Training data statistics
├── model_artifacts/                       # Saved models and vectorizers
│   ├── linear_svm/fold_*/
│   ├── logistic_regression/fold_*/
│   └── ...
└── temp/                                  # Temporary files
```

## 📝 Documentation

- **[Training Pipeline Guide](docs/TRAINING_PIPELINE_GUIDE.md)**: Comprehensive training documentation
- **[Usage Guide](docs/HOW_TO_USE_TRAINING_PIPELINE.md)**: Step-by-step usage instructions
- **[Advanced Analysis](docs/ADVANCED_ANALYSIS_GUIDE.md)**: Advanced analysis techniques
- **[Quick Start](docs/QUICK_START_ANALYSIS.md)**: Quick reference guide

## 🔄 Import Pattern

The project uses a consistent import pattern for maintainability:

```python
# Models
from src.models.models_ml import run_linear_svm
from src.models.models_nn import run_lstm_like

# Utilities
from src.utils.metrics import compute_metrics
from src.utils.smote import apply_smote_multilabel

# Data and embeddings
from src.data.preprocessor import load_and_clean_data
from src.embeddings.word2vec import Word2VecVectorizer

# Configuration
from src.training.config import LABEL_COLUMNS, CommonConfig
```

## 🛠️ Development

To add new functionality:

1. **New model**: Add to `src/models/models_<type>.py`
2. **New utility**: Add to `src/utils/<utility>.py`
3. **New analysis tool**: Add to `src/analysis/<tool>.py`
4. **New script**: Add to `scripts/<script>.py` and update imports

## 📦 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- Gensim (Word2Vec)
- Pandas, NumPy
- See `requirements.txt` for full list

## ✨ Key Features

- **Multi-Model Support**: 10+ model architectures
- **Multilabel Classification**: Handles multiple labels per sample
- **Class Imbalance Handling**: SMOTE for balanced training
- **Cross-Validation**: K-fold validation support
- **Comprehensive Metrics**: Per-label and aggregated metrics
- **Model Artifacts**: Full reproducibility with saved models
- **Detailed Analysis**: Error analysis and per-label breakdowns
- **Chinese Text Support**: Optimized for Chinese NLP

## 📄 License

See LICENSE file for details

## 👥 Contributors

Developed for multilabel text classification research and production use
