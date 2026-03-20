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
│   ├── train.py                            # Main training script (supports 8+ model types)
│   ├── research_comparison.py              # Research paper generation & comparison
│   └── research_modules/                   # Organized research output generation
│       ├── __init__.py
│       ├── visualizations.py               # Publication-ready visualizations
│       ├── metrics_analysis.py             # Comprehensive metrics analysis
│       ├── report_builders.py              # Research report generation
│       └── table_generators.py             # Comparison tables and summaries
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

Run the main training pipeline with all supported models (default: 8 models, 1 fold):

```bash
python scripts/train.py
```

Train specific models:

```bash
python scripts/train.py --models linear_svm naive_bayes logistic_regression
```

Train with cross-validation:

```bash
python scripts/train.py --models bert roberta cnn_attention lstm bilstm --n_folds 10
```

Train with SMOTE and custom settings:

```bash
python scripts/train.py \
  --models lstm bilstm \
  --n_folds 5 \
  --rnn_epochs 15 \
  --seed 42 \
  --glove_path embeddings/cc.zh.300.vec.gz
```

### Research Paper Generation & Comparison

Generate comprehensive research paper with visualizations and metrics analysis:

```bash
python scripts/research_comparison.py --n_folds 10 --seed 42
```

This generates:
- Multilabel-specific metrics reports
- Per-label confusion matrices (normalized heatmaps)
- Training curves for deep learning models
- Model comparison visualizations
- Comprehensive LaTeX tables and metrics summaries
- Research paper appendix with detailed analysis

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

## 🔧 Configuration & Customization

### Training Configuration

Edit `src/training/config.py` to customize globally:

```python
from src.training.config import (
    LABEL_COLUMNS,           # Classification labels
    CommonConfig,            # Common training settings
    CNNConfig,               # CNN hyperparameters
    RNNConfig,               # LSTM/BiLSTM hyperparameters
    TransformerConfig,       # BERT/RoBERTa settings
    LLMConfig,              # LLM parameters
)
```

Configuration options:

```python
# Common settings (used by all models)
CommonConfig(
    test_size=0.2,           # Holdout test split
    seed=42,                 # Random seed
    use_smote=True,          # Enable SMOTE for imbalanced data
    output_dir="results/modular_multimodel",
)

# Neural Network settings
RNNConfig(
    epochs=10,
    batch_size=32,
    embedding_dim=300,       # Word embedding dimension
    hidden_dim=256,
    lr=0.001,
    dropout=0.3,
    glove_path="embeddings/cc.zh.300.vec.gz",
    glove_trainable=False,   # Freeze pretrained embeddings
)

# Transformer settings
TransformerConfig(
    model_name="bert-base-chinese",
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
)
```

### Command-Line Arguments

```bash
# Model selection (default: all 8 models)
--models bert roberta lstm bilstm linear_svm naive_bayes logistic_regression cnn_attention

# Cross-validation
--n_folds 10                 # Number of folds (1 = holdout split)

# Training parameters
--rnn_epochs 15              # Epochs for LSTM/BiLSTM
--cnn_epochs 8               # Epochs for CNN
--bert_epochs 5              # Epochs for BERT
--roberta_epochs 5           # Epochs for RoBERTa

# Data & embeddings
--test_size 0.2              # Test split ratio
--seed 42                    # Random seed
--glove_path embeddings/cc.zh.300.vec.gz
--freeze_glove               # Freeze embedding layer

# LLM parameters (zero-shot/few-shot)
--llm_model_name "llama-3.1-8b-instant"
--llm_few_shot_k 100         # Number of few-shot examples
--llm_max_new_tokens 64
--llm_temperature 0.0        # Deterministic output

# Output
--output_dir results/modular_multimodel
```

## 📊 Output Structure

Training results are organized in `results/modular_multimodel/`:

```
results/modular_multimodel/
├── best_fold_per_model.csv                    # Best performing fold per model
├── model_comparison_macro_micro.csv           # F1-macro vs F1-micro comparison
├── model_ranking_by_macro_micro_f1.csv        # Models ranked by F1 scores
├── model_results_detailed.csv                 # Per-fold detailed metrics
├── training_process.json                      # Full training metadata
├── training_process.jsonl                     # Line-delimited training log
│
├── global_train_data_analysis/                # Training data statistics
│   ├── train_before_smote.csv
│   ├── train_after_smote_labels.csv
│   └── train_features_after_smote.npz
│
├── model_artifacts/                           # Saved models & vectorizers
│   ├── linear_svm/fold_1/
│   ├── naive_bayes/fold_1/
│   ├── logistic_regression/fold_1/
│   ├── cnn_attention/fold_1/
│   │   ├── model.pt                           # PyTorch model weights
│   │   ├── metadata.json                      # Model hyperparameters
│   │   └── training_history.json              # Training curves data
│   ├── lstm/fold_1/
│   ├── bilstm/fold_1/
│   ├── bert/fold_1/
│   │   ├── model.safetensors                  # Model weights (HuggingFace format)
│   │   ├── tokenizer.json
│   │   └── config.json
│   ├── roberta/fold_1/
│   ├── llm_zero_shot/fold_1/
│   ├── llm_few_shot/fold_1/
│   └── ...
│
└── temp/                                      # Temporary training files
```

Research paper generation (`scripts/research_comparison.py`) produces:

```
results/research_comparison/
├── smote_impact_visualization.png              # SMOTE class balance visualization
├── model_f1_comparison.png                     # F1-Macro vs F1-Micro comparison
├── model_multilabel_metrics.png                # Hamming loss vs Subset accuracy
├── confusion_matrices_all_models.png           # Binary confusion matrices
├── training_curves.png                         # Training/validation curves (DL models)
├── comprehensive_metrics_heatmap.png           # All metrics × all models heatmap
│
├── per_label_confusion_matrices/               # Per-class normalized heatmaps
│   ├── confusion_matrix_linear_svm.png         # [relevance, concreteness, constructive]
│   ├── confusion_matrix_bert.png
│   ├── confusion_matrix_lstm.png
│   └── ...                                     # One file per model
│
├── per_model_metrics/                          # Per-model detailed metrics
│   ├── bert.json
│   ├── roberta.json
│   └── ...
│
├── all_models_all_metrics.csv                  # Comprehensive metrics table
├── model_configurations.json                   # Model hyperparameter configs
├── dataset_report.json                         # Dataset statistics
├── training_process_report.json                # Detailed training log
├── research_paper_appendix.json                # Research findings summary
├── per_label_metrics_report.json               # Per-class performance breakdown
├── multilabel_metrics_report.json              # Multilabel-specific metrics
│
└── *.tex                                       # LaTeX tables for papers
```

## 📝 Documentation

- **[Training Pipeline Guide](docs/TRAINING_PIPELINE_GUIDE.md)**: Comprehensive training instructions and examples
- **[Usage Guide](docs/HOW_TO_USE_TRAINING_PIPELINE.md)**: Step-by-step usage and configuration
- **[Advanced Analysis](docs/ADVANCED_ANALYSIS_GUIDE.md)**: Advanced analysis techniques and troubleshooting
- **[Quick Start Guide](docs/QUICK_START_ANALYSIS.md)**: Quick reference for common tasks
- **[Metrics Reference](docs/METRICS_REFERENCE.md)**: Explanation of all evaluation metrics
- **[Research Methodology](docs/RESEARCH_METHODOLOGY.md)**: Detailed research approach and methodology
- **[Project Structure](docs/STRUCTURE.md)**: In-depth project organization guide
- **[Navigation Guide](docs/NAVIGATION.md)**: Guide to finding specific features and tools

## � Visualizations & Research Features

### Publication-Ready Visualizations

The project generates **300 DPI publication-quality** visualizations automatically:

1. **Per-Label Confusion Matrices** (`per_label_confusion_matrices/`)
   - Normalized heatmaps for each class
   - Shows prediction performance per label
   - One visualization per model across all 3 labels
   - Format: `confusion_matrix_{model_name}.png`

2. **Model Performance Comparisons**
   - F1-Macro vs F1-Micro bar charts
   - Multilabel-specific metrics (Hamming loss, Subset accuracy)
   - SMOTE impact visualization showing class balance

3. **Training Curves** (`training_curves.png`)
   - Training/validation loss across epochs
   - F1-Macro and F1-Micro trends
   - Deep learning models (LSTM, CNN, BiLSTM)

4. **Comprehensive Metrics Heatmap** (`comprehensive_metrics_heatmap.png`)
   - All models × all metrics visualization
   - Normalized scores for easy comparison
   - Identifies best performers across metrics

### Research Paper Components

`scripts/research_comparison.py` generates complete research artifacts:

- **Metrics Reports**: Per-label, multilabel, and aggregated metrics
- **Model Configurations**: Hyperparameters for all models in JSON format
- **Training Process Report**: Detailed training logs and statistics
- **LaTeX Tables**: Publication-ready tables for research papers
- **Dataset Report**: Statistical breakdown of training data
- **Research Appendix**: Comprehensive findings and analysis

### Deep Learning Training History

Neural network models (CNN, LSTM, BiLSTM) now save training history:

```
model_artifacts/{model}/fold_N/training_history.json
```

Contains:
- `train_loss`, `val_loss` per epoch
- `train_f1_macro`, `val_f1_macro` per epoch  
- `train_f1_micro`, `val_f1_micro` per epoch
- Used for generating publication-quality training curves

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

# Research & Visualization
from scripts.research_modules import (
    generate_per_label_confusion_matrices,
    generate_training_curves,
    generate_model_comparison_visualizations,
)
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

- **Multi-Model Support**: 10+ model architectures (ML, DL, Transformers, LLMs)
- **Multilabel Classification**: Handles multiple labels per sample with specialized metrics
- **Class Imbalance Handling**: SMOTE with multilabel support for balanced training
- **Cross-Validation**: K-fold validation for robust evaluation
- **Comprehensive Metrics**: F1-macro, F1-micro, Hamming loss, Subset accuracy, per-label breakdown
- **Model Artifacts**: Full reproducibility with saved models, weights, and hyperparameters
- **Detailed Analysis**: Per-label confusion matrices, error analysis, metric breakdowns
- **Chinese Text Support**: Pre-trained Chinese embeddings (FastText cc.zh.300)
- **Publication-Ready Visualizations**: 300 DPI figures for research papers
- **Training History Tracking**: Automatic epoch-by-epoch loss and metrics for DL models
- **Research Automation**: One-command generation of comprehensive research reports
- **Organized Research Modules**: Structured pipeline for metrics, visualizations, and tables

## 📄 License

See LICENSE file for details

## 👥 Contributors

Developed for multilabel text classification research and production use
