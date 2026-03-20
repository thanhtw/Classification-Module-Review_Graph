# Research Paper Generation: Comprehensive Multilabel Classification Analysis

## Overview

This pipeline generates comprehensive research reports, visualizations, and metrics analysis for multilabel text classification comparing **10+ model architectures** across various training configurations.

Supported models:
- **Machine Learning**: Linear SVM, Logistic Regression, Naive Bayes
- **Deep Learning**: CNN + Attention, LSTM, BiLSTM
- **Transformers**: BERT, RoBERTa
- **LLM-based**: Llama 3.1 (Few-shot, Zero-shot)

## Pipeline Structure

### 1. **Dataset Analysis** (`generate_dataset_report()`)
- Loads cleaned dataset: `data/cleaned_3label_data.csv`
- Computes comprehensive statistics:
  - Sample size, label cardinality, label density
  - Text length distribution (characters, tokens)
  - Per-label positive/negative ratios and imbalance metrics
  - Label co-occurrence patterns
- Output: `results/research_comparison/dataset_report.json`

### 2. **SMOTE Impact Analysis** (via `visualizations.py`)
- **Visualization**: `results/research_comparison/smote_impact_visualization.png`
- Compares training data before/after SMOTE-based balancing
- Shows per-label class distribution changes
- Displays sample growth and balance improvement metrics
- Includes per-label histogram comparisons (before/after)

### 3. **Model Training** (via `scripts/train.py`)
- Trains all configured models (8 by default)
- **Cross-validation**: K-fold (default 10 folds) with stratified splits
- **Data augmentation**: SMOTE applied to training fold only (no data leakage)
- **Per-fold outputs**:
  - Model checkpoints/weights
  - Predictions on validation fold
  - Per-fold performance metrics
  - Training history (for DL models)
- **Aggregated metrics**: Mean, std, min/max across all folds
- **Saved artifacts**: `results/modular_multimodel/model_artifacts/`

### 4. **Training Curves Generation** (`generate_training_curves()`)
- **Visualization**: `results/research_comparison/training_curves.png`
- Applicable to: CNN, LSTM, BiLSTM models
- Shows per-epoch metrics:
  - Training loss vs Validation loss
  - Training F1-Macro vs Validation F1-Macro
  - Training F1-Micro vs Validation F1-Micro
- Loaded from: `model_artifacts/{model}/fold_1/training_history.json`
- Automatically generated when training history exists

### 5. **Per-Label Confusion Matrices** (`generate_per_label_confusion_matrices()`)
- **Visualization Directory**: `results/research_comparison/per_label_confusion_matrices/`
- **Output Files**: One PNG per model
  - `confusion_matrix_linear_svm.png`
  - `confusion_matrix_lstm.png`
  - `confusion_matrix_bert.png`
  - ... (one for each model)
- **Contents**: 
  - 3 subplots (one per label: relevance, concreteness, constructive)
  - Normalized heatmaps showing prediction patterns
  - Row-normalized values (proportions for each true label)
  - Color-coded for easy interpretation
  - 300 DPI publication quality

### 6. **Model Comparison Analysis** (`generate_model_comparison_visualizations()`)
- **F1 Score Comparison**: `model_f1_comparison.png`
  - Bar charts: F1-Macro vs F1-Micro for each model
  - Side-by-side comparison across all models
  - Value labels for precise readings

- **Multilabel Metrics**: `model_multilabel_metrics.png`
  - Hamming Loss comparison (lower = better)
  - Subset Accuracy comparison (higher = better)
  - Horizontal bar charts for easy ranking

### 7. **Comprehensive Metrics Heatmap** (`generate_comprehensive_heatmaps()`)
- **Visualization**: `results/research_comparison/comprehensive_metrics_heatmap.png`
- All models × all metrics display
- Normalized scores (0-1) for fair comparison
- Identifies performance leaders across metrics
- Publication-ready 300 DPI format

### 8. **Detailed Metrics Reports** (`metrics_analysis.py`)

#### Per-Label Metrics Report
- File: `per_label_metrics_report.json`
- Contains:
  - Precision, Recall, F1-score per label
  - Per-label performance breakdown
  - Label-specific insights (which models excel at which labels)
  - Per-fold variation analysis

#### Multilabel Metrics Report
- File: `multilabel_metrics_report.json`
- Contains:
  - Hamming Loss (avg label error rate)
  - Subset Accuracy (exact match rate)
  - Jaccard Index
  - Zero-one Loss
  - Aggregate statistics across folds

### 9. **Comprehensive Comparison Tables** (`table_generators.py`)
- File: `all_models_all_metrics.csv`
- Contains: All models × all computed metrics in tabular format
- Includes: Mean, std, min, max per model
- LaTeX export: `comprehensive_model_comparison.tex` (for research papers)

### 10. **Model Configurations Report** (`generate_model_configurations()`)
- File: `model_configurations.json`
- Stores hyperparameters for each model:
  - Architecture details
  - Training parameters (epochs, batch size, lr)
  - Regularization settings
  - Embedding specifications (for neural models)
  - Reproducibility info (seed, padding settings)

### 11. **Training Process Report** (`generate_training_process_report()`)
- File: `training_process_report.json`
- Records:
  - Per-fold execution details
  - Training time per model
  - Inference time per model
  - SMOTE application details
  - Resource utilization summary

### 12. **Research Paper Appendix** (`generate_research_paper_appendix()`)
- File: `research_paper_appendix.json`
- Comprehensive findings summary:
  - Best performing models
  - Per-label performance analysis
  - Model ranking and recommendations
  - Statistical significance notes
  - Limitations and future work suggestions

[... 92 additional examples following same pattern, sourced from real dataset ...]
```

**Test Input:**
```
Classify the following text according to the three labels above.
Respond ONLY with valid JSON in this format: {"relevance": 0/1, "concreteness": 0/1, "constructive": 0/1}

Text to classify: "[INPUT TEXT HERE]"
```

**Expected Response Format:**
```json
{"relevance": 1, "concreteness": 1, "constructive": 1}
```

### 5. **Model Comparison Analysis** (`generate_model_comparison()`)
- Creates side-by-side comparison of all models:
  - BERT (fine-tuned)
  - RoBERTa (fine-tuned)
  - Llama 3.1 (few-shot LLM)
- Aggregates 10-fold results with statistics:
  - Per-label metrics (precision, recall, F1)
  - Multilabel metrics (Hamming Loss, Subset Accuracy)
  - Statistical significance tests
- Output: `results/research_comparison/model_comparison_report.json`

### 6. **Visualizations** (Automatically Generated)
All visualizations are high-quality (300 DPI) for research paper publication:

#### SMOTE Visualization
- **File**: `results/research_comparison/smote_impact_analysis.png`
- **Contents**:
  - Before/after label distribution
  - New sample count per class
  - Class balance improvement metrics

#### Model Comparison Visualizations
- **F1 Score Comparison** (`model_f1_comparison.png`)
  - Grouped barplot: Per-label F1 for all models
  - Heatmap: Model performance matrix
  - Statistical annotations with significance indicators

- **Multilabel Metrics** (`model_multilabel_metrics.png`)
  - Hamming Loss comparison (lower is better)
  - Subset Accuracy comparison
  - Radar chart: Overall performance profile

## Running the Full Pipeline

### Quick Start (Default Settings)

```bash
# Generate research paper with default settings (8 models, 1 fold, cross-validation via train.py)
python scripts/research_comparison.py
```

This automatically:
1. ✅ Analyzes dataset characteristics
2. ✅ Generates SMOTE impact visualization
3. ✅ Calls train.py to train/evaluate all models
4. ✅ Generates per-label confusion matrices
5. ✅ Creates training curves (for DL models)
6. ✅ Produces comprehensive metrics reports
7. ✅ Generates publication-ready visualizations (300 DPI)
8. ✅ Creates LaTeX tables for research papers

### Full Research Pipeline (Training + Analysis)

```bash
# 1. First, train models with cross-validation (generates training_history.json)
python scripts/train.py --models bert roberta lstm bilstm cnn_attention --n_folds 10

# 2. Then, generate research reports and visualizations
python scripts/research_comparison.py --n_folds 10
```

### Custom Research Configuration

```bash
# Train specific models with custom parameters
python scripts/train.py \
  --models linear_svm logistic_regression naive_bayes bert roberta \
  --n_folds 5 \
  --seed 42 \
  --glove_path embeddings/cc.zh.300.vec.gz \
  --no_smote  # Optional: disable SMOTE

# Generate research comparison for the trained models
python scripts/research_comparison.py --seed 42
```

## Output Structure

```
results/
├── modular_multimodel/                          # From train.py
│   ├── model_artifacts/                         # Saved models and weights
│   │   ├── linear_svm/fold_1/
│   │   ├── naive_bayes/fold_1/
│   │   ├── logistic_regression/fold_1/
│   │   ├── cnn_attention/fold_1/
│   │   │   ├── model.pt
│   │   │   ├── metadata.json
│   │   │   └── training_history.json            # ← Used for training curves
│   │   ├── lstm/fold_1/
│   │   ├── bilstm/fold_1/
│   │   │   └── training_history.json
│   │   ├── bert/fold_1/
│   │   ├── roberta/fold_1/
│   │   ├── llm_zero_shot/fold_1/
│   │   └── llm_few_shot/fold_1/
│   │
│   ├── global_train_data_analysis/              # SMOTE analysis
│   │   ├── train_before_smote.csv
│   │   ├── train_after_smote_labels.csv
│   │   └── train_smote_analysis_summary.json
│   │
│   ├── best_fold_per_model.csv                  # Best fold per model
│   ├── model_comparison_macro_micro.csv         # F1-macro vs F1-micro
│   ├── model_ranking_by_macro_micro_f1.csv     # Model ranking
│   ├── model_results_detailed.csv               # Per-fold details
│   ├── training_process.json                    # Full training metadata
│   └── training_process.jsonl                   # Training log (line-delimited)
│
└── research_comparison/                         # From research_comparison.py
    ├── VISUALIZATIONS (300 DPI, Publication Quality):
    │   ├── smote_impact_visualization.png        # SMOTE before/after
    │   ├── model_f1_comparison.png               # F1-Macro vs F1-Micro
    │   ├── model_multilabel_metrics.png          # Hamming Loss vs Subset Accuracy
    │   ├── confusion_matrices_all_models.png     # Binary confusion matrices (all models)
    │   ├── training_curves.png                   # Training/validation curves (DL models)
    │   ├── comprehensive_metrics_heatmap.png     # All models × all metrics
    │
    ├── per_label_confusion_matrices/            # NEW: Normalized per-class heatmaps
    │   ├── confusion_matrix_linear_svm.png
    │   ├── confusion_matrix_naive_bayes.png
    │   ├── confusion_matrix_logistic_regression.png
    │   ├── confusion_matrix_cnn_attention.png
    │   ├── confusion_matrix_lstm.png
    │   ├── confusion_matrix_bilstm.png
    │   ├── confusion_matrix_bert.png
    │   ├── confusion_matrix_roberta.png
    │   ├── confusion_matrix_llm_zero_shot.png
    │   └── confusion_matrix_llm_few_shot.png
    │
    ├── per_model_metrics/                       # Per-model detailed metrics
    │   ├── {model_name}.json
    │   └── ...
    │
    ├── DATA & REPORTS:
    │   ├── dataset_report.json                  # Dataset statistics
    │   ├── model_configurations.json            # All hyperparameters
    │   ├── training_process_report.json         # Training details
    │   ├── per_label_metrics_report.json        # Per-class performance
    │   ├── multilabel_metrics_report.json       # Multilabel-specific metrics
    │   ├── research_paper_appendix.json         # Findings summary
    │   └── all_models_all_metrics.csv           # Comprehensive metrics table
    │
    └── LaTeX EXPORTS (for research papers):
        ├── comprehensive_model_comparison.tex
        ├── dataset_statistics.tex
        └── training_process.tex
```

## Key Research Metrics

### Per-Label Metrics (per class: relevance, concreteness, constructive)
- **Precision**: TP / (TP + FP) - Reliability of positive predictions per label
- **Recall**: TP / (TP + FN) - Ability to find positive instances per label
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Balanced metric per label
- **Support**: Number of true positive instances per label in test set

### Multilabel-Specific Metrics
- **F1-Macro**: Mean F1 across all labels (treats all labels equally)
- **F1-Micro**: Aggregated TP/FP/FN then compute F1 (sample-weighted)
- **Hamming Loss**: Fraction of labels that are incorrectly predicted (0=perfect, 1=all wrong)
- **Subset Accuracy**: % of samples where ALL labels are correctly predicted (exact match rate)
- **Jaccard Index**: (TP) / (TP + FP + FN) - Set similarity metric
- **Zero-One Loss**: Binary loss (0 if all labels match, 1 otherwise)

### Per-Model Analysis Metrics
- **Accuracy**: Overall accuracy across all samples and labels
- **Training Time**: How long training took per fold
- **Inference Time**: How fast model makes predictions
- **Memory Usage**: Model size and memory requirements
- **Fold Variability**: Std deviation of metrics across folds (stability indicator)

## Configuration

### Training Configuration (Edit `src/training/config.py`)

```python
from src.training.config import CommonConfig, RNNConfig, CNNConfig, TransformerConfig

# Common settings (apply to all models)
COMMON = CommonConfig(
    seed=42,
    test_size=0.2,           # Holdout test split (when n_folds < 2)
    use_smote=True,          # Enable SMOTE-based balancing
    output_dir="results/modular_multimodel",
)

# Neural Network settings (LSTM, BiLSTM, RNN)
RNN = RNNConfig(
    epochs=10,
    batch_size=32,
    embedding_dim=300,        # Word2Vec/FastText dimension
    hidden_dim=256,
    lr=0.001,
    dropout=0.3,
    weight_decay=0.0001,
    glove_path="embeddings/cc.zh.300.vec.gz",
    glove_trainable=False,    # Freeze pretrained embeddings
)

# CNN settings
CNN = CNNConfig(
    epochs=5,
    batch_size=32,
    embedding_dim=300,
    num_filters=100,
    filter_sizes=(3, 4, 5),
    dropout=0.3,
)

# Transformer settings (BERT, RoBERTa)
TRANSFORMER = TransformerConfig(
    model_name="bert-base-chinese",
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
)
```

### Research Comparison Configuration (Edit `scripts/research_comparison.py`)

```python
# Main configuration in research_comparison.py
run_research_comparison(
    n_folds=10,              # Number of cross-validation folds
    seed=42,                 # Random seed for reproducibility
)
```

## Customization Guide

### Add New Visualization

1. Add function to `scripts/research_modules/visualizations.py`
2. Export in `scripts/research_modules/__init__.py`
3. Call in `scripts/research_comparison.py`
4. Implementation example:

```python
def generate_custom_visualization(comparison_results, output_dir):
    """Your custom visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Your visualization code
    plt.savefig(output_dir / "custom_visualization.png", dpi=300, bbox_inches='tight')
    print(f"✓ Custom visualization saved to: {output_dir / 'custom_visualization.png'}")
    return output_dir / "custom_visualization.png"
```

### Modify Metrics Analysis

1. Edit `scripts/research_modules/metrics_analysis.py`
2. Functions: `calculate_per_label_metrics()`, `aggregate_per_label_metrics_across_folds()`
3. Add custom metric computation logic
4. Export new functions in `__init__.py` if needed

## Dependencies

```
torch>=2.0              # Deep learning
transformers>=4.30      # BERT, RoBERTa, etc.
pandas>=1.3             # Data manipulation
numpy>=1.21             # Numerical computing
matplotlib>=3.5         # Plotting
seaborn>=0.12          # Statistical visualization
scikit-learn>=1.0       # ML utilities & metrics
gensim>=4.0             # Word2Vec embeddings
groq>=0.4.1            # LLM API (optional, for LLM models)
```

## Performance Expectations

### Training Time (per fold, single GPU)
| Model | Time | Device | Notes |
|-------|------|--------|-------|
| Linear SVM | ~2-5 sec | CPU | Fast, no GPU needed |
| Logistic Regression | ~1-2 sec | CPU | Very fast |
| Naive Bayes | <1 sec | CPU | Instant |
| CNN | ~30-60 sec | GPU | Fast deep learning |
| LSTM | ~1-2 min | GPU | Moderate speed |
| BiLSTM | ~2-3 min | GPU | Slightly slower than LSTM |
| BERT | ~5-10 min | GPU | Fine-tuning intensive |
| RoBERTa | ~5-10 min | GPU | Similar to BERT |
| LLM (Llama 3.1) | ~10-20 sec per fold | API | Fast API-based inference |

**Total pipeline time (10 folds):**
- GPU: ~90-120 minutes
- CPU only: 4-6 hours

### Typical Results (10-fold CV)

| Model | F1-Macro | F1-Micro | Hamming Loss | Subset Accuracy |
|-------|----------|----------|--------------|-----------------|
| Linear SVM | 0.65-0.75 | 0.70-0.80 | 0.20-0.30 | 0.35-0.50 |
| Logistic Regression | 0.68-0.78 | 0.72-0.82 | 0.18-0.28 | 0.40-0.55 |
| Naive Bayes | 0.60-0.70 | 0.68-0.78 | 0.22-0.32 | 0.30-0.45 |
| CNN | 0.70-0.80 | 0.75-0.85 | 0.15-0.25 | 0.45-0.60 |
| LSTM | 0.72-0.82 | 0.77-0.87 | 0.13-0.23 | 0.50-0.65 |
| BiLSTM | 0.75-0.85 | 0.80-0.90 | 0.10-0.20 | 0.55-0.70 |
| BERT | 0.78-0.88 | 0.82-0.92 | 0.08-0.18 | 0.60-0.75 |
| RoBERTa | 0.80-0.90 | 0.84-0.94 | 0.06-0.16 | 0.65-0.80 |
| LLM Few-shot | 0.70-0.80 | 0.75-0.85 | 0.15-0.25 | 0.45-0.60 |

*Note: Results vary based on dataset, hyperparameters, seed, and language. Chinese text may have different performance characteristics.*

## Notes for Research Paper

### Methodological Considerations (Reproducibility)

1. **Validation Strategy**
   - K-fold stratified cross-validation (default: 10 folds)
   - Ensures balanced label distribution in each fold
   - Prevents data leakage between train/test

2. **Multilabel Problem Approach**
   - Per-label binary classification (One-vs-Rest)
   - SMOTE applied per-label on training fold only
   - Metrics: F1-macro, F1-micro, Hamming Loss, Subset Accuracy

3. **Data Augmentation**
   - SMOTE-based balancing on training data only
   - No augmentation on test/validation data
   - Prevents artificial performance inflation

4. **Model Reproducibility**
   - Fixed random seed (default: 42)
   - Deterministic hyperparameters
   - LLM temperature=0.0 for determinism
   - All model weights and configs saved

5. **Performance Metrics**
   - **Reported**: Mean ± std across all folds
   - **Statistical**: Min/max to show range
   - **Per-label**: Individual performance per class

### Recommendations for Paper Structure

**Section 1: Dataset & Methodology**
- Use `dataset_report.json` for dataset statistics
- Use `model_configurations.json` for reproducibility details
- Include SMOTE impact visualization

**Section 2: Experimental Setup**
- Table: Model architectures and hyperparameters
- Include: Training time, inference speed
- Reference: k-fold cross-validation setup

**Section 3: Results**
- Main results table: All models × key metrics
- Use LaTeX export: `comprehensive_model_comparison.tex`
- Include per-label confusion matrices

**Section 4: Analysis**
- Discussion of per-label performance
- Multilabel metrics interpretation
- Model comparison ranking

**Appendix**
- Per-model detailed metrics (per-label breakdown)
- Training history curves (DL models)
- Error analysis and edge cases
- Use `research_paper_appendix.json`

### Performance Analysis Tips

1. **Model Selection** - Compare by:
   - F1-Macro for balanced evaluation
   - F1-Micro for sample-weighted importance
   - Subset Accuracy for exact match rate
   - Training time vs accuracy trade-off

2. **Label-Specific Insights**
   - Review confusion matrices per label
   - Identify which models excel at which labels
   - Check per-label_metrics_report.json

3. **Error Analysis**
   - Look at samples with high disagreement
   - Identify label combinations models struggle with
   - Use multilabel_metrics_report.json

## Troubleshooting

### Training Issues

**Issue: Out of Memory (OOM)**
- Solution: Reduce `batch_size` in config
- For BERT: Try batch_size=8 or 16
- For LSTM: Try batch_size=16 or 32

**Issue: LLM API timeout**
- Solution: Check Groq API key validity
- Verify network connection
- Increase timeout in `src/models/models_llm.py`

**Issue: Missing training_history.json**
- Cause: DL model training didn't complete
- Solution: Ensure `n_epochs > 0` and training succeeded
- Check model_artifacts/{model}/fold_1/metadata.json

### Visualization Issues

**Issue: Visualizations not generated**
- Solution: Check `results/modular_multimodel/model_artifacts/` exists
- Verify fold directories have prediction files (predictions*.npy)
- Run `python scripts/research_comparison.py` after training

**Issue: Per-label confusion matrices blank**
- Cause: Missing predictions or labels
- Solution: Ensure training completed successfully
- Check that y_test files exist in model artifact directories

## FAQ

**Q: How long does the full pipeline take?**
A: ~2 hours with GPU, 4-6 hours on CPU (for 10 folds, all models)

**Q: Can I train on CPU only?**
A: Yes, but much slower. ML models and LLM are fast on CPU. DL models (CNN, LSTM) will be slow.

**Q: How do I add a new model?**
A: Add training function to `src/models/`, call from `train.py`, it will auto-generate reports.

**Q: Can I use my own dataset?**
A: Yes, format as CSV with 'text' column and label columns, update `src/training/config.py` LABEL_COLUMNS

**Q: How do I customize visualizations?**
A: Edit functions in `scripts/research_modules/visualizations.py`, modify colors, titles, metrics shown

**Q: Why are LLM results different each run?**
A: Even with temperature=0.0, LLM may have slight variations. Use seed for best reproducibility.

## Citation

If you use this research pipeline, consider citing:

```bibtex
@article{multilabel_classification_2024,
  title={Comprehensive Multilabel Text Classification: Comparing Machine Learning, 
         Deep Learning, and Large Language Models},
  author={Your Name},
  journal={Journal/Conference},
  year={2024},
  note={Implemented using Classification-Module-Review\_Graph pipeline}
}
```

## References

- **Multilabel Metrics**: Zhang & Zhou (2014) "A Review on Multi-Label Learning Algorithms"
- **SMOTE**: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- **Cross-validation**: Kohavi (1995) "A Study of Cross-Validation and Bootstrap for Accuracy Estimation"
- **BERT**: Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
- **LLM Prompting**: Brown et al. (2020) "Language Models are Few-Shot Learners"

## Support & Feedback

For issues, questions, or suggestions:
1. Check documentation in `docs/` directory
2. Review examples in this README
3. Check `PROJECT_VERIFICATION.md` for project status
4. Refer to individual module docstrings in source code