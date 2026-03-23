# Classification-Module-Review_Graph

A multilabel text classification project for comparing classical ML, deep learning, transformer, and LLM-based methods on three labels:
- relevance
- concreteness
- constructive

## Current Pipeline Scope

Current train and comparison scripts actively support these 9 models:
- linear_svm
- logistic_regression
- naive_bayes
- lstm
- bilstm
- bert
- roberta
- llm_zero_shot
- llm_few_shot

Notes:
- CNN-related paths are not part of the current active training/comparison flow.
- Research comparison now uses best-fold-per-model selection for ranking and reporting.
- Training curves and comprehensive metrics heatmap are not generated in the current research output set.

## Project Layout

- src/: model code, data processing, utilities, and training config
- scripts/train.py: modular training runner
- scripts/research_comparison.py: end-to-end research comparison pipeline
- scripts/research_modules/: reporting, metrics, tables, and visualization modules
- data/cleaned_3label_data.csv: main dataset
- results/: generated outputs

## Setup

```bash
pip install -r requirements.txt
```

Optional (conda):

```bash
conda create -n classification python=3.10
conda activate classification
pip install -r requirements.txt
```

## Training

Run default training models:

```bash
python scripts/train.py
```

Run selected models:

```bash
python scripts/train.py --models linear_svm logistic_regression naive_bayes bert roberta
```

Run k-fold CV:

```bash
python scripts/train.py --models bert roberta lstm bilstm --n_folds 10 --seed 42
```

Use holdout mode:

```bash
python scripts/train.py --n_folds 1 --test_size 0.2
```

## Research Comparison

Run full comparison and reporting:

```bash
python scripts/research_comparison.py --n_folds 10 --seed 42
```

This will:
1. Train/evaluate models one-by-one through scripts/train.py.
2. Select the best fold per model (priority: f1_macro, then f1_micro, then subset_accuracy).
3. Export best-fold feature split files.
4. Generate publication-style figures and metric reports.

## Main Outputs

Training outputs (results/modular_multimodel):
- model_results_detailed.csv
- best_fold_per_model.csv
- model_comparison_macro_micro.csv
- model_ranking_by_macro_micro_f1.csv
- training_process.json
- training_process.jsonl
- model_artifacts/{model}/fold_{k}/...
- global_train_data_analysis/train_smote_analysis_summary.json

Research outputs (results/research_comparison):
- all_models_comparison.csv
- best_fold_model_comparison.csv
- all_models_comparison_report.txt
- model_f1_comparison.png
- model_multilabel_metrics.png
- smote_impact_visualization.png
- confusion_matrices_all_models.png
- per_label_confusion_matrices/
- all_models_all_metrics.csv
- all_models_complete_data.csv
- key_models_metrics_table.csv
- comprehensive_model_comparison.tex
- per_label_metrics_report.json
- multilabel_metrics_report.json
- model_configurations.json
- training_process_report.json
- dataset_report.json
- research_paper_appendix.json

Best-fold feature split exports:
- results/research_comparison/best_fold_feature_analysis/best_fold_split_summary.csv
- results/research_comparison/best_fold_feature_analysis/{model_key}_fold_{n}/train_split.csv
- results/research_comparison/best_fold_feature_analysis/{model_key}_fold_{n}/test_split.csv

## Important Behavioral Notes

- SMOTE is applied on training split only and only for ML models in the current runner:
  - linear_svm
  - logistic_regression
  - naive_bayes
- LLM models run via Groq API (llama-3.1-8b-instant).
- Research visualizations consume best-fold mappings from comparison artifacts when available.

## Useful Commands

List key generated figures:

```bash
ls -lh results/research_comparison/model_f1_comparison.png \
       results/research_comparison/model_multilabel_metrics.png \
       results/research_comparison/smote_impact_visualization.png
```

Open a figure on Linux:

```bash
xdg-open results/research_comparison/model_multilabel_metrics.png
```

## Documentation

See docs/ for detailed guides, especially:
- docs/RESEARCH_COMPARISON_README.md
- docs/TRAINING_PIPELINE_GUIDE.md
- docs/METRICS_REFERENCE.md
- docs/PROJECT_VERIFICATION.md
