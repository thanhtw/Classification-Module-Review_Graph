# Research Comparison Pipeline (Current)

This document reflects the current behavior of scripts/research_comparison.py and scripts/train.py.

## Goal

Run a complete best-fold model comparison pipeline for multilabel classification with three labels:
- relevance
- concreteness
- constructive

## Models in Current Comparison Run

scripts/research_comparison.py compares these models:
- Linear SVM (linear_svm)
- Logistic Regression (logistic_regression)
- Naive Bayes (naive_bayes)
- LSTM (lstm)
- BiLSTM (bilstm)
- BERT (bert)
- RoBERTa (roberta)
- llama-3.1-8b-instant (LLM, Zero-shot) (llm_zero_shot)
- llama-3.1-8b-instant (LLM, Few-shot k=100) (llm_few_shot)

## How the Pipeline Works

1. For each model, research_comparison.py calls scripts/train.py.
2. It loads results/modular_multimodel/model_results_detailed.csv.
3. It selects one best fold per model by:
   - f1_macro (desc)
   - f1_micro (desc)
   - subset_accuracy (desc)
4. It builds a best-fold summary row per model.
5. It exports best-fold train/test split CSVs for feature analysis.
6. It generates comparison tables, reports, and visualizations.

## Run Commands

Quick run:

```bash
python scripts/research_comparison.py --n_folds 10 --seed 42
```

Single-fold run:

```bash
python scripts/research_comparison.py --n_folds 1 --seed 42
```

## Current Visualization Outputs

Generated in results/research_comparison:
- smote_impact_visualization.png
- model_f1_comparison.png
- model_multilabel_metrics.png
- confusion_matrices_all_models.png
- per_label_confusion_matrices/confusion_matrix_3labels_{model}_{fold}.png

Notes:
- Per-label confusion output is one image per model/fold with three 2x2 panels (one panel per label).
- No training_curves.png generation in current research pipeline.
- No comprehensive_metrics_heatmap.png generation in current research pipeline.

## Current Data and Report Outputs

Core comparison files in results/research_comparison:
- all_models_comparison.csv
- best_fold_model_comparison.csv
- all_models_comparison_report.txt
- all_models_all_metrics.csv
- all_models_complete_data.csv
- key_models_metrics_table.csv
- comprehensive_model_comparison.tex
- metrics_summary_statistics.txt

Metrics reports:
- per_label_metrics_report.json
- multilabel_metrics_report.json

Other reports:
- dataset_report.json
- model_configurations.json
- training_process_report.json
- research_paper_appendix.json

Per-model metrics text reports:
- per_model_metrics/*_metrics.txt

Best-fold feature split exports:
- best_fold_feature_analysis/best_fold_split_summary.csv
- best_fold_feature_analysis/best_fold_split_summary.txt
- best_fold_feature_analysis/{model_key}_fold_{n}/train_split.csv
- best_fold_feature_analysis/{model_key}_fold_{n}/test_split.csv

## Inputs Used by Research Comparison

Primary input artifacts:
- results/modular_multimodel/model_results_detailed.csv
- results/modular_multimodel/model_artifacts/{model}/fold_{k}/predictions.npy
- results/modular_multimodel/model_artifacts/{model}/fold_{k}/labels.npy
- results/modular_multimodel/model_artifacts/{model}/fold_{k}/metadata.json (thresholds if present)
- results/modular_multimodel/model_artifacts/{model}/fold_{k}/training_history.json (threshold fallback for LSTM/BiLSTM)
- results/modular_multimodel/global_train_data_analysis/train_smote_analysis_summary.json

Best-fold mapping priority used by visualization helpers:
1. results/research_comparison/best_fold_model_comparison.csv
2. results/research_comparison/all_models_comparison.csv
3. results/modular_multimodel/best_fold_per_model.csv

## Training-Side Notes

scripts/train.py current runner supports:
- bert, roberta
- linear_svm, naive_bayes, logistic_regression
- lstm, bilstm
- llm_zero_shot, llm_few_shot

SMOTE behavior in current train runner:
- applied to training split only
- enabled for ML models only (linear_svm, naive_bayes, logistic_regression)

## Troubleshooting

If a figure is missing:
1. Check that model artifacts exist under results/modular_multimodel/model_artifacts.
2. Check predictions.npy and labels.npy for each selected fold.
3. Re-run comparison:

```bash
python scripts/research_comparison.py --n_folds 10 --seed 42
```

If LLM models fail:
1. Verify Groq API configuration.
2. Retry with fewer folds to validate setup.

## Practical Recommendation

For publication tables/figures, prefer n_folds=10 and keep seed fixed for reproducibility.
