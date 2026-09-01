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

Python 3.8 or newer is required.

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

The recommended root-level interface is:

```bash
python main.py train        # seven models, stratified 10-fold CV
python main.py compare      # all nine methods plus research reports
python main.py llm          # OpenAI zero-shot/few-shot evaluation
python main.py sensitivity  # 10-fold BERT resampling sensitivity
```

Running `python main.py` without a subcommand executes the complete nine-method
research workflow with seed 42. Seven learned models are trained and tested;
the two LLM methods perform prediction only on held-out test rows.

Execution order is fixed to keep inexpensive baselines first:

```text
Linear SVM → Logistic Regression → Naive Bayes → LSTM → BiLSTM
→ BERT → RoBERTa → LLM zero-shot → LLM few-shot
```

Transformer training prints device, batch-level progress, current loss, and
epoch-level train/validation loss so long-running folds do not appear stalled.

BERT and RoBERTa checkpoints use the persistent cache configured by
`HF_MODEL_CACHE_DIR` in `.env`. The pipeline checks for a complete local
snapshot first and downloads only when it is missing. Each fold creates a fresh
model instance from the cached files so cross-validation remains independent;
this reloads weights from disk but does not download them again.

Before LLM evaluation, add your private key to the ignored `.env` file. Use
`.env.example` as the shareable configuration template.

Completed metrics are organized under both `by_model/{model}/fold_XX/` and
`by_fold/fold_XX/{model}/`. LLM classification uses reasoning effort `none`, a
512-token initial completion budget, and one bounded retry for output-limit
errors.

Every completed model/fold also writes `test_results_with_ground_truth.csv`
inside its artifact directory. It contains the held-out text and source index,
true/predicted labels, per-label correctness, and exact-match result. LLM runs
additionally write `prediction_results_with_ground_truth.csv` with the raw API
response and parsing/API error details.

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

The publication protocol uses stratified 10-fold cross-validation. Resampling
is fitted independently inside each training fold; validation/test folds are
never resampled.

Run the BERT resampling sensitivity analysis (no resampling plus target
minority/majority ratios 0.50, 0.75, and 1.00):

```bash
python scripts/smote_sensitivity.py --models bert --seed 42
```

The analysis writes all 10 fold-level observations and mean ± standard
deviation summaries to `results/smote_sensitivity/`. For transformer inputs,
the implementation resamples complete labeled examples rather than
interpolating token IDs, which would create invalid synthetic text.

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
1. Export shared train/test rows for every fold.
2. Train/test the seven learned models and run prediction-only LLM evaluation
   on the same held-out rows.
3. Aggregate every model using cross-fold mean and sample standard deviation.
4. Select a best fold only for artifact-based figures and feature inspection.
5. Generate publication-style figures and combined nine-method reports.

When all seven learned models are complete, the pipeline runs Friedman and
paired Wilcoxon analyses for macro precision, recall, accuracy, and F1-score.
It reports Holm corrections within each metric and globally across all 84
pairwise tests, effect sizes, and paired bootstrap confidence intervals.
Outputs are stored in
`results/research_comparison/statistical_significance/`.

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
- per_label_metrics_report.txt
- all_models_per_label_all_folds.csv
- all_models_per_label_summary.csv
- all_models_per_label_report.json
- all_models_per_label_report.txt
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

- Resampling is restricted to training folds and is available for BERT,
  RoBERTa, Linear SVM, Logistic Regression, and Naive Bayes.
- LLM models run via the OpenAI API using strict JSON schema output (`gpt-5.6-luna` by default).
- LLM models do not fit parameters. Zero-shot uses only each held-out test row;
  few-shot uses training-fold examples solely as prompt context and predicts
  only held-out test rows.
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

Generate fold-level per-label CSVs for the LLM runs:

```bash
python scripts/generate_llm_per_label_report.py
```

## Documentation

See docs/ for detailed guides, especially:
- docs/RESEARCH_COMPARISON_README.md
- docs/TRAINING_PIPELINE_GUIDE.md
- docs/METRICS_REFERENCE.md
- docs/PROJECT_VERIFICATION.md
