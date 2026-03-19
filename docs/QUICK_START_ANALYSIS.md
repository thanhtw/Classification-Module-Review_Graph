# Quick Start: Advanced Analysis

Complete workflow to train and analyze all models with per-label metrics, error analysis, and reproducibility tracking.

## TL;DR - 3 Commands

```bash
# 1. Train all models (10-fold cross-validation)
conda run -n ThomasAgent python run_modular_multimodel_train.py \
  --models bert roberta svm decision_tree cnn_attention lstm bilstm llm_zero_shot llm_few_shot

# 2. Generate comprehensive analysis
conda run -n ThomasAgent python function/comprehensive_analysis.py

# 3. View results
cat results/modular_multimodel/comprehensive_analysis/ANALYSIS_SUMMARY.txt
```

## Files Generated

After running analysis, you'll get:

```
results/modular_multimodel/comprehensive_analysis/
├── ANALYSIS_SUMMARY.txt
├── all_folds_summary.csv           ← All results
├── best_folds_comparison.csv       ← Best model per fold
├── bert/
│   └── model_analysis.json         ← BERT statistics
├── roberta/
│   └── model_analysis.json
└── (other models...)
```

Also in `results/modular_multimodel/model_artifacts/`:
```
model_artifacts/
├── bert/fold_1/
│   ├── per_label_report_bert_fold1.json     ← F1 per label
│   ├── error_analysis_bert_fold1.json       ← Misclassifications
│   ├── reproducibility_manifest_fold1.json  ← Hardware/versions
│   └── confusion_matrices_bert_fold1.png    ← Visual CM
├── bert/fold_2/
│   └── ...
└── (all models and folds)
```

## Key Metrics to Check

### 1. Per-Label Performance
Find: `per_label_report_*.json`

Shows F1, precision, recall **for each label separately**:
- Reveals which labels are harder
- Shows class imbalance impact
- Identifies label-specific weaknesses

### 2. Confusion Matrices
Find: `confusion_matrices_*.png`

Visual summary of:
- False Positives ← Not constructive but model thinks it is
- False Negatives ← Constructive but model misses it
- True Positives / True Negatives

### 3. Error Analysis
Find: `error_analysis_*.json`

Lists actual misclassified comments:
- What types of comments does BERT miss?
- Are errors systematic? (e.g., always misses sarcasm)
- Which label causes most errors?

### 4. Reproducibility Manifest
Find: `reproducibility_manifest_*.json`

Contains:
- Exact model checkpoint (e.g., "bert-base-chinese")
- Python/package versions
- GPU used
- Training time
- Hyperparameters

## Interpreting Results

### Best Model for Your System
1. Check `best_folds_comparison.csv`
2. Look at `f1_macro` (overall) or check per-label if one label matters more
3. Check `infer_time` if latency is critical
4. Review error patterns in JSON files

### Why F1-Macro Matters
- Treats all labels equally
- Good metric when you care about overall performance
- Not ideal for severe class imbalance (constructiveness: 9.79%)

### Handling Class Imbalance
If constructiveness F1 is low:
1. Check `positive_rate` in per_label_report
2. Many false negatives? Model is ignoring class
3. High precision, low recall? Model is too conservative
4. Consider adjusting decision threshold or sampling strategy

## Example: Understanding BERT Performance

```bash
# 1. Check overall best fold for BERT
grep "bert" results/modular_multimodel/comprehensive_analysis/best_folds_comparison.csv

# 2. Read per-label breakdown (best fold)
cat results/modular_multimodel/model_artifacts/bert/fold_X/per_label_report_bert_foldX.json

# 3. See misclassified examples
cat results/modular_multimodel/model_artifacts/bert/fold_X/error_analysis_bert_foldX.json | 
  python -m json.tool | head -100

# 4. Check exact versions used
cat results/modular_multimodel/model_artifacts/bert/fold_X/reproducibility_manifest_foldX.json
```

## Common Questions

### Q: Why is constructiveness F1 so low?
A: Check:
1. `positive_rate` in per_label_report - If ~10%, class imbalance is the issue
2. False negatives? Increase SMOTE or weight this label higher
3. False positives? Model is too confident, lower threshold

### Q: Should I use BERT or RoBERTa?
A: Compare in `best_folds_comparison.csv`:
- Look at f1_macro (overall) AND per-label metrics
- Check inference time - trade-off speed vs accuracy
- Review error patterns - different errors might favor one

### Q: Can I reproduce results?
A: Yes! Check `reproducibility_manifest_*.json`:
- Use exact model checkpoint version
- Install exact package versions
- Same seed for determinism
- GPU type matters for some randomness

### Q: How do I deploy the best model?
A: From analysis:
1. Identify best fold (check CSV)
2. Load model from `results/modular_multimodel/model_artifacts/<model>/<fold>/`
3. Use metadata for inference settings
4. Check error analysis for known failure cases

## Troubleshooting

### Error: "No such file" when running analysis
Make sure you:
1. Ran training first: `python run_modular_multimodel_train.py`
2. Training completed successfully (check terminal for errors)
3. Using same `results_dir` path

### Metrics look wrong
1. Verify training completed (check log output)
2. Ensure predictions/labels are in correct format
3. Check for NaN values in metrics

### Need to re-run analysis only
```bash
# Doesn't require re-training
conda run -n ThomasAgent python function/comprehensive_analysis.py
```

## Advanced Usage

### Compare specific models
```bash
# Only train BERT vs LLM
python run_modular_multimodel_train.py --models bert llm_zero_shot

# Analyze
python function/comprehensive_analysis.py
```

### Different Llama checkpoint
```bash
python run_modular_multimodel_train.py \
  --models llm_zero_shot \
  --llm_model_name "meta-llama/Llama-3-8B-Instruct"
```

### Fewer folds for quick test
```bash
python run_modular_multimodel_train.py \
  --models bert roberta \
  --n_folds 3
```

## Next Steps

1. **Run training** with all models
2. **Generate analysis** to identify best model
3. **Deep dive** into per-label and error analysis
4. **Select model** based on your specific needs
5. **Deploy** with confidence (reproducibility tracked)

See `ADVANCED_ANALYSIS_GUIDE.md` for detailed documentation.
