# Research Paper Comparison: LLM Few-shot vs Fine-tuned Transformers

## Executive Summary

This document describes the research comparison between:
1. **LLM Approach**: qwen/qwen3-32b via Groq API (Few-shot learning)
2. **Traditional Approach**: BERT and RoBERTa (Fine-tuned transformers)

**Research Question**: How does few-shot LLM inference compare to fine-tuned transformers for multilabel text classification?

---

## Hypothesis

We hypothesize that **few-shot LLM inference** with qwen/qwen3-32b will:
- Achieve **competitive or superior performance** compared to fine-tuned BERT/RoBERTa
- Require **no training/fine-tuning**, reducing computational costs
- **Demonstrate generalization** through in-context learning

---

## Experimental Setup

### Task Definition
- **Type**: Multilabel text classification
- **Labels**: 3 binary labels
  - Relevance: Is the text relevant to the topic?
  - Concreteness: Is the text concrete/specific?
  - Constructive: Is the feedback constructive?
- **Dataset**: 3-label annotated multilingual text corpus

### Models Under Comparison

#### 1. qwen/qwen3-32b (LLM, Few-shot)
- **Provider**: Groq API (cloud-based)
- **Approach**: Zero-shot and Few-shot prompting (3-shot)
- **Training**: None (pre-trained only)
- **Parameters**: 32B
- **Inference**: API calls via OpenAI-compatible interface
- **Cost**: Per-token pricing

**Few-shot Template**:
```
You are a strict multi-label classifier.
Predict 3 binary labels for the input text and return ONLY valid JSON.
Keys must be: relevance, concreteness, constructive.
Values must be integers 0 or 1.

Here are 3 labeled examples:
[Example 1]
[Example 2]
[Example 3]

Text: [Test sample]
Answer JSON:
```

#### 2. BERT (Fine-tuned)
- **Model**: bert-base-multilingual-cased (110M parameters)
- **Approach**: Fine-tuned on labeled training data
- **Training**: 5 epochs, Adam optimizer, CE loss
- **Batch Size**: 16
- **Hyperparameters**: Learning rate 2e-5, weight decay 1e-3

#### 3. RoBERTa (Fine-tuned)
- **Model**: roberta-base (125M parameters)
- **Approach**: Fine-tuned on labeled training data
- **Training**: 5 epochs, Adam optimizer, CE loss
- **Batch Size**: 16
- **Hyperparameters**: Learning rate 2e-5, weight decay 1e-3

---

## Methodology

### Cross-Validation
- **Type**: 10-fold cross-validation
- **Stratification**: Stratified by label distribution
- **Reproducibility**: Fixed seed (42)

### Data Preprocessing
- Train/test split (per-fold): 80/20
- SMOTE applied to training set (for imbalanced classes)
- Text normalization: lowercase, remove special chars (varies by model)

### Evaluation Metrics
For each fold, compute:

1. **Per-Label Metrics** (for each of 3 labels):
   - Precision (True Positives / (True Positives + False Positives))
   - Recall (True Positives / (True Positives + False Negatives))
   - F1-score (harmonic mean of precision and recall)

2. **Macro-Averaged Metrics** (average across 3 labels):
   - Precision-macro
   - Recall-macro
   - F1-macro

3. **Micro-Averaged Metrics** (global True Positives, False Positives...):
   - Precision-micro
   - Recall-micro
   - F1-micro

4. **Multilabel-Specific Metrics**:
   - Hamming Loss (fraction of incorrect labels)
   - Subset Accuracy (label set exactly matches)

### Inference Time & Cost Analysis
- **Inference Time**: Time per sample (seconds)
- **Total Inference Time**: Time for complete test set
- **API Cost**: Tokens × price per 1M tokens (if using Groq API)
- **Computational Cost**: Hardware cost (GPU hours for BERT/RoBERTa)

---

## Research Questions to Answer

1. **Performance**: Which approach achieves highest F1-macro?
2. **Efficiency**: What is the inference time trade-off?
3. **Cost**: Which approach is most cost-effective?
4. **Generalization**: How stable are results across folds?
5. **Consistency**: Which approach has lower variance across folds?

---

## Expected Results Format

```
Model                    | F1-Macro ± Std | Precision-Macro | Recall-Macro | Inference Time
-----                    | -------------- | --------------- | ------------ | --------------
qwen/qwen3-32b (few-shot)| 0.7234 ± 0.042| 0.7156          | 0.7315       | 2.34s
BERT (fine-tuned)        | 0.6892 ± 0.051| 0.7123          | 0.6654       | 0.12s
RoBERTa (fine-tuned)     | 0.7041 ± 0.038| 0.7234          | 0.6851       | 0.14s
```

---

## Running the Comparison

### Quick Test (2 folds)
```bash
export GROQ_API_KEY="your-api-key-here"
python scripts/research_comparison.py --n_folds 2 --seed 42
```

### Full Research (10 folds)
```bash
export GROQ_API_KEY="your-api-key-here"
python scripts/research_comparison.py --n_folds 10 --seed 42
```

### Output
- `results/research_comparison/model_comparison.csv` - Summary table
- `results/research_comparison/comparison_detailed.json` - Detailed metrics per fold
- Per-fold results in `results/modular_multimodel/model_artifacts/`

---

## Key Advantages of This Comparison for Publication

1. **Novelty**: Few-shot LLM approach vs traditional fine-tuned transformers
2. **Reproducibility**: Fixed seeds, open-source datasets, cloud API
3. **Rigor**: 10-fold CV with stratification and SMOTE
4. **Practical Impact**: Cost-benefit analysis for practitioners
5. **Multilingual**: Tests on multilingual text (BERT-multilingual)
6. **Multilabel**: Challenging problem (not binary classification)

---

## Conclusion Framework

This comparison will demonstrate whether modern LLMs can compete with fine-tuned transformers without requiring:
- Expensive fine-tuning procedures
- GPU resources
- Model downloads and local deployment

**Target Journals**: ACL, EMNLP, COLING, IEEE Transactions on Neural Networks and Learning Systems
