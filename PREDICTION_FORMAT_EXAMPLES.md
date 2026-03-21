#!/usr/bin/env python3
"""
PREDICTION FORMAT EXAMPLES FOR ALL MODELS

This document shows exactly how each model generates predictions,
the intermediate formats, and the final binary output format.

All models ultimately output the same format:
- Shape: (n_samples, 3) - n_samples test samples, 3 binary labels
- Dtype: int64 (values are 0 or 1)
- Columns: [relevance, concreteness, constructive]
"""

# ============================================================================
# OVERVIEW TABLE
# ============================================================================
"""
┌─────────────────────┬──────────────┬────────────────────────┬──────────────┐
│ Model Type          │ Framework    │ Intermediate Format    │ Final Output │
├─────────────────────┼──────────────┼────────────────────────┼──────────────┤
│ LinearSVM           │ scikit-learn │ Binary directly        │ (n, 3) int64 │
│ NaiveBayes          │ scikit-learn │ Binary directly        │ (n, 3) int64 │
│ LogisticRegression  │ scikit-learn │ Binary directly        │ (n, 3) int64 │
│ BERT                │ Transformers │ Logits → Sigmoid → 0.5 │ (n, 3) int64 │
│ RoBERTa             │ Transformers │ Logits → Sigmoid → 0.5 │ (n, 3) int64 │
│ LSTM                │ PyTorch      │ Logits → Sigmoid → 0.5 │ (n, 3) int64 │
│ BiLSTM              │ PyTorch      │ Logits → Sigmoid → 0.5 │ (n, 3) int64 │
│ CNN Attention       │ PyTorch      │ Logits → Sigmoid → 0.5 │ (n, 3) int64 │
│ LLM Zero-Shot       │ Groq API     │ JSON Parse             │ (n, 3) int64 │
│ LLM Few-Shot        │ Groq API     │ JSON Parse             │ (n, 3) int64 │
└─────────────────────┴──────────────┴────────────────────────┴──────────────┘
"""

# ============================================================================
# 1. MACHINE LEARNING MODELS (LinearSVM, NaiveBayes, LogisticRegression)
# ============================================================================
"""
These models use scikit-learn's OneVsRestClassifier with Word2Vec embeddings.

Pipeline:
1. Text → Word2Vec embedding (300-dim dense vector)
2. OneVsRestClassifier trains 3 binary classifiers (one per label)
3. For each classifier: predict() returns 0 or 1 directly

Code from models_ml.py:
─────────────────────────────────────────────────────────────────────────────
    # Step 1: Vectorization
    x_train = vectorizer.fit(train_texts).transform(train_texts)  # (n, 300)
    x_test = vectorizer.transform(test_texts)                    # (n_test, 300)
    
    # Step 2: Train OneVsRestClassifier
    model = OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced", ...))
    model.fit(x_train, y_train)
    
    # Step 3: Get binary predictions
    preds = model.predict(x_test)  # Shape: (n_test, 3), dtype: int64, values: 0 or 1
─────────────────────────────────────────────────────────────────────────────

Input to model.predict():
  x_test shape: (240, 300)
  Example first sample: [0.15, -0.32, 0.78, ..., 0.02]  (300-dim vector)

Output from model.predict():
  preds shape: (240, 3)
  
  Example predictions (first 10 samples):
  [[0, 1, 0],    # Sample 1: not relevant, concrete, not constructive
   [1, 1, 1],    # Sample 2: relevant, concrete, constructive
   [1, 0, 1],    # Sample 3: relevant, NOT concrete, constructive
   [0, 0, 0],    # Sample 4: not relevant, NOT concrete, not constructive
   [1, 1, 0],    # Sample 5: relevant, concrete, not constructive
   [0, 1, 1],    # Sample 6: not relevant, concrete, constructive
   [1, 0, 0],    # Sample 7: relevant, NOT concrete, not constructive
   [1, 1, 1],    # Sample 8: relevant, concrete, constructive
   [0, 0, 1],    # Sample 9: not relevant, NOT concrete, constructive
   [1, 0, 1]]    # Sample 10: relevant, NOT concrete, constructive

Format:
  - dtype: int64 (native binary values)
  - shape: (240, 3)
  - values: Only 0 or 1
  - no threshold needed (already binary)
"""

# ============================================================================
# 2. TRANSFORMER MODELS (BERT, RoBERTa)
# ============================================================================
"""
These models use HuggingFace Transformers with sigmoid + threshold.

Pipeline:
1. Text → Tokenize and encode (BERT tokenizer)
2. Pass through BERT/RoBERTa encoder
3. Get logits (unbounded real values)
4. Apply sigmoid activation: sigmoid(logits) = 1 / (1 + exp(-logits))
5. Threshold at 0.5: probabilities > 0.5 become 1, else 0

Code from models_transformers.py:
─────────────────────────────────────────────────────────────────────────────
    # Step 1-2: Encode and get logits
    pred_out = trainer.predict(ds_test)
    logits = _extract_logits(pred_out)  # Shape: (n_test, 3), unbounded values
    
    # Step 3: Sigmoid + threshold
    preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)  # (n_test, 3)
─────────────────────────────────────────────────────────────────────────────

Raw logits from model (before sigmoid):
  [[2.15, -1.32, 0.78],      # Sample 1: high confidence 0, low confidence 1, medium confidence 2
   [-0.45, 1.89, 2.41],      # Sample 2
   [1.52, -2.13, 0.34],      # Sample 3
   ...]

After sigmoid activation (probabilities between 0-1):
  [[0.896, 0.210, 0.685],    # Sample 1
   [0.389, 0.869, 0.918],    # Sample 2
   [0.820, 0.106, 0.584],    # Sample 3
   ...]

After threshold (> 0.5):
  [[1, 0, 1],       # Sample 1: 0.896 > 0.5 → 1, 0.210 < 0.5 → 0, 0.685 > 0.5 → 1
   [0, 1, 1],       # Sample 2: 0.389 < 0.5 → 0, 0.869 > 0.5 → 1, 0.918 > 0.5 → 1
   [1, 0, 1],       # Sample 3: 0.820 > 0.5 → 1, 0.106 < 0.5 → 0, 0.584 > 0.5 → 1
   ...]

Final format (same as ML models):
  - dtype: int64
  - shape: (240, 3)
  - values: Only 0 or 1
"""

# ============================================================================
# 3. NEURAL NETWORK MODELS (LSTM, BiLSTM, CNN Attention)
# ============================================================================
"""
These models use PyTorch with sigmoid activation + threshold.

Pipeline:
1. Text → Tokenize and convert to sequences
2. Pass through embedding layer (300-dim)
3. Pass through LSTM/BiLSTM or CNN
4. Apply dense layer to get logits
5. Apply sigmoid activation: sigmoid(logits) = 1 / (1 + exp(-logits))
6. Threshold at 0.5: probabilities > 0.5 become 1, else 0

Code from models_nn.py (_train_eval function):
─────────────────────────────────────────────────────────────────────────────
    # During inference loop
    all_logits = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            all_logits.append(model(bx).cpu().numpy())  # Raw logits from model
    
    logits = np.vstack(all_logits)  # Shape: (n_test, 3), dtype: float32
    
    # Apply sigmoid + threshold
    preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)  # Shape: (n_test, 3)
─────────────────────────────────────────────────────────────────────────────

Logits from PyTorch model (after Dense layer, before sigmoid):
  Text: "这个产品很相关且具体的评论"
  Model output: [1.82, -0.56, 1.34]
  
  Another text: "无关且模糊的评论"
  Model output: [-2.15, -1.89, -0.72]

Sigmoid transformation:
  [1.82, -0.56, 1.34] → [0.860, 0.363, 0.793]
  [-2.15, -1.89, -0.72] → [0.104, 0.130, 0.327]

After threshold (> 0.5):
  [1, 0, 1]   # First text
  [0, 0, 0]   # Second text

For 240 test samples (example):
  [[1, 0, 1],
   [0, 0, 0],
   [1, 1, 1],
   [0, 1, 0],
   ...]

Final format:
  - dtype: int64
  - shape: (240, 3)
  - values: Only 0 or 1
"""

# ============================================================================
# 4. LANGUAGE MODEL (LLM Zero-Shot & Few-Shot)
# ============================================================================
"""
These models use Groq API with JSON parsing.

Pipeline:
1. Format prompt with text and optional few-shot examples
2. Call Groq API (claude or llama model)
3. Get text response with JSON
4. Extract and parse JSON
5. Convert values to binary (0 or 1)

Code from models_llm.py:
─────────────────────────────────────────────────────────────────────────────
    # Build prompt
    prompt = _build_prompt(
        text=test_text,
        mode="zero_shot",  # or "few_shot"
        few_shot_examples=[]
    )
    
    # Call API
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64
    )
    generated = completion.choices[0].message.content
    
    # Parse prediction
    pred, ok = _parse_prediction(generated)  # Returns (np.array([0,1,0], dtype=int), bool)
─────────────────────────────────────────────────────────────────────────────

Example LLM API Response:
  Raw text from API:
  \"\"\"
  {"relevance": 1, "concreteness": 1, "constructive": 0}
  \"\"\"

Parsing (_parse_prediction function):
  1. Extract JSON block: {"relevance": 1, "concreteness": 1, "constructive": 0}
  2. Load JSON: {"relevance": 1, "concreteness": 1, "constructive": 0}
  3. Convert each to binary: _safe_int01(value)
     - Result: [1, 1, 0]
  4. Return as numpy array: np.array([1, 1, 0], dtype=int)

For 240 test samples:
  Sample 1 - LLM generates: {"relevance": 1, "concreteness": 0, "constructive": 1}  → [1, 0, 1]
  Sample 2 - LLM generates: {"relevance": 0, "concreteness": 1, "constructive": 1}  → [0, 1, 1]
  Sample 3 - LLM generates: {"relevance": 1, "concreteness": 1, "constructive": 0}  → [1, 1, 0]
  ...
  
  Final array (240, 3):
  [[1, 0, 1],
   [0, 1, 1],
   [1, 1, 0],
   ...]

Special handling:
  - Invalid JSON → [0, 0, 0] (default)
  - Non-integer values → _safe_int01() converts (0 or 1)
  - Missing keys → 0 (default)

Final format:
  - dtype: int64
  - shape: (240, 3)
  - values: Only 0 or 1 (after parsing and conversion)
"""

# ============================================================================
# UNIFIED SAVED FORMAT (ALL MODELS)
# ============================================================================
"""
After prediction, all models save in identical format:

File: results/modular_multimodel/model_artifacts/{model_name}/fold_1/predictions.npy

Shape: (240, 3)
Dtype: int64
Values: 0 or 1

Example content (first 20 test samples):
─────────────────────────────────────────────────────────────────────────────
[[0, 1, 0],    # Sample 1
 [1, 1, 1],    # Sample 2
 [1, 0, 1],    # Sample 3
 [0, 0, 0],    # Sample 4
 [1, 1, 0],    # Sample 5
 [0, 1, 1],    # Sample 6
 [1, 0, 0],    # Sample 7
 [1, 1, 1],    # Sample 8
 [0, 0, 1],    # Sample 9
 [1, 0, 1],    # Sample 10
 [0, 1, 0],    # Sample 11
 [1, 0, 1],    # Sample 12
 [1, 1, 0],    # Sample 13
 [0, 1, 1],    # Sample 14
 [1, 1, 1],    # Sample 15
 [0, 0, 0],    # Sample 16
 [1, 0, 1],    # Sample 17
 [0, 1, 0],    # Sample 18
 [1, 1, 1],    # Sample 19
 [1, 0, 1]]    # Sample 20

Column meanings:
  Column 0: Relevance binary prediction
  Column 1: Concreteness binary prediction
  Column 2: Constructive binary prediction

Loading in Python:
  import numpy as np
  preds = np.load('results/modular_multimodel/model_artifacts/bert/fold_1/predictions.npy')
  print(preds.shape)        # (240, 3)
  print(preds.dtype)        # int64
  print(preds[0])           # [0 1 0]
  print(preds[:5, 0])       # First 5 'relevance' predictions: [0 1 1 0 1]
"""

# ============================================================================
# COMPARISON BY MODEL
# ============================================================================
"""
Raw vs Final Output Comparison:

LinearSVM (scikit-learn):
  Raw output:    [0, 1, 0] ← Already binary from classifier
  Thresholding:  None needed
  Final output:  [0, 1, 0] ✓

BERT/RoBERTa (HuggingFace):
  Raw output:    [2.15, -1.32, 0.78] ← Unbounded logits
  After sigmoid: [0.896, 0.210, 0.685] ← Probabilities
  Thresholding:  > 0.5 → [1, 0, 1] ✓
  Final output:  [1, 0, 1] ✓

LSTM/BiLSTM/CNN (PyTorch):
  Raw output:    [1.82, -0.56, 1.34] ← Unbounded logits
  After sigmoid: [0.860, 0.363, 0.793] ← Probabilities
  Thresholding:  > 0.5 → [1, 0, 1] ✓
  Final output:  [1, 0, 1] ✓

LLM (Groq API):
  Raw output:    '{"relevance": 1, "concreteness": 0, "constructive": 1}'
  After parsing: [1, 0, 1] ✓
  Final output:  [1, 0, 1] ✓
"""

# ============================================================================
# KEY POINTS
# ============================================================================
"""
1. ALL models output IDENTICAL format when saved:
   - Shape: (n_samples, 3)
   - Dtype: int64
   - Values: 0 or 1 only
   - Columns: [relevance, concreteness, constructive]

2. Different intermediate processing:
   - ML models: Direct binary from OneVsRest classifiers
   - Transformers: Logits → Sigmoid → Threshold at 0.5
   - Neural networks: Logits → Sigmoid → Threshold at 0.5
   - LLM: API response → JSON parse → Binary conversion

3. This unified format enables:
   - Confusion matrix generation (same code for all models)
   - Metric computation (same code for all models)
   - Cross-model comparison (directly comparable)

4. Standard multilabel evaluation applies:
   - Compute metrics per label independently
   - Macro/micro averaging for overall scores
   - Row-wise normalization for confusion matrices
"""

# ============================================================================
# PRACTICAL EXAMPLE: Loading and Using Predictions
# ============================================================================
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Load predictions for one model and one fold
model_name = 'bert'
fold_num = 1
artifacts_path = Path('results/modular_multimodel/model_artifacts')

preds = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'predictions.npy')
labels = np.load(artifacts_path / model_name / f'fold_{fold_num}' / 'labels.npy')

print(f"Shape: {preds.shape}")        # (240, 3)
print(f"Dtype: {preds.dtype}")        # int64
print(f"Unique values: {np.unique(preds)}")  # [0 1]

# Generate confusion matrix for 'relevance' label (column 0)
cm = confusion_matrix(labels[:, 0], preds[:, 0])
print("\\nConfusion matrix for 'relevance':")
print(cm)
#     pred_0  pred_1
# true_0  [TN     FP]
# true_1  [FN     TP]

# Compute accuracy per label
accuracy_per_label = []
for label_idx in range(3):
    acc = np.mean(preds[:, label_idx] == labels[:, label_idx])
    accuracy_per_label.append(acc)

print(f"\\nAccuracy per label: {accuracy_per_label}")
# [0.85, 0.90, 0.78]  ← Relevance, Concreteness, Constructive

# Aggregate across all folds
all_preds = np.load(artifacts_path / model_name / 'all_folds_predictions.npy')
all_labels = np.load(artifacts_path / model_name / 'all_folds_labels.npy')

print(f"\\nAggregated shape: {all_preds.shape}")  # (2398, 3)
print(f"Total samples: {all_preds.shape[0]}")     # 2398
"""

# ============================================================================
