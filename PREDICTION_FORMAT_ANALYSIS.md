# Prediction Format Analysis - All 10 Models

## Overview

All models generate **binary multilabel predictions** with the same final format:
- **Shape**: `(n_samples, 3)`
- **Data Type**: `np.int64` (binary: 0 or 1)
- **Columns**: `[relevance, concreteness, constructive]`
- **Values**: `0` (label not present) or `1` (label present)

However, the **intermediate processing pipeline differs** significantly between model types.

---

## Model Categories & Prediction Pipelines

### **CATEGORY 1: Machine Learning Models (Direct Binary Output)**

**Models**: LinearSVM, NaiveBayes, LogisticRegression
**File**: `src/models/models_ml.py`

#### Pipeline:
```
Input: Test embeddings (300-dim Word2Vec vectors)
    ↓
OneVsRestClassifier with {LinearSVC | GaussianNB | LogisticRegression}
    ↓
model.predict(x_test)  ← Direct binary prediction
    ↓
Output: Binary multilabel array (n_samples, 3)
```

#### Prediction Generation Code:
```python
# LinearSVM Example
preds = model.predict(x_test)  # (n_samples, 3)
# Values already 0 or 1
# No post-processing needed
```

#### Output Format:
```
preds shape: (240, 3)  [per-fold example]
preds dtype: int64/int
preds values: {0, 1}

Example:
[[0 1 0]
 [1 1 1]
 [1 0 1]]
```

#### Key Characteristics:
- **Direct output from sklearn classifiers**
- **No intermediate real-valued outputs**
- **Each column is a separate binary classification**
- **Very fast prediction (no sigmoid computation)**

---

### **CATEGORY 2: Transformer Models (Logits → Sigmoid → Threshold)**

**Models**: BERT, RoBERTa
**File**: `src/models/models_transformers.py`

#### Pipeline:
```
Input: Tokenized text (max_len=512)
    ↓
[BERT/RoBERTa Forward Pass]
    ↓
Raw logits (n_samples, 3)
[Real-valued scores: typically ±5 range]
    ↓
Sigmoid activation: σ(logits) = 1 / (1 + e^(-logits))
[Converts to probabilities: 0.0-1.0 range]
    ↓
Threshold comparison: prob > 0.5
    ↓
Output: Binary multilabel array (n_samples, 3)
```

#### Prediction Generation Code:
```python
# From models_transformers.py
logits = _extract_logits(pred_out)  # (n_samples, 3), float values ±5
preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)  # Sigmoid + threshold
# Result: (n_samples, 3), {0, 1}
```

#### Why This Approach:
- **Sigmoid activation**: Converts unbounded logits to probability [0, 1]
  ```
  logit = 0 → σ(0) = 0.5
  logit = 2.3 → σ(2.3) ≈ 0.91
  logit = -2.3 → σ(-2.3) ≈ 0.09
  ```
- **Threshold at 0.5**: Standard decision boundary
  ```
  prob ≥ 0.5 → predict 1 (label present)
  prob < 0.5 → predict 0 (label absent)
  ```

#### Output Format:
```
logits shape: (240, 3)
logits dtype: float64
logits values: ~[-5, 5] range (unbounded)

After sigmoid + threshold:
preds shape: (240, 3)
preds dtype: int64
preds values: {0, 1}

Example logits:
[[-0.45  2.13 -1.23]
 [ 1.89  2.45  3.12]
 [ 1.23 -0.89  1.45]]

Example predictions (after threshold):
[[0 1 0]
 [1 1 1]
 [1 0 1]]
```

#### Key Characteristics:
- **Probabilistic output (confidence scores)**
- **Threshold-dependent (default: 0.5)**
- **Loss function: Weighted BCE (Binary Cross Entropy)**

---

### **CATEGORY 3: Neural Network Models (Logits → Sigmoid → Threshold)**

**Models**: LSTM, BiLSTM, CNN Attention
**File**: `src/models/models_nn.py`

#### Pipeline:
```
Input: Tokenized sequences (max_len=128)
    ↓
[Embedding Layer]
    ↓
[LSTM/BiLSTM or CNN]
    ↓
[Pooling Layer (mean/max)]
    ↓
[Dense Layer] → Output logits (batch_size, 3)
[Real-valued scores]
    ↓
Convert to numpy & Sigmoid activation
    ↓
Threshold comparison: prob > 0.5
    ↓
Output: Binary multilabel array (n_samples, 3)
```

#### Prediction Generation Code (LSTM/BiLSTM/CNN):
```python
# From models_nn.py - _train_eval()
all_logits = []
with torch.no_grad():
    for bx, _ in test_loader:
        bx = bx.to(device)
        all_logits.append(model(bx).cpu().numpy())

logits = np.vstack(all_logits)  # (n_samples, 3), float values
preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
# Result: (n_samples, 3), {0, 1}
```

#### Model Architecture Details:

**LSTM/BiLSTM:**
```
Embedding (vocab_size, 300)
    ↓
[LSTM/BiLSTM] → hidden_dim=128 (or BiLSTM=256)
    ↓
Mean Pooling over time
    ↓
Dropout(0.3)
    ↓
Dense(hidden_dim, 3) → logits
```

**CNN Attention:**
```
Embedding (vocab_size, 300)
    ↓
[Conv1d] with multiple filter sizes
    ↓
[Max Pooling] per filter
    ↓
[Attention] to weight filters
    ↓
Dense(attention_dim, 3) → logits
```

#### Output Format:
```
logits shape: (240, 3)
logits dtype: float32/float64
logits values: ~[-5, 5] range

After sigmoid + threshold:
preds shape: (240, 3)
preds dtype: int64
preds values: {0, 1}

Example (LSTM):
[[−0.12  1.89 −0.45]
 [ 2.34  2.67  3.01]
 [ 0.98 −1.23  1.56]]

Example predictions:
[[0 1 0]
 [1 1 1]
 [1 0 1]]
```

#### Key Characteristics:
- **Same sigmoid + threshold as Transformers**
- **Similar loss function: BCEWithLogitsLoss (combines sigmoid + BCE)**
- **Batched inference**
- **PyTorch models (GPU-accelerated)**

---

### **CATEGORY 4: LLM Models (Text Parsing → Binary)**

**Models**: LLM Zero-Shot, LLM Few-Shot
**File**: `src/models/models_llm.py`

#### Pipeline:
```
Input: Test text sample
    ↓
Build prompt (instruction + examples for few-shot)
    ↓
Groq API Call (llama-3.1-8b-instant)
    ↓
Generated JSON response
    ↓
Parse JSON & Extract label values
    ↓
Convert to binary (≥1 → 1, else → 0)
    ↓
Stack into array (n_samples, 3)
    ↓
Output: Binary multilabel array (n_samples, 3)
```

#### Prediction Generation Code:
```python
# From models_llm.py
def _parse_prediction(raw_text: str) -> Tuple[np.ndarray, bool]:
    json_blob = _extract_json_block(raw_text)  # Extract JSON from response
    if not json_blob:
        return np.array([0, 0, 0], dtype=int), False
    
    try:
        data = json.loads(json_blob)
    except Exception:
        return np.array([0, 0, 0], dtype=int), False
    
    # Convert to binary (safe_int01 ensures 0 or 1)
    pred = np.array([_safe_int01(data.get(k, 0)) for k in LABEL_COLUMNS], dtype=int)
    return pred, True

# Main inference loop
pred_rows: List[np.ndarray] = []
for text in test_texts:
    prompt = _build_prompt(text, mode, few_shot_examples)
    completion = client.chat.completions.create(...)
    generated = completion.choices[0].message.content
    pred, ok = _parse_prediction(generated)
    pred_rows.append(pred)

y_pred = np.stack(pred_rows)  # (n_samples, 3)
```

#### Example LLM Response:
```json
{
  "relevance": 1,
  "concreteness": 0,
  "constructive": 1
}
```

#### Output Format:
```
Before parsing: JSON string from LLM
    "relevance": 1, "concreteness": 0, "constructive": 1

After parsing:
preds shape: (240, 3)
preds dtype: int64
preds values: {0, 1}

Example:
[[1 0 1]
 [0 1 1]
 [1 1 0]]
```

#### Key Characteristics:
- **Semantic understanding (text-to-label)**
- **No numerical intermediate values**
- **Parsing-based (vulnerable to format errors)**
- **One prediction per API call (slower)**
- **Natural language interpretation**
- **Fallback to [0,0,0] on parse failure**

---

## Comparative Analysis Table

| Aspect | LinearSVM | NaiveBayes | LogReg | BERT | RoBERTa | LSTM | BiLSTM | CNN | LLM |
|--------|-----------|-----------|---------|------|---------|------|--------|-----|-----|
| **Input Type** | Embeddings | Embeddings | Embeddings | Tokens | Tokens | Tokens | Tokens | Tokens | Text |
| **Intermediate Output** | Binary | Binary | Binary | Logits | Logits | Logits | Logits | Logits | JSON |
| **Intermediate Range** | {0,1} | {0,1} | {0,1} | ±5 | ±5 | ±5 | ±5 | ±5 | String |
| **Post-Processing** | None | None | None | Sigmoid+Threshold | Sigmoid+Threshold | Sigmoid+Threshold | Sigmoid+Threshold | Sigmoid+Threshold | Parse+Binary |
| **Final Shape** | (N,3) | (N,3) | (N,3) | (N,3) | (N,3) | (N,3) | (N,3) | (N,3) | (N,3) |
| **Final Dtype** | int64 | int64 | int64 | int64 | int64 | int64 | int64 | int64 | int64 |
| **Final Values** | {0,1} | {0,1} | {0,1} | {0,1} | {0,1} | {0,1} | {0,1} | {0,1} | {0,1} |
| **Speed** | Fast | Fast | Fast | Medium | Medium | Slow | Slow | Slow | Very Slow |
| **GPU Required** | No | No | No | Yes* | Yes* | Yes* | Yes* | Yes* | No |

---

## Unified Final Format (After Processing)

### Shape
```python
(n_samples, 3)

Per fold:    ~(240, 3)
All folds:   (2398, 3)
```

### Data Type
```python
np.int64 or Python int
```

### Values
```python
Binary encoding: {0, 1}

Column Mapping:
- Column 0: relevance
- Column 1: concreteness
- Column 2: constructive
```

### Example Array (All Models)
```python
predictions = np.array([
    [0, 1, 0],  # Sample 1: not relevant, concrete, not constructive
    [1, 1, 1],  # Sample 2: all labels present
    [1, 0, 1],  # Sample 3: relevant, not concrete, constructive
    [0, 0, 0],  # Sample 4: no labels present
    [1, 1, 0],  # Sample 5: relevant, concrete, not constructive
], dtype=np.int64)

predictions.shape   # (5, 3)
predictions.dtype   # dtype('int64')
np.unique(predictions)  # array([0, 1])
```

---

## Intermediate Representations (Before Final Conversion)

### Machine Learning Models
```
No intermediate step - direct binary
predictions = model.predict(x_test)
```

### Deep Learning Models (LSTM/CNN)
```
Logits: (batch_size, 3), dtype=float32/float64
Range: approximately [-5, 5]
Example:
[[-0.45  2.13 -1.23]
 [ 1.89  2.45  3.12]
 [ 1.23 -0.89  1.45]]

After sigmoid:
Probabilities: (batch_size, 3), dtype=float64, range [0, 1]
Example:
[[0.39  0.89  0.22]
 [0.87  0.92  0.96]
 [0.77  0.28  0.81]]

After threshold (> 0.5):
Binary: (batch_size, 3), dtype=int64, values {0, 1}
```

### Transformer Models (BERT/RoBERTa)
```
Logits: (n_samples, 3), dtype=float32
Range: approximately [-5, 5]
(Same as neural networks)

After sigmoid + threshold:
Binary: (n_samples, 3), dtype=int64, values {0, 1}
```

### LLM Models
```
Raw response: JSON string
{"relevance": 1, "concreteness": 0, "constructive": 1}

After parsing:
Binary: (n_samples, 3), dtype=int64, values {0, 1}
```

---

## Implementation Details

### Sigmoid Function Used
```python
# All neural/transformer models use this
sigmoid = 1 / (1 + np.exp(-logits))

# Mathematically equivalent to:
# sigmoid = sp.special.expit(logits)

# Properties:
# sigmoid(-∞) = 0
# sigmoid(0) = 0.5
# sigmoid(+∞) = 1
# sigmoid(x) = 1 - sigmoid(-x)  [symmetric around 0.5]
```

### Threshold Behavior
```python
prediction = 1 if sigmoid(logit) ≥ 0.5 else 0

# At the decision boundary (logit=0):
sigmoid(0) = 0.5
# Behavior: Rounds to 1 (≥ comparison)

# Sensitivity analysis (sigmoid values):
sigmoid(-2.0) ≈ 0.119 → predict 0
sigmoid(-1.0) ≈ 0.269 → predict 0
sigmoid(-0.5) ≈ 0.378 → predict 0
sigmoid(0.0)  = 0.5   → predict 1 (edge case)
sigmoid(+0.5) ≈ 0.622 → predict 1
sigmoid(+1.0) ≈ 0.731 → predict 1
sigmoid(+2.0) ≈ 0.881 → predict 1
```

---

## Consistency Verification

All 10 models produce consistent output:

```python
# For any model
predictions = model_inference(test_data)

# Guaranteed properties:
assert predictions.shape == (n_test_samples, 3)
assert predictions.dtype in [np.int64, np.int32, int]
assert np.all(np.isin(predictions, [0, 1]))
```

---

## Summary

| **Step** | **LinearSVM/NB/LogReg** | **LSTM/BiLSTM/CNN** | **BERT/RoBERTa** | **LLM** |
|---------|------------------------|-------------------|------------------|---------|
| 1. Input | Word2Vec embeddings | Token sequences | Token sequences | Text string |
| 2. Model Processing | Classifier | Neural network | Transformer | LLM API |
| 3. Intermediate | Binary {0,1} | Logits [−5,5] | Logits [−5,5] | JSON string |
| 4. Conversion | — | Sigmoid + threshold | Sigmoid + threshold | Parse + binary |
| 5. Final Output | (N,3) {0,1} | (N,3) {0,1} | (N,3) {0,1} | (N,3) {0,1} |

**All models converge to the same final format: (n_samples, 3) np.int64 array with binary values {0, 1}**
