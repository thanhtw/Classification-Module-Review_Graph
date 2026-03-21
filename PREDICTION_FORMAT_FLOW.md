#!/usr/bin/env python3
"""
VISUAL PREDICTION FORMAT FLOW FOR EACH MODEL TYPE

This document shows how each model generates predictions step-by-step
with concrete examples and intermediate values.
"""

# ============================================================================
# FLOW DIAGRAM 1: MACHINE LEARNING MODELS (LinearSVM, NaiveBayes, LogisticRegression)
# ============================================================================
"""
INPUT TEXT → VECTORIZATION → PREDICTION → OUTPUT

Example input:
  Test text: "这是一个相关且具体的评论"

STEP 1: Text Preprocessing & Vectorization
┌─────────────────────────────────────────────────────────────────┐
│ Original text: "这是一个相关且具体的评论"                        │
│      ↓                                                           │
│ Tokenize: ["这", "是", "一", "个", "相关", "且", "具体", "评论"] │
│      ↓                                                           │
│ Word2Vec embedding (each token → 300-dim vector)               │
│      ↓                                                           │
│ Mean pooling: Average all token vectors                         │
│      ↓                                                           │
│ Final embedding: [0.15, -0.32, 0.78, ..., 0.02]  (300 dims)    │
└─────────────────────────────────────────────────────────────────┘

STEP 2: OneVsRestClassifier Prediction
┌─────────────────────────────────────────────────────────────────┐
│ Input vector: [0.15, -0.32, 0.78, ..., 0.02]  (300-dim)        │
│      ↓                                                           │
│ Classifier 1 (Relevance):          decision_function → 0.85     │
│ → predict() → 1                                                 │
│      ↓                                                           │
│ Classifier 2 (Concreteness):       decision_function → -0.42    │
│ → predict() → 0                                                 │
│      ↓                                                           │
│ Classifier 3 (Constructive):       decision_function → 0.51     │
│ → predict() → 1                                                 │
│      ↓                                                           │
│ Final prediction: [1, 0, 1]                                     │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: [1, 0, 1]
  Column 0 = 1: Relevant (YES)
  Column 1 = 0: Concrete (NO)
  Column 2 = 1: Constructive (YES)

Saved format: np.array([1, 0, 1], dtype=int64)
"""

# ============================================================================
# FLOW DIAGRAM 2: TRANSFORMER MODELS (BERT, RoBERTa)
# ============================================================================
"""
INPUT TEXT → TOKENIZATION → ENCODER → LOGITS → SIGMOID → THRESHOLD → OUTPUT

Example input:
  Test text: "这是一个相关的评论"

STEP 1: Tokenization
┌─────────────────────────────────────────────────────────────────┐
│ Original: "这是一个相关的评论"                                  │
│     ↓                                                            │
│ BERT tokenizer: [CLS] + tokens + [SEP]                          │
│     ↓                                                            │
│ Token IDs: [101, 3341, 1928, 671, 2110, 3019, 3244, 1905, 102] │
│ (each token mapped to vocab index)                              │
└─────────────────────────────────────────────────────────────────┘

STEP 2: BERT Encoder
┌─────────────────────────────────────────────────────────────────┐
│ Token IDs: [101, 3341, 1928, 671, 2110, 3019, 3244, 1905, 102] │
│     ↓                                                            │
│ Embedding layer → 768-dim vectors for each token               │
│     ↓                                                            │
│ 12 transformer layers (self-attention, feed-forward)            │
│     ↓                                                            │
│ [CLS] token final hidden state: [0.23, -0.15, 0.89, ..., 0.1]  │
│ (768-dim vector)                                                │
└─────────────────────────────────────────────────────────────────┘

STEP 3: Classification Head (Dense Layer)
┌─────────────────────────────────────────────────────────────────┐
│ [CLS] embedding: [0.23, -0.15, 0.89, ..., 0.1]  (768 dims)     │
│     ↓                                                            │
│ Dense layer (768 → 3) with no activation                        │
│     ↓                                                            │
│ Raw logits: [2.15, -1.32, 0.78]                                 │
│ (unbounded real values)                                         │
└─────────────────────────────────────────────────────────────────┘

STEP 4: Sigmoid Activation
┌─────────────────────────────────────────────────────────────────┐
│ Logits: [2.15, -1.32, 0.78]                                     │
│     ↓                                                            │
│ Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))                         │
│     ↓                                                            │
│ sigmoid([2.15, -1.32, 0.78])                                    │
│ = [1/(1+e^-2.15), 1/(1+e^1.32), 1/(1+e^-0.78)]                 │
│ = [0.896, 0.210, 0.685]                                         │
│ (probabilities: 0 to 1 range)                                   │
└─────────────────────────────────────────────────────────────────┘

STEP 5: Threshold at 0.5
┌─────────────────────────────────────────────────────────────────┐
│ Probabilities: [0.896, 0.210, 0.685]                            │
│     ↓                                                            │
│ > 0.5 check:                                                    │
│   0.896 > 0.5 → 1 (YES, confident)                              │
│   0.210 > 0.5 → 0 (NO, not confident)                           │
│   0.685 > 0.5 → 1 (YES, confident)                              │
│     ↓                                                            │
│ Final predictions: [1, 0, 1]                                    │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: [1, 0, 1]
Saved format: np.array([1, 0, 1], dtype=int64)

Code implementation:
  preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
"""

# ============================================================================
# FLOW DIAGRAM 3: NEURAL NETWORK MODELS (LSTM, BiLSTM, CNN Attention)
# ============================================================================
"""
INPUT TEXT → TOKENIZATION → EMBEDDING → NETWORK → LOGITS → SIGMOID → THRESHOLD → OUTPUT

Example for LSTM:

STEP 1: Text to Sequences
┌─────────────────────────────────────────────────────────────────┐
│ Text: "这是一个相关的评论"                                      │
│     ↓                                                            │
│ Tokenize: ["这", "是", "一", "个", "相关", "的", "评论"]        │
│     ↓                                                            │
│ Convert to word IDs: [45, 23, 8, 102, 156, 12, 234]             │
│ (from vocabulary built during training)                         │
│     ↓                                                            │
│ Pad/truncate to max_len (e.g., 50):                              │
│ [45, 23, 8, 102, 156, 12, 234, 0, 0, ..., 0]  (50 total)       │
└─────────────────────────────────────────────────────────────────┘

STEP 2: Embedding Layer
┌─────────────────────────────────────────────────────────────────┐
│ Token sequence: [45, 23, 8, 102, 156, 12, 234, 0, 0, ..., 0]   │
│ (50 tokens)                                                     │
│     ↓                                                            │
│ Embedding (each token → 300-dim):                               │
│ [[e_45], [e_23], [e_8], [e_102], [e_156], [e_12], [e_234],    │
│  [e_0], [e_0], ..., [e_0]]                                      │
│ Shape: (50, 300)                                                │
│ (50 timesteps × 300-dim embeddings)                             │
└─────────────────────────────────────────────────────────────────┘

STEP 3: LSTM/BiLSTM Layer
┌─────────────────────────────────────────────────────────────────┐
│ Embedded sequence: Shape (50, 300)                              │
│     ↓                                                            │
│ LSTM (bidirectional=False):                                     │
│ - Process sequence left-to-right                                │
│ - Hidden state: 128-dim (LSTM hidden_dim)                       │
│ - Output for each timestep: (50, 128)                           │
│     ↓                                                            │
│ BiLSTM (bidirectional=True):                                    │
│ - Forward and backward passes                                   │
│ - Output: (50, 256) = 128 forward + 128 backward               │
│     ↓                                                            │
│ Mean pooling over timesteps:                                    │
│ Average all 50 timestep vectors → (256,) or (128,)             │
└─────────────────────────────────────────────────────────────────┘

STEP 4: Dropout
┌─────────────────────────────────────────────────────────────────┐
│ Pooled representation: [0.23, -0.15, 0.89, ..., 0.1]  (256-dim)│
│     ↓                                                            │
│ Dropout (p=0.5): Randomly set some values to 0 for regularization
│     ↓                                                            │
│ Result: [0.23, 0.0, 0.89, ..., 0.1]  (some values zeroed)      │
└─────────────────────────────────────────────────────────────────┘

STEP 5: Dense Classification Layer
┌─────────────────────────────────────────────────────────────────┐
│ Input: [0.23, 0.0, 0.89, ..., 0.1]  (256-dim)                  │
│     ↓                                                            │
│ Dense layer (256 → 3):                                          │
│ output = W @ input + b                                          │
│     ↓                                                            │
│ Raw logits: [1.82, -0.56, 1.34]                                 │
│ (unbounded real values)                                         │
└─────────────────────────────────────────────────────────────────┘

STEP 6-7: Sigmoid + Threshold (same as Transformers)
┌─────────────────────────────────────────────────────────────────┐
│ Logits: [1.82, -0.56, 1.34]                                     │
│     ↓                                                            │
│ Sigmoid: [0.860, 0.363, 0.793]                                  │
│     ↓                                                            │
│ > 0.5: [1, 0, 1]                                                │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: [1, 0, 1]
Saved format: np.array([1, 0, 1], dtype=int64)

Code for sigmoid + threshold:
  preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
"""

# ============================================================================
# FLOW DIAGRAM 4: LANGUAGE MODEL (LLM Zero-Shot & Few-Shot)
# ============================================================================
"""
INPUT TEXT → PROMPT BUILDING → LLM API CALL → JSON RESPONSE → PARSING → OUTPUT

Example:

STEP 1: Prompt Building
┌──────────────────────────────────────────────────────────────────┐
│ Mode: "zero_shot"                                               │
│ Test text: "这是一个相关的评论"                                 │
│     ↓                                                            │
│ Build prompt:                                                   │
│                                                                 │
│ "You are a strict multi-label classifier. Predict 3 binary     │
│  labels for the input text. Return ONLY the JSON object on a   │
│  single line:                                                  │
│                                                                 │
│  Output format: {\"relevance\": 0 or 1, \"concreteness\": 0 or 1,
│  \"constructive\": 0 or 1}                                      │
│                                                                 │
│  Text: 这是一个相关的评论                                       │
│  Output:"                                                       │
│     ↓                                                            │
│ Prompt for few_shot would include examples:                     │
│  Text: 相关且具体的评论                                         │
│  Answer: {"relevance": 1, "concreteness": 1, "constructive": 0} │
│  ...                                                            │
└──────────────────────────────────────────────────────────────────┘

STEP 2: LLM API Call (via Groq)
┌──────────────────────────────────────────────────────────────────┐
│ API Call to Groq:                                               │
│   model: "llama-3.1-8b-instant"                                 │
│   messages: [{"role": "user", "content": prompt}]               │
│   temperature: 0.0 (deterministic)                              │
│   max_tokens: 64                                                │
│     ↓                                                            │
│ LLM Processing:                                                 │
│ - Read prompt and text                                          │
│ - Analyze text for each label                                   │
│ - Generate JSON response                                        │
└──────────────────────────────────────────────────────────────────┘

STEP 3: LLM Response
┌──────────────────────────────────────────────────────────────────┐
│ Raw API response (text):                                        │
│                                                                 │
│ "{"relevance": 1, "concreteness": 1, "constructive": 0}"       │
│                                                                 │
│ Or with extra thinking text:                                   │
│ "Based on the text... the labels are:                           │
│  {"relevance": 1, "concreteness": 1, "constructive": 0}"       │
│                                                                 │
│ Or malformed:                                                  │
│ "The text appears relevant... confidence: high"                 │
│ (no valid JSON)                                                │
└──────────────────────────────────────────────────────────────────┘

STEP 4: JSON Extraction
┌──────────────────────────────────────────────────────────────────┐
│ Raw text: "...the labels are: {"relevance": 1, ...}"           │
│     ↓                                                            │
│ Regex search for {...}:                                        │
│ Found: {"relevance": 1, "concreteness": 1, "constructive": 0}  │
│     ↓                                                            │
│ JSON parse:                                                     │
│ {"relevance": 1,                                               │
│  "concreteness": 1,                                             │
│  "constructive": 0}                                             │
│     ↓                                                            │
│ Extract values:                                                 │
│ [1, 1, 0]                                                       │
└──────────────────────────────────────────────────────────────────┘

STEP 5: Value Conversion via _safe_int01()
┌──────────────────────────────────────────────────────────────────┐
│ Extracted values: [1, 1, 0]                                     │
│ Function _safe_int01(value):                                    │
│   - If int(value) >= 1: return 1                                │
│   - Else: return 0                                              │
│     ↓                                                            │
│ _safe_int01(1) → 1  (correct)                                   │
│ _safe_int01(1) → 1  (correct)                                   │
│ _safe_int01(0) → 0  (correct)                                   │
│     ↓                                                            │
│ Final prediction: np.array([1, 1, 0], dtype=int)                │
└──────────────────────────────────────────────────────────────────┘

Handling edge cases (all return valid binary or default):
  LLM output: {"relevance": 0.8, "concreteness": "yes", ...}
  After _safe_int01:
    0.8 → int(0.8) = 0 → 0
    "yes" → int("yes") fails → 0 (catched exception)
  Result: [0, 0, ...]

  No JSON found:
  Returns: np.array([0, 0, 0], dtype=int)  (default)

OUTPUT: [1, 1, 0]
Saved format: np.array([1, 1, 0], dtype=int64)

Code implementation:
  # Parse LLM response
  json_blob = _extract_json_block(raw_text)  # Regex to find {...}
  if not json_blob:
      return np.array([0, 0, 0], dtype=int), False
  
  try:
      data = json.loads(json_blob)
  except Exception:
      return np.array([0, 0, 0], dtype=int), False
  
  pred = np.array([_safe_int01(data.get(k, 0)) for k in LABEL_COLUMNS], dtype=int)
  return pred, True
"""

# ============================================================================
# SUMMARY TABLE: INPUT → OUTPUT FOR ALL MODELS
# ============================================================================
"""
All models follow similar patterns but with different intermediate steps:

Test Sample: "这是一个相关的评论"

┌──────────────┬─────────────────┬──────────────────┬──────────────┐
│ Model Type   │ Intermediate    │ Final Format     │ Final Output │
├──────────────┼─────────────────┼──────────────────┼──────────────┤
│ LinearSVM    │ [0.85, -0.42,   │ Direct binary    │ [1, 0, 1]   │
│              │  0.51]          │ (no threshold)   │ dtype:int64  │
├──────────────┼─────────────────┼──────────────────┼──────────────┤
│ BERT         │ [2.15, -1.32,   │ Sigmoid +        │ [1, 0, 1]   │
│              │  0.78]          │ Threshold=0.5    │ dtype:int64  │
├──────────────┼─────────────────┼──────────────────┼──────────────┤
│ LSTM         │ [1.82, -0.56,   │ Sigmoid +        │ [1, 0, 1]   │
│              │  1.34]          │ Threshold=0.5    │ dtype:int64  │
├──────────────┼─────────────────┼──────────────────┼──────────────┤
│ LLM          │ {"relevance": 1 │ JSON parse +     │ [1, 0, 1]   │
│              │  "concrete": 0, │ _safe_int01()    │ dtype:int64  │
│              │  "construct": 1}│                  │              │
└──────────────┴─────────────────┴──────────────────┴──────────────┘
"""

# ============================================================================
# BATCH PROCESSING: 240 TEST SAMPLES
# ============================================================================
"""
How all predictions are combined for one fold of one model:

Input: 240 test samples

ML Model (LinearSVM):
  Loop: for each sample in x_test:
    sample_vec (300,) → ML classifier → [0/1, 0/1, 0/1]
  Result array shape: (240, 3)

Transformer (BERT):
  Batch process: 240 samples together
    → BERT encoder → (240, 768) → Dense → (240, 3) logits
    → Sigmoid → (240, 3) probabilities
    → Threshold → (240, 3) binary
  Result array shape: (240, 3)

Neural Network (LSTM):
  Batch process: 240 samples in mini-batches
    → Embedding → LSTM → Pooling → Dense → (batch, 3) logits
    → Sigmoid → Threshold
  Concatenate batches: (240, 3)

LLM:
  Sequential: for each sample in test_texts:
    sample_text → Build prompt → LLM API → Parse JSON → [0/1, 0/1, 0/1]
  Stack: (240, 3)

Final saved array (all models):
  Shape: (240, 3)
  Dtype: int64
  Example:
  [[0, 1, 0],
   [1, 1, 1],
   [1, 0, 1],
   ...,
   [0, 1, 1]]

Across 10 folds:
  Fold 1: (240, 3)
  Fold 2: (240, 3)
  ...
  Fold 10: (240, 3)
  
  Aggregated: np.vstack([fold_1, fold_2, ..., fold_10]) = (2398, 3)
"""

# ============================================================================
