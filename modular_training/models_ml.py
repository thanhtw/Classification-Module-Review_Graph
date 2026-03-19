"""
Machine learning models for multilabel text classification.

Pipeline:
1. Preprocess Chinese text (tokenization, character cleaning)
2. Word2Vec vectorization (300-dim dense embeddings)
3. Train/validation/test split
4. Optional class imbalance handling (SMOTE)
5. OneVsRestClassifier with LinearSVC, LogisticRegression, or GaussianNB
6. Multilabel evaluation (accuracy, precision, recall, F1, hamming loss)
7. Save vectorizer + model + label mapping
"""

import time
import json
import os
import pickle
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression

from .data_utils import preprocess_for_tfidf
from .embeddings_word2vec import Word2VecVectorizer
from .metrics_utils import compute_metrics
from .smote_utils import apply_smote_multilabel


LABEL_NAMES = ["relevance", "concreteness", "constructive"]


def _prepare_data(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    use_smote: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Word2VecVectorizer, Dict]:
    """
    Prepare data: Word2Vec embedding extraction and optional SMOTE.
    
    Returns:
        x_train: Word2Vec document embeddings for training (dense)
        y_train: Training labels (possibly resampled)
        x_test: Word2Vec document embeddings for testing (dense)
        vectorizer: Fitted Word2VecVectorizer
        smote_stats: Dictionary with SMOTE application details
    """
    # Step 1: Preprocess Chinese text (already embedded in Word2VecVectorizer)
    train_texts_list = list(train_texts)
    test_texts_list = list(test_texts)

    # Step 2: Word2Vec vectorization (dense embeddings)
    vectorizer = Word2VecVectorizer(
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        seed=seed,
    )
    x_train = vectorizer.fit(train_texts_list).transform(train_texts_list)  # (n_train, 300)
    x_test = vectorizer.transform(test_texts_list)  # (n_test, 300)

    # Step 3-4: Optional class imbalance handling
    y_train = train_labels
    x_train_fit = x_train
    smote_stats = {"applied": False, "method": "disabled"}

    if use_smote:
        # SMOTE works with dense arrays
        x_res, y_res, smote_stats = apply_smote_multilabel(
            x_train.astype(np.float32), train_labels, seed=seed
        )
        x_train_fit = x_res
        y_train = y_res
        smote_stats["applied"] = True

    return x_train_fit, y_train, x_test, vectorizer, smote_stats


def run_linear_svm(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    use_smote: bool,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    """
    Linear Support Vector Machine with OneVsRestClassifier.
    
    Pipeline:
    1. Preprocess Chinese text
    2. TF-IDF vectorization
    3. Optional SMOTE for imbalance handling  
    4. OneVsRestClassifier(LinearSVC) - one binary classifier per label
    5. Evaluate multilabel metrics
    6. Save model, vectorizer, and metadata
    
    Args:
        train_texts: Training text samples
        train_labels: Training binary multilabel targets (n_samples, 3)
        test_texts: Test text samples
        test_labels: Test binary multilabel targets
        use_smote: Whether to apply SMOTE for imbalance handling
        seed: Random seed for reproducibility
        save_dir: Directory to save model artifacts
        
    Returns:
        metrics: Dict with accuracy, precision, recall, F1, hamming_loss (micro/macro)
        train_time: Training time in seconds
        infer_time: Inference time in seconds
    """
    x_train, y_train, x_test, vectorizer, smote_stats = _prepare_data(
        train_texts, train_labels, test_texts, use_smote, seed
    )

    # Step 5: OneVsRestClassifier with LinearSVC
    # LinearSVC: fast, efficient for high-dimensional sparse data
    model = OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            dual="auto",
        )
    )

    # Training
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start

    # Step 7: Inference and evaluation
    infer_start = time.perf_counter()
    preds = model.predict(x_test)
    infer_time = time.perf_counter() - infer_start

    metrics = compute_metrics(test_labels, preds)

    # Step 8: Save artifacts
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model and vectorizer
        with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "labels": LABEL_NAMES,
                },
                f,
            )
        
        # Save metadata
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "linear_svm_ovr",
                    "model_description": "OneVsRestClassifier with LinearSVC",
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_train)),
                    "vectorizer_config": {
                        "max_features": 10000,
                        "ngram_range": [1, 2],
                        "min_df": 2,
                        "max_df": 0.95,
                    },
                    "labels": LABEL_NAMES,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, train_time, infer_time


def run_naive_bayes(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    use_smote: bool,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    """
    Naive Bayes with OneVsRestClassifier.
    
    Pipeline:
    1. Preprocess Chinese text
    2. TF-IDF vectorization
    3. Optional SMOTE for imbalance handling
    4. OneVsRestClassifier(MultinomialNB) - one binary classifier per label
    5. Evaluate multilabel metrics
    6. Save model, vectorizer, and metadata
    
    Args:
        train_texts: Training text samples
        train_labels: Training binary multilabel targets (n_samples, 3)
        test_texts: Test text samples
        test_labels: Test binary multilabel targets
        use_smote: Whether to apply SMOTE for imbalance handling
        seed: Random seed for reproducibility
        save_dir: Directory to save model artifacts
        
    Returns:
        metrics: Dict with accuracy, precision, recall, F1, hamming_loss (micro/macro)
        train_time: Training time in seconds
        infer_time: Inference time in seconds
    """
    x_train, y_train, x_test, vectorizer, smote_stats = _prepare_data(
        train_texts, train_labels, test_texts, use_smote, seed
    )

    # Step 5: OneVsRestClassifier with GaussianNB
    # GaussianNB: suitable for continuous-valued Word2Vec embeddings
    # (MultinomialNB requires non-negative features, but Word2Vec embeddings can be negative)
    model = OneVsRestClassifier(GaussianNB())

    # Training
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start

    # Step 7: Inference and evaluation
    infer_start = time.perf_counter()
    preds = model.predict(x_test)
    infer_time = time.perf_counter() - infer_start

    metrics = compute_metrics(test_labels, preds)

    # Step 8: Save artifacts
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model and vectorizer
        with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "labels": LABEL_NAMES,
                },
                f,
            )
        
        # Save metadata
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "naive_bayes_ovr",
                    "model_description": "OneVsRestClassifier with GaussianNB for Word2Vec embeddings",
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_train)),
                    "vectorizer_config": {
                        "type": "Word2Vec",
                        "vector_size": 300,
                        "window": 5,
                        "sg": 1,
                    },
                    "labels": LABEL_NAMES,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, train_time, infer_time


def run_logistic_regression(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    use_smote: bool,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    """
    Logistic Regression with OneVsRestClassifier (bonus model).
    
    Pipeline:
    1. Preprocess Chinese text
    2. TF-IDF vectorization
    3. Optional SMOTE for imbalance handling
    4. OneVsRestClassifier(LogisticRegression) - one binary classifier per label
    5. Inference with default threshold (0.5)
    6. Evaluate multilabel metrics
    7. Save model, vectorizer, and metadata
    
    Args:
        train_texts: Training text samples
        train_labels: Training binary multilabel targets (n_samples, 3)
        test_texts: Test text samples
        test_labels: Test binary multilabel targets
        use_smote: Whether to apply SMOTE for imbalance handling
        seed: Random seed for reproducibility
        save_dir: Directory to save model artifacts
        
    Returns:
        metrics: Dict with accuracy, precision, recall, F1, hamming_loss (micro/macro)
        train_time: Training time in seconds
        infer_time: Inference time in seconds
    """
    x_train, y_train, x_test, vectorizer, smote_stats = _prepare_data(
        train_texts, train_labels, test_texts, use_smote, seed
    )

    # Step 5: OneVsRestClassifier with LogisticRegression
    # LogisticRegression: probabilistic model with threshold support
    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
    )

    # Training
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start

    # Step 6-7: Inference and evaluation
    infer_start = time.perf_counter()
    preds = model.predict(x_test)
    infer_time = time.perf_counter() - infer_start

    metrics = compute_metrics(test_labels, preds)

    # Step 8: Save artifacts
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model and vectorizer
        with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "labels": LABEL_NAMES,
                },
                f,
            )
        
        # Save metadata
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "logistic_regression_ovr",
                    "model_description": "OneVsRestClassifier with LogisticRegression",
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_train)),
                    "vectorizer_config": {
                        "max_features": 10000,
                        "ngram_range": [1, 2],
                        "min_df": 2,
                        "max_df": 0.95,
                    },
                    "threshold": 0.5,
                    "labels": LABEL_NAMES,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metrics, train_time, infer_time
