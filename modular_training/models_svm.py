import time
import json
import os
import pickle
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from .data_utils import preprocess_for_tfidf
from .metrics_utils import compute_metrics
from .smote_utils import apply_smote_multilabel


def run_svm(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    test_texts: Sequence[str],
    test_labels: np.ndarray,
    use_smote: bool,
    seed: int,
    save_dir: str = "",
) -> Tuple[Dict[str, float], float, float]:
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    x_train = vectorizer.fit_transform([preprocess_for_tfidf(t) for t in train_texts])
    x_test = vectorizer.transform([preprocess_for_tfidf(t) for t in test_texts])

    x_fit = x_train
    y_fit = train_labels
    x_test_eval = x_test
    smote_stats = {"applied": 0, "method": "disabled"}
    if use_smote:
        x_res, y_res, smote_stats = apply_smote_multilabel(x_train.toarray().astype(np.float32), train_labels, seed=seed)
        x_fit = x_res
        y_fit = y_res
        x_test_eval = x_test.toarray().astype(np.float32)

    model = MultiOutputClassifier(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))

    train_start = time.perf_counter()
    model.fit(x_fit, y_fit)
    train_time = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    preds = model.predict(x_test_eval)
    infer_time = time.perf_counter() - infer_start

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "labels": ["relevance", "concreteness", "constructive"],
                },
                f,
            )
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "svm_multioutput",
                    "use_smote": bool(use_smote),
                    "smote_stats": smote_stats,
                    "train_size_before": int(len(train_labels)),
                    "train_size_after": int(len(y_fit)),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return compute_metrics(test_labels, preds), train_time, infer_time
