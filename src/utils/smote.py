from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def encode_combos(labels: np.ndarray) -> np.ndarray:
    combos = []
    for row in labels:
        combos.append(int("".join(str(int(v)) for v in row), 2))
    return np.asarray(combos, dtype=np.int64)


def decode_combos(combo_labels: np.ndarray, n_labels: int) -> np.ndarray:
    out = np.zeros((len(combo_labels), n_labels), dtype=np.int64)
    for i, c in enumerate(combo_labels):
        bits = format(int(c), f"0{n_labels}b")
        out[i] = np.array([int(b) for b in bits], dtype=np.int64)
    return out


def apply_smote_multilabel(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    postprocess: str = "float",
    clip_min: int = 0,
    clip_max: int = 1,
    target_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Resample minority labels using complete, label-preserving samples.

    ``target_ratio`` is the desired minority/majority ratio for each binary label.
    Sampling is sequential because labels overlap. Unlike the previous implementation,
    this never fabricates secondary labels or interpolates categorical token IDs.
    Consequently it is safe for tokenized transformer/RNN inputs as well as continuous
    ML features. Resampling must only be applied to a training fold.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if y.ndim != 2:
        raise ValueError("y must be a two-dimensional multilabel matrix")
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ratio must be in the interval (0, 1]")

    n_samples, n_labels = y.shape
    rng = np.random.default_rng(seed)
    selected = np.arange(n_samples, dtype=np.int64)
    added_per_label = []

    for label_idx in range(n_labels):
        current = y[selected, label_idx]
        classes, counts = np.unique(current, return_counts=True)
        if len(classes) < 2:
            added_per_label.append(0)
            continue
        minority_class = classes[int(np.argmin(counts))]
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        desired_minority = int(np.ceil(target_ratio * majority_count))
        deficit = max(0, desired_minority - minority_count)
        candidates = selected[current == minority_class]
        if deficit and len(candidates):
            selected = np.concatenate([selected, rng.choice(candidates, size=deficit, replace=True)])
        added_per_label.append(deficit)

    x_resampled = x[selected].copy()
    y_resampled = y[selected].copy()

    if postprocess == "int":
        x_resampled = np.rint(x_resampled).astype(np.int64)
        x_resampled = np.clip(x_resampled, clip_min, clip_max)

    pos_before = [int(y[:, i].sum()) for i in range(y.shape[1])]
    neg_before = [int(len(y) - y[:, i].sum()) for i in range(y.shape[1])]


    pos_after = [int(y_resampled[:, i].sum()) for i in range(y_resampled.shape[1])]
    neg_after = [int(len(y_resampled) - y_resampled[:, i].sum()) for i in range(y_resampled.shape[1])]
    residual_diff_after = [int(abs(p - n)) for p, n in zip(pos_after, neg_after)]

    # For reporting
    combos_after = encode_combos(y_resampled)
    uniq_after, counts_after = np.unique(combos_after, return_counts=True)
    combo_balanced = int(np.all(counts_after == counts_after[0]))

    stats = {
        "applied": int(len(y_resampled) > len(y)),
        "n_before": int(len(y)),
        "n_after": int(len(y_resampled)),
        "method": "multilabel_label_preserving_random_oversampling",
        "backend": "numpy_random_generator",
        "target_ratio": float(target_ratio),
        "label_pos_before": pos_before,
        "label_neg_before": neg_before,
        "label_pos_after": pos_after,
        "label_neg_after": neg_after,
        "label_abs_diff_after": residual_diff_after,
        "added_per_label_step": added_per_label,
        "combo_balanced": combo_balanced,
        "combo_counts_after": {str(int(c)): int(cnt) for c, cnt in zip(uniq_after, counts_after)},
        "label_balance_steps": int(len(y_resampled) - len(y)),
    }
    return x_resampled, y_resampled, stats
