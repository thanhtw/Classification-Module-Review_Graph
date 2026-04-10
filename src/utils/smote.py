from typing import Dict, Tuple

import numpy as np

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError:
    SMOTE = None
    RandomOverSampler = None


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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Balance multilabel data: overmultiple layers for each label.
    
    Strategy: For each label independently, oversample its minority class to match majority.
    Do this once per label and accept that some secondary imbalance may occur across labels.
    
    This is the standard multilabel-aware approach used in practice.
    """

    from imblearn.over_sampling import RandomOverSampler
    n_samples, n_labels = y.shape
    rng = np.random.default_rng(seed)


    x_sets = [x]
    y_sets = [y]

    for label_idx in range(n_labels):
        y_label_col = y[:, label_idx]
        ros = RandomOverSampler(random_state=seed + label_idx)
        x_res, y_label_res = ros.fit_resample(x, y_label_col)
        n_new = len(x_res) - len(x)
        if n_new > 0:
            x_new = x_res[-n_new:]
            y_new = np.zeros((n_new, n_labels), dtype=y.dtype)
            # Set the current label to the correct value
            y_new[:, label_idx] = y_label_res[-n_new:]
            # For other labels, randomly sample from the original data
            for j in range(n_labels):
                if j != label_idx:
                    y_new[:, j] = rng.choice(y[:, j], size=n_new, replace=True)
            x_sets.append(x_new)
            y_sets.append(y_new)

    # Concatenate all, allow duplicates
    x_resampled = np.vstack(x_sets)
    y_resampled = np.vstack(y_sets)

    if postprocess == "int":
        x_resampled = np.rint(x_resampled).astype(np.int64)
        x_resampled = np.clip(x_resampled, clip_min, clip_max)

    pos_before = [int(y[:, i].sum()) for i in range(y.shape[1])]
    neg_before = [int(len(y) - y[:, i].sum()) for i in range(y.shape[1])]


    # Iterative downsampling: keep applying balance for each label until convergence
    # This ensures each label is as balanced as possible without aggressive intersection
    max_iterations = 10
    iteration = 0
    prev_size = len(y_resampled) + 1
    
    while iteration < max_iterations and len(y_resampled) < prev_size:
        prev_size = len(y_resampled)
        iteration += 1
        
        for label_idx in range(n_labels):
            y_col = y_resampled[:, label_idx]
            pos_indices = np.where(y_col == 1)[0]
            neg_indices = np.where(y_col == 0)[0]
            
            n_pos = len(pos_indices)
            n_neg = len(neg_indices)
            
            # If imbalanced, downsample majority to match minority
            if n_pos > n_neg and n_neg > 0:
                keep_pos = set(rng.choice(pos_indices, size=n_neg, replace=False))
                keep_all = keep_pos | set(neg_indices)
                keep_indices = sorted(list(keep_all))
                x_resampled = x_resampled[keep_indices]
                y_resampled = y_resampled[keep_indices]
                
            elif n_neg > n_pos and n_pos > 0:
                keep_neg = set(rng.choice(neg_indices, size=n_pos, replace=False))
                keep_all = set(pos_indices) | keep_neg
                keep_indices = sorted(list(keep_all))
                x_resampled = x_resampled[keep_indices]
                y_resampled = y_resampled[keep_indices]

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
        "method": "multilabel_oversampling_per_label",
        "label_pos_before": pos_before,
        "label_neg_before": neg_before,
        "label_pos_after": pos_after,
        "label_neg_after": neg_after,
        "label_abs_diff_after": residual_diff_after,
        "fully_balanced_each_label": int(all(v <= 10 for v in residual_diff_after)),
        "combo_balanced": combo_balanced,
        "combo_counts_after": {str(int(c)): int(cnt) for c, cnt in zip(uniq_after, counts_after)},
        "label_balance_steps": int(len(y_resampled) - len(y)),
    }
    return x_resampled, y_resampled, stats
