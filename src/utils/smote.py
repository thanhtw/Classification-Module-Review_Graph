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
    if SMOTE is None and RandomOverSampler is None:
        raise ModuleNotFoundError("SMOTE requested but imbalanced-learn is not installed")

    combo = encode_combos(y)
    uniq, cnt = np.unique(combo, return_counts=True)
    if len(uniq) < 2:
        stats = {
            "applied": 0,
            "n_before": int(len(y)),
            "n_after": int(len(y)),
            "method": "none",
            "label_pos_before": [int(y[:, i].sum()) for i in range(y.shape[1])],
            "label_neg_before": [int(len(y) - y[:, i].sum()) for i in range(y.shape[1])],
            "label_pos_after": [int(y[:, i].sum()) for i in range(y.shape[1])],
            "label_neg_after": [int(len(y) - y[:, i].sum()) for i in range(y.shape[1])],
        }
        return x, y, stats

    x_work = x.astype(np.float32, copy=True)
    y_combo = combo.copy()
    method = "none"

    singleton_strategy = {int(c): 2 for c, n in zip(uniq, cnt) if int(n) == 1}
    if singleton_strategy and RandomOverSampler is not None:
        ros = RandomOverSampler(random_state=seed, sampling_strategy=singleton_strategy)
        x_work, y_combo = ros.fit_resample(x_work, y_combo)
        method = "ros_singleton"

    uniq_w, cnt_w = np.unique(y_combo, return_counts=True)
    target = int(np.max(cnt_w))
    smote_strategy = {int(c): target for c, n in zip(uniq_w, cnt_w) if int(n) < target}

    if smote_strategy and SMOTE is not None and int(np.min(cnt_w)) >= 2:
        k = min(5, int(np.min(cnt_w)) - 1)
        sm = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=smote_strategy)
        x_res, y_res_combo = sm.fit_resample(x_work, y_combo)
        method = "smote" if method == "none" else "ros+smote"
    else:
        x_res, y_res_combo = x_work, y_combo

    # Decode combo labels first, then enforce per-label balancing.
    y_res = decode_combos(np.asarray(y_res_combo), y.shape[1]).astype(np.int64)

    # Second stage: bounded combo-greedy balancing.
    # This avoids expensive repeated concatenations and guarantees termination.
    rng = np.random.default_rng(seed)
    x_bal = x_res
    y_bal = y_res

    # diff_j = pos_j - neg_j = 2*pos_j - N. Goal is diff_j -> 0 for each label.
    pos_counts = y_bal.sum(axis=0).astype(int)
    diff = 2 * pos_counts - len(y_bal)

    # Build available combo vectors and source indices for fast sampling.
    combo_bits = [tuple(row.tolist()) for row in y_bal]
    combo_to_indices: Dict[Tuple[int, ...], list] = {}
    for idx, key in enumerate(combo_bits):
        combo_to_indices.setdefault(key, []).append(idx)
    available = list(combo_to_indices.keys())
    if available:
        deltas = {k: (2 * np.array(k, dtype=np.int64) - 1) for k in available}
    else:
        deltas = {}

    max_new = max(1000, len(y_bal))
    chosen_indices = []
    steps = 0
    while steps < max_new and int(np.abs(diff).sum()) > 0 and available:
        current_score = int(np.abs(diff).sum())
        best_key = None
        best_score = current_score

        for key in available:
            next_diff = diff + deltas[key]
            score = int(np.abs(next_diff).sum())
            if score < best_score:
                best_score = score
                best_key = key

        if best_key is None:
            break

        src = int(rng.choice(combo_to_indices[best_key]))
        chosen_indices.append(src)
        diff = diff + deltas[best_key]
        steps += 1

    if chosen_indices:
        pick = np.asarray(chosen_indices, dtype=np.int64)
        x_bal = np.vstack([x_bal, x_bal[pick]])
        y_bal = np.vstack([y_bal, y_bal[pick]])

    x_res = x_bal
    y_res = y_bal.astype(np.int64)

    if postprocess == "int":
        x_res = np.rint(x_res).astype(np.int64)
        x_res = np.clip(x_res, clip_min, clip_max)

    pos_before = [int(y[:, i].sum()) for i in range(y.shape[1])]
    neg_before = [int(len(y) - y[:, i].sum()) for i in range(y.shape[1])]
    pos_after = [int(y_res[:, i].sum()) for i in range(y.shape[1])]
    neg_after = [int(len(y_res) - y_res[:, i].sum()) for i in range(y.shape[1])]
    residual_diff_after = [int(abs(p - n)) for p, n in zip(pos_after, neg_after)]

    stats = {
        "applied": int(len(y_res) > len(y)),
        "n_before": int(len(y)),
        "n_after": int(len(y_res)),
        "method": f"{method}+label_balance",
        "label_pos_before": pos_before,
        "label_neg_before": neg_before,
        "label_pos_after": pos_after,
        "label_neg_after": neg_after,
        "label_abs_diff_after": residual_diff_after,
        "fully_balanced_each_label": int(all(v == 0 for v in residual_diff_after)),
        "label_balance_steps": int(steps),
    }
    return x_res, y_res, stats
