import json
import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from src.training.config import LABEL_COLUMNS
from src.data.preprocessor import preprocess_for_tfidf
from src.utils.smote import apply_smote_multilabel, encode_combos


def _label_distribution(labels: np.ndarray, split: str) -> pd.DataFrame:
    rows = []
    n = len(labels)
    for i, name in enumerate(LABEL_COLUMNS):
        pos = int(labels[:, i].sum())
        neg = int(n - pos)
        rows.append(
            {
                "split": split,
                "label": name,
                "positive_count": pos,
                "negative_count": neg,
                "positive_ratio": float(pos / n if n else 0.0),
            }
        )
    return pd.DataFrame(rows)


def _combo_distribution(labels: np.ndarray, split: str) -> pd.DataFrame:
    combo = encode_combos(labels)
    uniq, cnt = np.unique(combo, return_counts=True)
    n = len(labels)
    rows = []
    for c, k in zip(uniq, cnt):
        rows.append(
            {
                "split": split,
                "combo_int": int(c),
                "combo_bin": format(int(c), f"0{labels.shape[1]}b"),
                "count": int(k),
                "ratio": float(k / n if n else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def _stat_table(labels: np.ndarray) -> pd.DataFrame:
    n = len(labels)
    rows = []
    pretty = ["Relevance", "Concreteness", "Constructiveness"]
    for i, p in enumerate(pretty):
        pos = int(labels[:, i].sum())
        rows.append(
            {
                "Quality Indicator": p,
                "Positive Samples": pos,
                "Negative Samples": int(n - pos),
                "Total Comments": int(n),
                "Proportion": f"{(pos / n * 100):.2f}%" if n else "0.00%",
            }
        )
    return pd.DataFrame(rows)


def export_train_smote_analysis(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    output_dir: str,
    seed: int,
    use_smote: bool,
    target_ratio: float = 1.0,
    full_labels: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    x = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95).fit_transform(
        [preprocess_for_tfidf(t) for t in train_texts]
    )
    x_before = x.toarray().astype(np.float32)

    y_before = train_labels.astype(np.int64)
    stats = {"applied": 0, "method": "disabled", "n_before": int(len(y_before)), "n_after": int(len(y_before))}

    if use_smote:
        x_after, y_after, stats = apply_smote_multilabel(
            x_before, y_before, seed=seed, target_ratio=target_ratio
        )
    else:
        x_after = x_before.copy()
        y_after = y_before.copy()

    # Persist train data snapshots for reproducibility and future inference/debug.
    before_df = pd.DataFrame(
        {
            "text": list(train_texts),
            "relevance": y_before[:, 0],
            "concreteness": y_before[:, 1],
            "constructive": y_before[:, 2],
        }
    )
    before_df.to_csv(os.path.join(output_dir, "train_before_smote.csv"), index=False, encoding="utf-8")

    after_df = pd.DataFrame(
        {
            "sample_id": [f"A{i+1:06d}" for i in range(len(y_after))],
            "relevance": y_after[:, 0],
            "concreteness": y_after[:, 1],
            "constructive": y_after[:, 2],
            "is_original_sample": [1 if i < len(y_before) else 0 for i in range(len(y_after))],
        }
    )
    after_df.to_csv(os.path.join(output_dir, "train_after_smote_labels.csv"), index=False, encoding="utf-8")

    np.savez_compressed(
        os.path.join(output_dir, "train_features_before_smote.npz"),
        x=x_before,
        y=y_before,
    )
    np.savez_compressed(
        os.path.join(output_dir, "train_features_after_smote.npz"),
        x=x_after,
        y=y_after,
    )

    label_before = _label_distribution(y_before, "before_smote_train")
    label_after = _label_distribution(y_after, "after_smote_train")
    combo_before = _combo_distribution(y_before, "before_smote_train")
    combo_after = _combo_distribution(y_after, "after_smote_train")

    stat_before = _stat_table(y_before)
    stat_after = _stat_table(y_after)

    label_before.to_csv(os.path.join(output_dir, "train_label_distribution_before_smote.csv"), index=False, encoding="utf-8")
    label_after.to_csv(os.path.join(output_dir, "train_label_distribution_after_smote.csv"), index=False, encoding="utf-8")
    combo_before.to_csv(os.path.join(output_dir, "train_label_combo_distribution_before_smote.csv"), index=False, encoding="utf-8")
    combo_after.to_csv(os.path.join(output_dir, "train_label_combo_distribution_after_smote.csv"), index=False, encoding="utf-8")

    stat_before.to_csv(os.path.join(output_dir, "train_statistical_characteristics_before_smote.csv"), index=False, encoding="utf-8")
    stat_after.to_csv(os.path.join(output_dir, "train_statistical_characteristics_after_smote.csv"), index=False, encoding="utf-8")

    with open(os.path.join(output_dir, "train_statistical_characteristics_before_smote.tex"), "w", encoding="utf-8") as f:
        f.write(stat_before.to_latex(index=False, escape=False))
    with open(os.path.join(output_dir, "train_statistical_characteristics_after_smote.tex"), "w", encoding="utf-8") as f:
        f.write(stat_after.to_latex(index=False, escape=False))

    merged = label_before.merge(label_after, on="label", suffixes=("_before", "_after"))

    reliability = merged[
        ["label", "positive_count_before", "negative_count_before", "positive_count_after", "negative_count_after"]
    ].copy()
    reliability = reliability.rename(
        columns={
            "positive_count_before": "original_positive_count",
            "negative_count_before": "original_negative_count",
            "positive_count_after": "resampled_positive_count",
            "negative_count_after": "resampled_negative_count",
        }
    )
    reliability["added_positive_copies"] = (
        reliability["resampled_positive_count"] - reliability["original_positive_count"]
    )
    reliability["positive_count_inflation_factor"] = np.where(
        reliability["original_positive_count"] > 0,
        reliability["resampled_positive_count"] / reliability["original_positive_count"],
        np.nan,
    )
    reliability["independent_positive_evidence"] = reliability["original_positive_count"]
    reliability["resampling_adds_independent_evidence"] = False
    reliability["interpretation"] = (
        "Resampling changes class balance for fitting but does not increase independent sample support."
    )
    reliability.to_csv(
        os.path.join(output_dir, "smote_before_after_reliability.csv"), index=False, encoding="utf-8"
    )

    constructive = reliability.loc[reliability["label"] == "constructive"].iloc[0].to_dict()
    constructive_index = LABEL_COLUMNS.index("constructive")
    full_original_positive_count = (
        int(np.asarray(full_labels)[:, constructive_index].sum())
        if full_labels is not None
        else int(constructive["original_positive_count"])
    )
    reliability_summary = {
        "resampling_method": stats.get("method", "disabled"),
        "target_ratio": float(target_ratio),
        "original_training_samples": int(len(y_before)),
        "resampled_training_rows": int(len(y_after)),
        "full_dataset_original_constructive_positive_count": full_original_positive_count,
        "constructiveness": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in constructive.items()
        },
        "reliability_conclusion": (
            "Only original observations count as independent evidence. Oversampled copies can help model fitting, "
            "but they do not increase confidence in constructiveness Precision, Recall, or F1. Those metrics must "
            "be reported on untouched validation/test observations with uncertainty intervals."
        ),
    }
    with open(os.path.join(output_dir, "smote_before_after_reliability.json"), "w", encoding="utf-8") as f:
        json.dump(reliability_summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "smote_before_after_reliability.md"), "w", encoding="utf-8") as f:
        f.write("# Constructiveness reliability before and after resampling\n\n")
        f.write(
            f"The complete dataset contains **{full_original_positive_count} original constructive-positive "
            f"observations**. After reserving the untouched 20% final test set, the 80% development set contains "
            f"**{int(constructive['original_positive_count'])} original "
            f"constructive-positive observations** before resampling. After resampling it contains "
            f"**{int(constructive['resampled_positive_count'])} positive rows**, including "
            f"**{int(constructive['added_positive_copies'])} added copies**.\n\n"
        )
        f.write(
            "The added rows are not independent observations and therefore do not increase the statistical "
            "support for constructiveness. They address training imbalance only. Reliability is assessed using "
            "10-fold validation within the original 80% development split and bootstrap confidence intervals "
            "on the untouched 20% final test set.\n"
        )

    plt.figure(figsize=(12, 5))
    x_idx = np.arange(len(LABEL_COLUMNS))
    w = 0.35
    plt.subplot(1, 2, 1)
    plt.bar(x_idx - w / 2, merged["positive_count_before"], width=w, label="Before", color="#2563EB")
    plt.bar(x_idx + w / 2, merged["positive_count_after"], width=w, label="After", color="#14B8A6")
    plt.xticks(x_idx, merged["label"], rotation=20)
    plt.ylabel("Count")
    plt.title("Train Positive Label Counts")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(x_idx - w / 2, merged["positive_ratio_before"], width=w, label="Before", color="#2563EB")
    plt.bar(x_idx + w / 2, merged["positive_ratio_after"], width=w, label="After", color="#14B8A6")
    plt.xticks(x_idx, merged["label"], rotation=20)
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.title("Train Positive Label Ratios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_train_label_distribution_before_after_smote.png"), dpi=300)
    plt.close()

    top_union = pd.concat(
        [
            combo_before[["combo_bin", "count"]].assign(source="before"),
            combo_after[["combo_bin", "count"]].assign(source="after"),
        ]
    )
    top_bins = (
        top_union.groupby("combo_bin", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(10)["combo_bin"]
        .tolist()
    )
    b_top = combo_before.set_index("combo_bin").reindex(top_bins).fillna(0)
    a_top = combo_after.set_index("combo_bin").reindex(top_bins).fillna(0)

    plt.figure(figsize=(12, 5))
    idx = np.arange(len(top_bins))
    plt.bar(idx - w / 2, b_top["count"], width=w, label="Before", color="#2563EB")
    plt.bar(idx + w / 2, a_top["count"], width=w, label="After", color="#14B8A6")
    plt.xticks(idx, top_bins, rotation=30)
    plt.ylabel("Count")
    plt.title("Train Top Label Combination Distribution Before/After SMOTE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paper_train_combo_distribution_before_after_smote.png"), dpi=300)
    plt.close()

    with open(os.path.join(output_dir, "train_smote_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def export_constructiveness_evaluation_reliability(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    seed: int,
    bootstrap_iterations: int = 2000,
) -> None:
    """Report minority-class metrics and paired-row bootstrap uncertainty on untouched test data."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    label_index = LABEL_COLUMNS.index("constructive")

    for model, (y_true_all, y_pred_all) in model_results.items():
        y_true = np.asarray(y_true_all)[:, label_index].astype(int)
        y_pred = np.asarray(y_pred_all)[:, label_index].astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        samples = {"precision": [], "recall": [], "f1": []}
        for _ in range(bootstrap_iterations):
            indices = rng.integers(0, len(y_true), size=len(y_true))
            p, r, score, _ = precision_recall_fscore_support(
                y_true[indices], y_pred[indices], average="binary", zero_division=0
            )
            samples["precision"].append(float(p))
            samples["recall"].append(float(r))
            samples["f1"].append(float(score))
        row = {
            "model": model,
            "evaluation_split": "untouched_final_20_percent_test",
            "test_samples": int(len(y_true)),
            "original_positive_support": int(y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
            "true_positive": int(np.logical_and(y_true == 1, y_pred == 1).sum()),
            "bootstrap_iterations": int(bootstrap_iterations),
        }
        for metric, point in (("precision", precision), ("recall", recall), ("f1", f1)):
            row[metric] = float(point)
            row[f"{metric}_ci95_low"] = float(np.percentile(samples[metric], 2.5))
            row[f"{metric}_ci95_high"] = float(np.percentile(samples[metric], 97.5))
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(output_dir, "constructiveness_final_test_reliability.csv"), index=False)
    with open(os.path.join(output_dir, "constructiveness_final_test_reliability.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "paired-row nonparametric bootstrap", "models": rows}, f, indent=2)
    with open(os.path.join(output_dir, "constructiveness_final_test_reliability.md"), "w", encoding="utf-8") as f:
        f.write("# Constructiveness final-test reliability\n\n")
        f.write(
            "All support values below are original, untouched test observations; resampled training rows are "
            "excluded. Intervals are paired-row nonparametric bootstrap 95% confidence intervals.\n\n"
        )
        for row in rows:
            f.write(
                f"- **{row['model']}** — positive support: {row['original_positive_support']}; "
                f"Precision {row['precision']:.4f} [{row['precision_ci95_low']:.4f}, {row['precision_ci95_high']:.4f}]; "
                f"Recall {row['recall']:.4f} [{row['recall_ci95_low']:.4f}, {row['recall_ci95_high']:.4f}]; "
                f"F1 {row['f1']:.4f} [{row['f1_ci95_low']:.4f}, {row['f1_ci95_high']:.4f}].\n"
            )
