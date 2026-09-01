from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import numpy as np

from src.training.config import LABEL_COLUMNS


def export_test_predictions(
    *,
    save_dir: str,
    model: str,
    fold: int,
    source_indices,
    texts,
    y_true,
    y_pred,
) -> str:
    """Persist inspectable row-level test predictions for any model family."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match labels {true.shape}")
    if true.ndim != 2 or true.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(
            f"Expected (n, {len(LABEL_COLUMNS)}) multilabel results, got {true.shape}"
        )
    if len(texts) != len(true) or len(source_indices) != len(true):
        raise ValueError("Texts, source indices, labels, and predictions must have equal length")

    data = {
        "source_index": [int(index) for index in source_indices],
        "text": [str(text) for text in texts],
        "model": [str(model)] * len(true),
        "fold": [int(fold)] * len(true),
    }
    for label_index, label in enumerate(LABEL_COLUMNS):
        data[f"true_{label}"] = true[:, label_index].astype(int)
        data[f"pred_{label}"] = pred[:, label_index].astype(int)
        data[f"correct_{label}"] = (true[:, label_index] == pred[:, label_index]).astype(int)
    data["exact_match"] = np.all(true == pred, axis=1).astype(int)

    destination = Path(save_dir) / "test_results_with_ground_truth.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(destination, index=False, encoding="utf-8")
    return str(destination)


def export_combined_final_test_predictions(
    *,
    output_dir: str,
    source_indices: Sequence[int],
    texts: Sequence[str],
    y_true: np.ndarray,
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, str]:
    """Store final-test data and predictions in wide and long inspectable CSV files."""
    true = np.asarray(y_true)
    if true.ndim != 2 or true.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(f"Expected final-test labels with shape (n, {len(LABEL_COLUMNS)})")
    if len(source_indices) != len(true) or len(texts) != len(true):
        raise ValueError("Final-test source indices, texts, and labels must have equal length")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    wide_data = {
        "source_index": [int(index) for index in source_indices],
        "text": [str(text) for text in texts],
    }
    for label_index, label in enumerate(LABEL_COLUMNS):
        wide_data[f"true_{label}"] = true[:, label_index].astype(int)
    wide_df = pd.DataFrame(wide_data)
    long_frames = []

    for model, (saved_labels, saved_predictions) in model_results.items():
        model_true = np.asarray(saved_labels)
        pred = np.asarray(saved_predictions)
        if model_true.shape != true.shape or pred.shape != true.shape:
            raise ValueError(f"Final-test shape mismatch for model {model}")
        if not np.array_equal(model_true, true):
            raise ValueError(f"Final-test ground truth differs for model {model}")

        model_frame = pd.DataFrame(
            {
                "source_index": [int(index) for index in source_indices],
                "text": [str(text) for text in texts],
                "model": str(model),
            }
        )
        for label_index, label in enumerate(LABEL_COLUMNS):
            true_values = true[:, label_index].astype(int)
            pred_values = pred[:, label_index].astype(int)
            wide_df[f"{model}_pred_{label}"] = pred_values
            wide_df[f"{model}_correct_{label}"] = (true_values == pred_values).astype(int)
            model_frame[f"true_{label}"] = true_values
            model_frame[f"pred_{label}"] = pred_values
            model_frame[f"correct_{label}"] = (true_values == pred_values).astype(int)
        exact_match = np.all(true == pred, axis=1).astype(int)
        wide_df[f"{model}_exact_match"] = exact_match
        model_frame["exact_match"] = exact_match
        long_frames.append(model_frame)

    wide_path = root / "final_test_data_with_all_predictions.csv"
    long_path = root / "final_test_predictions_all_models_long.csv"
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    pd.concat(long_frames, ignore_index=True).to_csv(long_path, index=False, encoding="utf-8-sig")
    return {"wide": str(wide_path), "long": str(long_path)}


def export_fold_result(row: Dict[str, object], output_dir: str) -> None:
    """Persist one completed run in both model-first and fold-first layouts."""
    model = str(row["model"])
    fold = int(row["fold"])
    fold_name = f"fold_{fold:02d}"
    root = Path(output_dir)
    destinations = [
        root / "by_model" / model / fold_name,
        root / "by_fold" / fold_name / model,
    ]
    serializable = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([serializable]).to_csv(destination / "metrics.csv", index=False, encoding="utf-8")
        (destination / "metrics.json").write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _export_grouped_views(df: pd.DataFrame, output_dir: str) -> None:
    """Export aggregate tables grouped by model and by fold."""
    root = Path(output_dir)
    numeric_metrics = [
        column for column in df.select_dtypes(include="number").columns
        if column != "fold"
    ]
    for model, model_df in df.groupby("model", sort=True):
        model_dir = root / "by_model" / str(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_df.sort_values("fold").to_csv(model_dir / "all_folds.csv", index=False, encoding="utf-8")
        summary = model_df[numeric_metrics].agg(["mean", "std"]).transpose().reset_index()
        summary.columns = ["metric", "mean", "std"]
        summary.to_csv(model_dir / "summary.csv", index=False, encoding="utf-8")

    for fold, fold_df in df.groupby("fold", sort=True):
        fold_dir = root / "by_fold" / f"fold_{int(fold):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_df.sort_values("model").to_csv(fold_dir / "all_models.csv", index=False, encoding="utf-8")


def export_results(
    rows: List[Dict[str, float]],
    output_dir: str,
    append: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No completed model/fold results were available to export")

    # Staged runs (for example ML today and transformers tomorrow) must retain
    # the rows produced by earlier stages. A newly produced model/fold replaces
    # an older copy of that same model/fold, which also makes reruns safe.
    out_csv = os.path.join(output_dir, "model_results_detailed.csv")
    if append and os.path.exists(out_csv):
        previous = pd.read_csv(out_csv)
        df = pd.concat([previous, df], ignore_index=True, sort=False)
        if {"model", "fold"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["model", "fold"], keep="last")
        df = df.sort_values([c for c in ["model", "fold"] if c in df.columns])

    for row in rows:
        export_fold_result(row, output_dir)
    _export_grouped_views(df, output_dir)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    compare = (
        df.groupby("model", as_index=False)
        .agg(
            subset_accuracy_mean=("subset_accuracy", "mean"),
            subset_accuracy_std=("subset_accuracy", "std"),
            hamming_score_mean=("hamming_score", "mean"),
            hamming_score_std=("hamming_score", "std"),
            precision_micro_mean=("precision_micro", "mean"),
            precision_micro_std=("precision_micro", "std"),
            recall_micro_mean=("recall_micro", "mean"),
            recall_micro_std=("recall_micro", "std"),
            f1_micro_mean=("f1_micro", "mean"),
            f1_micro_std=("f1_micro", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", "std"),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            train_time_sec_mean=("train_time_sec", "mean"),
            infer_time_sec_mean=("infer_time_sec", "mean"),
        )
        .sort_values("f1_macro_mean", ascending=False)
    )

    compare_path = os.path.join(output_dir, "model_comparison_macro_micro.csv")
    compare.to_csv(compare_path, index=False, encoding="utf-8")

    best_fold_df = (
        df.sort_values(["model", "f1_macro", "f1_micro", "subset_accuracy"], ascending=[True, False, False, False])
        .groupby("model", as_index=False)
        .head(1)
        .rename(columns={
            "fold": "best_fold",
            "subset_accuracy": "best_subset_accuracy",
            "hamming_score": "best_hamming_score",
            "precision_micro": "best_precision_micro",
            "recall_micro": "best_recall_micro",
            "f1_micro": "best_f1_micro",
            "precision_macro": "best_precision_macro",
            "recall_macro": "best_recall_macro",
            "f1_macro": "best_f1_macro",
        })
    )

    keep_cols = [
        "model",
        "best_fold",
        "best_subset_accuracy",
        "best_hamming_score",
        "best_precision_micro",
        "best_recall_micro",
        "best_f1_micro",
        "best_precision_macro",
        "best_recall_macro",
        "best_f1_macro",
        "artifact_dir",
        "temp_dir",
        "train_time_sec",
        "infer_time_sec",
    ]
    best_fold_df = best_fold_df[[c for c in keep_cols if c in best_fold_df.columns]].sort_values("best_f1_macro", ascending=False)
    best_fold_path = os.path.join(output_dir, "best_fold_per_model.csv")
    best_fold_df.to_csv(best_fold_path, index=False, encoding="utf-8")

    overall_best_df = compare[["model", "f1_macro_mean", "f1_micro_mean", "precision_macro_mean", "recall_macro_mean", "precision_micro_mean", "recall_micro_mean", "subset_accuracy_mean", "hamming_score_mean"]].copy()
    overall_best_df = overall_best_df.sort_values(["f1_macro_mean", "f1_micro_mean"], ascending=[False, False])
    overall_best_path = os.path.join(output_dir, "model_ranking_by_macro_micro_f1.csv")
    overall_best_df.to_csv(overall_best_path, index=False, encoding="utf-8")

    print(f"Saved detailed results: {out_csv}")
    print(f"Saved comparison table: {compare_path}")
    print(f"Saved best fold per model: {best_fold_path}")
    print(f"Saved model ranking table: {overall_best_path}")
    
    return compare_path
