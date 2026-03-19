import os
from typing import Dict, List

import pandas as pd

from .report_enhanced import generate_summary_report


def export_results(rows: List[Dict[str, float]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "model_results_detailed.csv")
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
    
    # Generate enhanced summary report
    generate_summary_report(df, output_dir)
    
    return compare_path
