#!/usr/bin/env python
"""Run a 10-fold training-data resampling sensitivity analysis.

The held-out fold is never resampled. For transformers, complete tokenized
training examples are duplicated with their original multilabel vectors;
categorical token IDs are not synthetically interpolated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RATIOS = (0.50, 0.75, 1.00)
SUMMARY_METRICS = (
    "f1_macro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "subset_accuracy",
    "hamming_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="10-fold SMOTE/resampling sensitivity analysis")
    parser.add_argument("--models", nargs="+", default=["bert"], choices=["bert", "roberta"])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/smote_sensitivity")
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(not 0.0 < ratio <= 1.0 for ratio in args.ratios):
        raise ValueError("Every ratio must be in the interval (0, 1]")

    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configurations = [("none", None)] + [(f"ratio_{ratio:.2f}", ratio) for ratio in args.ratios]
    frames = []

    for config_name, ratio in configurations:
        run_dir = output_root / config_name
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--models",
            *args.models,
            "--n_folds",
            "10",
            "--seed",
            str(args.seed),
            "--output_dir",
            str(run_dir),
            "--bert_epochs",
            str(args.epochs),
            "--roberta_epochs",
            str(args.epochs),
        ]
        if ratio is None:
            cmd.append("--no_smote")
        else:
            cmd.extend(["--smote_target_ratio", str(ratio)])

        print(f"Running 10-fold configuration: {config_name}", flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        fold_df = pd.read_csv(run_dir / "model_results_detailed.csv")
        observed_folds = fold_df.groupby("model")["fold"].nunique()
        if not (observed_folds == 10).all():
            raise RuntimeError(f"Expected 10 completed folds for every model; observed {observed_folds.to_dict()}")
        fold_df.insert(0, "resampling_configuration", config_name)
        fold_df.insert(1, "target_ratio", 0.0 if ratio is None else ratio)
        frames.append(fold_df)

    all_folds = pd.concat(frames, ignore_index=True)
    all_folds_path = output_root / "smote_sensitivity_all_folds.csv"
    all_folds.to_csv(all_folds_path, index=False)

    available_metrics = [metric for metric in SUMMARY_METRICS if metric in all_folds.columns]
    summary = (
        all_folds.groupby(["resampling_configuration", "target_ratio", "model"])[available_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary_path = output_root / "smote_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    protocol = {
        "cross_validation_folds": 10,
        "seed": args.seed,
        "models": args.models,
        "configurations": [name for name, _ in configurations],
        "resampling_scope": "training fold only",
        "transformer_method": "label-preserving random oversampling of complete encoded samples",
        "test_fold_resampled": False,
        "all_folds_csv": str(all_folds_path),
        "summary_csv": str(summary_path),
    }
    (output_root / "smote_sensitivity_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"Sensitivity summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
