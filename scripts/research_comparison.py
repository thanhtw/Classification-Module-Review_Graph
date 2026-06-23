#!/usr/bin/env python
"""
Research Paper Comparison: comprehensive multilabel benchmark

This module orchestrates comprehensive research paper generation by importing
specialized modules for different aspects of the comparison.
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from src.data.preprocessor import load_and_clean_data
from src.training.config import LABEL_COLUMNS

# Import organized research modules
from research_modules import (
    # Report builders
    generate_dataset_report,
    generate_model_configurations,
    generate_training_process_report,
    generate_research_paper_appendix,
    # Metrics analysis
    generate_fold_level_per_label_report,
    generate_per_label_metrics_report,
    generate_multilabel_metrics_report,
    # Visualizations
    generate_smote_visualization,
    generate_confusion_matrix_visualizations,
    generate_per_label_confusion_matrices,
    generate_model_comparison_visualizations,
    # Table generators
    generate_comprehensive_metrics_report,
    generate_detailed_comparison_table,
)


MODELS_TO_COMPARE = [
    ("Linear SVM", "linear_svm"),
    ("Logistic Regression", "logistic_regression"),
    ("Naive Bayes", "naive_bayes"),
    ("LSTM", "lstm"),
    ("BiLSTM", "bilstm"),
    ("BERT", "bert"),
    ("RoBERTa", "roberta"),
    ("gpt-5.2-codex (LLM, Zero-shot)", "llm_zero_shot"),
    ("gpt-5.2-codex (LLM, Few-shot k=10)", "llm_few_shot"),
]

MODEL_CATEGORIES = {
    "Machine Learning": ["Linear SVM", "Logistic Regression", "Naive Bayes"],
    "Deep Learning": ["LSTM", "BiLSTM"],
    "Transformers": ["BERT", "RoBERTa"],
    "LLM (OpenAI API)": ["gpt-5.2-codex (LLM, Zero-shot)", "gpt-5.2-codex (LLM, Few-shot k=10)"],
}
MODEL_KEY_TO_DISPLAY = {model_key: model_display for model_display, model_key in MODELS_TO_COMPARE}
ALL_MODEL_KEYS = [model_key for _, model_key in MODELS_TO_COMPARE]


def _run_full_training_pipeline(models_to_compare: list[tuple[str, str]], n_folds: int, seed: int) -> None:
    """Run the training pipeline once for all requested models."""
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--n_folds",
        str(n_folds),
        "--seed",
        str(seed),
        "--models",
        *[model_key for _, model_key in models_to_compare],
    ]
    print("Running full project training pipeline in one pass...")
    result = subprocess.run(cmd, cwd=project_root, timeout=36000)
    if result.returncode != 0:
        raise RuntimeError("Full training pipeline failed. See logs above for details.")


def _build_best_fold_results(
    results_df: pd.DataFrame,
    models_to_compare: list[tuple[str, str]],
    full_df: pd.DataFrame,
    folds_for_export: List[Dict[str, np.ndarray]],
) -> tuple[list[dict], list[dict], list[str]]:
    """Build best-fold comparison rows from the unified training results."""
    comparison_results: list[dict] = []
    feature_split_records: list[dict] = []
    failed_models: list[str] = []

    model_col = None
    for candidate in ["Model", "model", "model_key"]:
        if candidate in results_df.columns:
            model_col = candidate
            break

    if model_col is None:
        raise ValueError("Could not find model column in model_results_detailed.csv")

    feature_dir = project_root / "results" / "research_comparison" / "best_fold_feature_analysis"

    for model_display_name, model_key in models_to_compare:
        model_df = results_df[results_df[model_col] == model_key]
        if model_df.empty:
            failed_models.append(model_display_name)
            continue

        best_row = _select_best_fold_row(model_df)
        best_fold = int(best_row.get("fold", best_row.get("Fold", 1)))

        best_metrics = {
            "model": model_display_name,
            "model_key": model_key,
            "num_folds": len(model_df),
            "selected_fold": best_fold,
            "selection_rule": "f1_macro > f1_micro > subset_accuracy",
        }

        numeric_cols = model_df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [col for col in numeric_cols if col not in [model_col, "fold", "Fold"]]
        for col in numeric_cols:
            if col in best_row and pd.notna(best_row[col]):
                best_metrics[f"{col}_mean"] = float(best_row[col])
                best_metrics[f"{col}_std"] = 0.0

        artifact_dir = str(best_row.get("artifact_dir", ""))
        if artifact_dir:
            best_metrics["artifact_dir"] = artifact_dir
            best_metrics.update(_load_thresholds_for_best_fold(artifact_dir))

        train_split_path, test_split_path, train_size, test_size = _export_best_fold_split_files(
            data_df=full_df,
            folds=folds_for_export,
            model_key=model_key,
            best_fold=best_fold,
            output_dir=feature_dir,
        )
        best_metrics["best_fold_train_split"] = train_split_path
        best_metrics["best_fold_test_split"] = test_split_path
        best_metrics["best_fold_train_size"] = train_size
        best_metrics["best_fold_test_size"] = test_size

        feature_split_records.append(
            {
                "model": model_display_name,
                "model_key": model_key,
                "best_fold": best_fold,
                "train_size": train_size,
                "test_size": test_size,
                "train_split_csv": train_split_path,
                "test_split_csv": test_split_path,
            }
        )
        comparison_results.append(best_metrics)

    return comparison_results, feature_split_records, failed_models


def _load_thresholds_from_artifact(artifact_dir: str) -> np.ndarray | None:
    """Load per-label thresholds from a fold artifact directory."""
    if not artifact_dir:
        return None

    artifact_path = Path(artifact_dir)
    if not artifact_path.exists():
        return None

    # First preference: metadata.json generated by model runners.
    metadata_file = artifact_path / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            thresholds = meta.get("thresholds")
            if isinstance(thresholds, list) and len(thresholds) == len(LABEL_COLUMNS):
                return np.asarray(thresholds, dtype=float)
        except Exception:
            pass

    # Fallback for LSTM/BiLSTM where thresholds are tracked in training history.
    history_file = artifact_path / "training_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            thresholds = history.get("tuned_thresholds")
            if isinstance(thresholds, list) and len(thresholds) == len(LABEL_COLUMNS):
                return np.asarray(thresholds, dtype=float)
        except Exception:
            pass

    return None


def _aggregate_threshold_stats(model_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate threshold means/std across folds for one model."""
    threshold_rows = []
    if "artifact_dir" in model_df.columns:
        for artifact_dir in model_df["artifact_dir"].dropna().astype(str).tolist():
            th = _load_thresholds_from_artifact(artifact_dir)
            if th is not None:
                threshold_rows.append(th)

    out: Dict[str, float] = {}
    if not threshold_rows:
        return out

    threshold_mat = np.vstack(threshold_rows)
    for i, label in enumerate(LABEL_COLUMNS):
        out[f"threshold_{label}_mean"] = float(threshold_mat[:, i].mean())
        out[f"threshold_{label}_std"] = float(threshold_mat[:, i].std()) if threshold_mat.shape[0] > 1 else 0.0

    return out


def _load_thresholds_for_best_fold(artifact_dir: str) -> Dict[str, float]:
    """Load per-label thresholds for one selected best fold."""
    th = _load_thresholds_from_artifact(artifact_dir)
    if th is None:
        return {}
    out: Dict[str, float] = {}
    for i, label in enumerate(LABEL_COLUMNS):
        out[f"threshold_{label}_mean"] = float(th[i])
        out[f"threshold_{label}_std"] = 0.0
    return out


def _make_folds_for_export(
    n_samples: int,
    n_folds: int,
    test_size: float,
    seed: int,
    labels: np.ndarray | None = None,
) -> List[Dict[str, np.ndarray]]:
    """Replicate train.py fold logic so best-fold train/test data can be exported."""
    idx_local = np.arange(n_samples)
    fold_list: List[Dict[str, np.ndarray]] = []
    strat_targets = None
    if labels is not None:
        strat_targets = np.array(["".join(row.astype(str).tolist()) for row in labels])

    if n_folds >= 2:
        if strat_targets is not None:
            try:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                for tr_idx, te_idx in skf.split(idx_local, strat_targets):
                    fold_list.append({"train_idx": tr_idx, "test_idx": te_idx})
                return fold_list
            except ValueError:
                pass

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(idx_local):
            fold_list.append({"train_idx": tr_idx, "test_idx": te_idx})
    else:
        holdout_stratify = strat_targets if strat_targets is not None else None
        try:
            tr_idx, te_idx = train_test_split(
                idx_local,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=holdout_stratify,
            )
        except ValueError:
            tr_idx, te_idx = train_test_split(
                idx_local,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
            )
        fold_list.append({"train_idx": tr_idx, "test_idx": te_idx})
    return fold_list


def _export_best_fold_split_files(
    data_df: pd.DataFrame,
    folds: List[Dict[str, np.ndarray]],
    model_key: str,
    best_fold: int,
    output_dir: Path,
) -> tuple[str, str, int, int]:
    """Export train/test rows for selected best fold to support feature analysis."""
    fold_idx = int(best_fold) - 1
    if fold_idx < 0 or fold_idx >= len(folds):
        raise ValueError(f"Best fold {best_fold} out of range for {model_key}")

    train_idx = folds[fold_idx]["train_idx"]
    test_idx = folds[fold_idx]["test_idx"]

    train_df = data_df.iloc[train_idx].copy()
    test_df = data_df.iloc[test_idx].copy()
    train_df.insert(0, "source_index", train_df.index.astype(int))
    test_df.insert(0, "source_index", test_df.index.astype(int))

    model_out_dir = output_dir / f"{model_key}_fold_{best_fold}"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    train_path = model_out_dir / "train_split.csv"
    test_path = model_out_dir / "test_split.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return str(train_path), str(test_path), int(len(train_df)), int(len(test_df))


def _select_best_fold_row(model_df: pd.DataFrame) -> pd.Series:
    """Select best fold by evaluation priority: f1_macro > f1_micro > subset_accuracy."""
    return model_df.sort_values(
        ["f1_macro", "f1_micro", "subset_accuracy"],
        ascending=[False, False, False],
    ).iloc[0]


def _export_txt_report(sorted_results: list[dict], output_path: Path) -> None:
    """Export a plain-text best-fold comparison report with accuracy and thresholds."""
    lines = []
    lines.append("BEST FOLD MODEL COMPARISON REPORT")
    lines.append("=" * 120)
    lines.append(
        f"{'Rank':<5} | {'Model':<30} | {'Fold':<6} | {'Acc-Micro':<10} | {'Acc-Macro':<10} | {'F1-Macro':<10} | {'F1-Micro':<10}"
    )
    lines.append("-" * 120)

    for rank, result in enumerate(sorted_results, 1):
        model = result.get("model", "")[:30]
        fold = int(result.get("selected_fold", 0))
        acc_micro = result.get("accuracy_micro_mean", 0.0)
        acc_macro = result.get("accuracy_macro_mean", 0.0)
        f1_macro = result.get("f1_macro_mean", 0.0)
        f1_micro = result.get("f1_micro_mean", 0.0)
        lines.append(
            f"{rank:<5} | {model:<30} | {fold:<6} | {acc_micro:<10.4f} | {acc_macro:<10.4f} | {f1_macro:<10.4f} | {f1_micro:<10.4f}"
        )

    lines.append("\nPER-LABEL THRESHOLDS (mean ± std)")
    lines.append("-" * 120)
    for result in sorted_results:
        model = result.get("model", "")
        threshold_parts = []
        for label in LABEL_COLUMNS:
            mean_key = f"threshold_{label}_mean"
            std_key = f"threshold_{label}_std"
            if mean_key in result:
                threshold_parts.append(f"{label}={result[mean_key]:.3f}±{result.get(std_key, 0.0):.3f}")

        if threshold_parts:
            lines.append(f"{model}: " + ", ".join(threshold_parts))
        else:
            lines.append(f"{model}: n/a")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_research_comparison(n_folds=10, seed=42, selected_model_keys=None):
    """Run comprehensive comparison: ML → DL → Transformers → LLM, then compare all"""
    if selected_model_keys is None:
        selected_model_keys = list(ALL_MODEL_KEYS)

    selected_model_keys = [model_key for model_key in selected_model_keys if model_key in MODEL_KEY_TO_DISPLAY]
    if not selected_model_keys:
        raise ValueError("No valid model keys were selected for research comparison.")

    models_to_compare = [
        (MODEL_KEY_TO_DISPLAY[model_key], model_key)
        for model_key in selected_model_keys
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON: All Models vs LLM (Zero-shot & Few-shot)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - LLM Model: gpt-5.2-codex (OpenAI API)")
    print(f"  - LLM Approaches: Zero-shot + Few-shot (k=10)")
    print(f"  - Selected Models: {', '.join(selected_model_keys)}")
    print(f"  - Folds: {n_folds}")
    print(f"  - Seed: {seed}")
    print(f"  - Task: Multilabel Classification (3 labels)")
    print(f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80 + "\n")

    full_df = load_and_clean_data("data/cleaned_3label_data.csv")
    all_labels = full_df[LABEL_COLUMNS].values.astype(int)
    folds_for_export = _make_folds_for_export(
        n_samples=len(full_df),
        n_folds=n_folds,
        test_size=0.2,
        seed=seed,
        labels=all_labels,
    )
    _run_full_training_pipeline(models_to_compare=models_to_compare, n_folds=n_folds, seed=seed)

    results_base = project_root / "results" / "modular_multimodel"
    detailed_results = results_base / "model_results_detailed.csv"
    if not detailed_results.exists():
        raise FileNotFoundError(f"Results file not found after training: {detailed_results}")

    df = pd.read_csv(detailed_results)
    comparison_results, feature_split_records, failed_models = _build_best_fold_results(
        results_df=df,
        models_to_compare=models_to_compare,
        full_df=full_df,
        folds_for_export=folds_for_export,
    )

    for result in comparison_results:
        print(f"\n✓ {result['model']} Complete")
        print(f"  - Folds: {result.get('num_folds', 0)}")
        print(f"  - Selected best fold: {result.get('selected_fold', 0)}")
        print(f"  - F1-macro (best fold): {result.get('f1_macro_mean', 0.0):.4f}")
    
    # Generate comprehensive comparison report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80 + "\n")
    
    if comparison_results:
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        results_dir = Path("results/research_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = results_dir / "all_models_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to: {csv_path}\n")
        best_csv_path = results_dir / "best_fold_model_comparison.csv"
        comparison_df.to_csv(best_csv_path, index=False)
        print(f"✓ Best-fold comparison saved to: {best_csv_path}\n")

        if feature_split_records:
            feature_summary_df = pd.DataFrame(feature_split_records)
            feature_summary_csv = results_dir / "best_fold_feature_analysis" / "best_fold_split_summary.csv"
            feature_summary_df.to_csv(feature_summary_csv, index=False)
            feature_summary_txt = results_dir / "best_fold_feature_analysis" / "best_fold_split_summary.txt"
            summary_lines = [
                "BEST FOLD TRAIN/TEST SPLIT FILES",
                "=" * 120,
                f"{'Model':<30} | {'Fold':<6} | {'Train':<8} | {'Test':<8} | {'Train CSV':<30} | {'Test CSV':<30}",
                "-" * 120,
            ]
            for r in feature_split_records:
                summary_lines.append(
                    f"{r['model'][:30]:<30} | {r['best_fold']:<6} | {r['train_size']:<8} | {r['test_size']:<8} | "
                    f"{r['train_split_csv'][:30]:<30} | {r['test_split_csv'][:30]:<30}"
                )
            feature_summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")
            print(f"✓ Feature split summary CSV: {feature_summary_csv}")
            print(f"✓ Feature split summary TXT: {feature_summary_txt}\n")
        
        # Display comprehensive comparison table
        print("=" * 180)
        print(f"{'Rank':<5} | {'Model':<30} | {'Fold':<6} | {'Acc-Micro':<12} | {'Acc-Macro':<12} | {'F1-Macro':<15} | {'F1-Micro':<15} | {'Prec-Macro':<15} | {'Recall-Macro':<15}")
        print("=" * 180)
        
        # Sort by F1-macro descending
        sorted_results = sorted(comparison_results, key=lambda x: x.get('f1_macro_mean', 0), reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            model_name = result['model'][:28]
            fold = int(result.get('selected_fold', 0))
            acc_micro = f"{result.get('accuracy_micro_mean', 0):.4f}±{result.get('accuracy_micro_std', 0):.4f}"
            acc_macro = f"{result.get('accuracy_macro_mean', 0):.4f}±{result.get('accuracy_macro_std', 0):.4f}"
            f1_macro = f"{result.get('f1_macro_mean', 0):.4f}±{result.get('f1_macro_std', 0):.4f}"
            f1_micro = f"{result.get('f1_micro_mean', 0):.4f}±{result.get('f1_micro_std', 0):.4f}"
            prec_macro = f"{result.get('precision_macro_mean', 0):.4f}±{result.get('precision_macro_std', 0):.4f}"
            rec_macro = f"{result.get('recall_macro_mean', 0):.4f}±{result.get('recall_macro_std', 0):.4f}"
            print(f"{rank:<5} | {model_name:<30} | {fold:<6} | {acc_micro:<12} | {acc_macro:<12} | {f1_macro:<15} | {f1_micro:<15} | {prec_macro:<15} | {rec_macro:<15}")
        
        print("=" * 180)

        txt_report_path = results_dir / "all_models_comparison_report.txt"
        _export_txt_report(sorted_results, txt_report_path)
        print(f"✓ Text report saved to: {txt_report_path}")
        
        # Model category summary
        print("\n" + "=" * 80)
        print("SUMMARY BY CATEGORY")
        print("=" * 80 + "\n")
        
        for category, model_names in MODEL_CATEGORIES.items():
            category_results = [r for r in sorted_results if any(name in r['model'] for name in model_names)]
            if category_results:
                best = category_results[0]
                print(f"{category}:")
                print(f"  Best: {best['model']}")
                print(f"  F1-macro: {best.get('f1_macro_mean', 0):.4f} ± {best.get('f1_macro_std', 0):.4f}")
                print(f"  F1-micro: {best.get('f1_micro_mean', 0):.4f} ± {best.get('f1_micro_std', 0):.4f}")
                print(f"  Accuracy-micro: {best.get('accuracy_micro_mean', 0):.4f}")
                print(f"  Accuracy-macro: {best.get('accuracy_macro_mean', 0):.4f}")
                print(f"  Precision-macro: {best.get('precision_macro_mean', 0):.4f}")
                print(f"  Precision-micro: {best.get('precision_micro_mean', 0):.4f}")
                print(f"  Recall-macro: {best.get('recall_macro_mean', 0):.4f}")
                print(f"  Recall-micro: {best.get('recall_micro_mean', 0):.4f}")
                print()
        
        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS & REPORTS FOR ALL MODELS")
        print("=" * 80 + "\n")
        
        smote_vis = generate_smote_visualization(results_dir)
        generate_model_comparison_visualizations(comparison_results, results_dir)
        cm_vis = generate_confusion_matrix_visualizations(results_dir)
        per_label_cm_dir = generate_per_label_confusion_matrices(results_dir)
        comp_csv = generate_comprehensive_metrics_report(comparison_results, results_dir)
        complete_csv, key_metrics_csv, latex_file = generate_detailed_comparison_table(comparison_results, results_dir)
        fold_level_per_label_report = generate_fold_level_per_label_report(
            artifact_root=results_base / "model_artifacts",
            output_dir=results_dir,
            model_keys=[model_key for _, model_key in models_to_compare],
            model_display_names={model_key: model_display_name for model_display_name, model_key in models_to_compare},
            filename_prefix="all_models_per_label",
        )
        per_label_report = generate_per_label_metrics_report(comparison_results, results_dir)
        multilabel_report = generate_multilabel_metrics_report(comparison_results, results_dir)
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATION & REPORT FILES GENERATED")
        print("=" * 80)
        
        print("\n📊 COMPREHENSIVE VISUALIZATIONS (Publication-Ready 300 DPI):")
        if smote_vis:
            print(f"  ✓ {smote_vis}")
        print(f"  ✓ {results_dir}/model_f1_comparison.png")
        print(f"  ✓ {results_dir}/model_multilabel_metrics.png")
        if cm_vis:
            print(f"  ✓ {cm_vis}")
        if per_label_cm_dir:
            print(f"  ✓ Per-label confusion matrices: {per_label_cm_dir}/")
        
        print("\n📋 COMPREHENSIVE DATA TABLES & REPORTS:")
        print(f"  ✓ {comp_csv}")
        print(f"  ✓ {results_dir}/per_model_metrics/")
        print(f"  ✓ {complete_csv}")
        print(f"  ✓ {key_metrics_csv}")
        print(f"  ✓ {latex_file}")
        print(f"  ✓ {fold_level_per_label_report['fold_level_csv']}")
        print(f"  ✓ {fold_level_per_label_report['summary_csv']}")
        print(f"  ✓ {fold_level_per_label_report['report_txt']}")
        print(f"  ✓ {results_dir}/per_label_metrics_report.json")
        print(f"  ✓ {results_dir}/per_label_metrics_report.txt")
        print(f"  ✓ {results_dir}/multilabel_metrics_report.json")
        
        print("\n✨ Reports generated successfully!")
        print("=" * 80)
    else:
        print("\n✗ No results to compare!")
        if failed_models:
            print(f"\nFailed models: {', '.join(failed_models)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research paper comparison: LLM vs Fine-tuned Transformers")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_KEYS,
        default=None,
        help="Optional subset of model keys to run",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Run only llm_zero_shot and llm_few_shot, then generate the same metrics reports",
    )
    
    args = parser.parse_args()

    selected_models = ["llm_zero_shot", "llm_few_shot"] if args.llm_only else args.models
    
    # Generate comprehensive research paper documentation
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE RESEARCH PAPER DOCUMENTATION")
    print("=" * 80)
    
    output_dir = "results/research_comparison"
    generate_research_paper_appendix(output_dir=output_dir)
    
    # Run model comparison
    run_research_comparison(n_folds=args.n_folds, seed=args.seed, selected_model_keys=selected_models)
