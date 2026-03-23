"""Module for generating visualizations (SMOTE, confusion matrices, training curves, heatmaps)"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.config import LABEL_COLUMNS
from src.utils.metrics import compute_metrics
from sklearn.metrics import confusion_matrix


def _thousands_formatter(x, _):
    return f"{int(x):,}"


def _short_model_name(name: str) -> str:
    """Normalize long model names for cleaner figure labels."""
    aliases = {
        "llama-3.1-8b-instant (LLM, Zero-shot)": "LLM Zero-shot",
        "llama-3.1-8b-instant (LLM, Few-shot k=100)": "LLM Few-shot",
        "Linear SVM": "Linear SVM",
        "Logistic Regression": "Logistic Reg.",
        "Naive Bayes": "Naive Bayes",
        "RoBERTa": "RoBERTa",
    }
    return aliases.get(name, name)


def _load_best_fold_map() -> dict:
    """Load model -> selected best fold mapping from comparison outputs.

    Priority:
    1) results/research_comparison/best_fold_model_comparison.csv
    2) results/research_comparison/all_models_comparison.csv
    3) results/modular_multimodel/best_fold_per_model.csv
    """
    alias_to_key = {
        "linear svm": "linear_svm",
        "logistic regression": "logistic_regression",
        "naive bayes": "naive_bayes",
        "lstm": "lstm",
        "bilstm": "bilstm",
        "bert": "bert",
        "roberta": "roberta",
        "llama-3.1-8b-instant (llm, zero-shot)": "llm_zero_shot",
        "llama-3.1-8b-instant (llm, few-shot k=100)": "llm_few_shot",
    }

    candidate_csvs = [
        Path("results/research_comparison/best_fold_model_comparison.csv"),
        Path("results/research_comparison/all_models_comparison.csv"),
        Path("results/modular_multimodel/best_fold_per_model.csv"),
    ]

    mapping = {}
    for csv_path in candidate_csvs:
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        fold_col = None
        for candidate in ["selected_fold", "best_fold", "fold", "Fold"]:
            if candidate in df.columns:
                fold_col = candidate
                break
        if fold_col is None:
            continue

        has_model_key = "model_key" in df.columns
        has_model_name = "model" in df.columns or "Model" in df.columns
        model_name_col = "model" if "model" in df.columns else ("Model" if "Model" in df.columns else None)

        for _, row in df.iterrows():
            try:
                fold_num = int(row[fold_col])
            except Exception:
                continue

            candidate_keys = []
            if has_model_key:
                candidate_keys.append(str(row["model_key"]).strip())
            if has_model_name and model_name_col is not None:
                raw_name = str(row[model_name_col]).strip()
                candidate_keys.append(raw_name)
                candidate_keys.append(alias_to_key.get(raw_name.lower(), ""))

            for key in candidate_keys:
                if not key:
                    continue
                # Keep first source hit per key to preserve file priority.
                mapping.setdefault(key, fold_num)

    return mapping


def _to_binary_label_matrix(arr: np.ndarray, n_labels: int) -> np.ndarray:
    """Convert model outputs to binary multilabel matrix with shape (n_samples, n_labels)."""
    matrix = np.asarray(arr)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, n_labels)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {matrix.shape}")
    if matrix.shape[1] != n_labels:
        raise ValueError(f"Expected {n_labels} labels, got shape {matrix.shape}")

    # If predictions are logits/probabilities, threshold at 0.5.
    if matrix.dtype.kind in {"f", "c"}:
        return (matrix > 0.5).astype(int)
    return (matrix >= 1).astype(int)


def _build_3x3_label_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> np.ndarray:
    """Build a multilabel-aware 3x3 label confusion matrix.

    For each sample, every true-positive label contributes an integer count to
    each predicted-positive label. This keeps matrix entries as integer counts.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    cm = np.zeros((n_labels, n_labels), dtype=int)
    for s in range(y_true.shape[0]):
        true_idx = np.where(y_true[s] == 1)[0]
        pred_idx = np.where(y_pred[s] == 1)[0]

        if true_idx.size == 0 or pred_idx.size == 0:
            continue

        for i in true_idx:
            for j in pred_idx:
                cm[i, j] += 1

    return cm


def _get_model_fold_dir(model_dir: Path, best_fold_map: dict) -> Path | None:
    """Pick fold directory from best fold map; fallback to first available fold."""
    model_name = model_dir.name
    fold_dirs = sorted([d for d in model_dir.glob("fold_*") if d.is_dir()])
    if not fold_dirs:
        return None

    best_fold = best_fold_map.get(model_name)
    if best_fold is not None:
        candidate = model_dir / f"fold_{best_fold}"
        if candidate.exists():
            return candidate

    return fold_dirs[0]


def load_smote_analysis(results_dir="results/modular_multimodel/global_train_data_analysis"):
    """Load SMOTE analysis data from results directory"""
    smote_file = Path(results_dir) / "train_smote_analysis_summary.json"
    
    if smote_file.exists():
        with open(smote_file, 'r') as f:
            return json.load(f)
    return None


def generate_smote_visualization(output_dir="results/research_comparison"):
    """Generate SMOTE class distribution visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    smote_data = load_smote_analysis()
    if not smote_data:
        print("⚠ SMOTE analysis file not found")
        return None
    
    labels = LABEL_COLUMNS
    before_pos = smote_data.get('label_pos_before', [])
    before_neg = smote_data.get('label_neg_before', [])
    after_pos = smote_data.get('label_pos_after', [])
    after_neg = smote_data.get('label_neg_after', [])

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, len(labels), figsize=(16, 4.8), constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]

    fig.suptitle("SMOTE Effect on Label Distribution", fontsize=14, fontweight="semibold")
    max_count = max(before_pos + before_neg + after_pos + after_neg)

    color_map = {
        "Before Negative": "#8FA3B8",
        "Before Positive": "#4C6A92",
        "After Negative": "#C9B37E",
        "After Positive": "#8F6D1E",
    }

    for idx, label in enumerate(labels):
        ax = axes[idx]
        categories = ["Before Negative", "Before Positive", "After Negative", "After Positive"]
        values = [before_neg[idx], before_pos[idx], after_neg[idx], after_pos[idx]]
        colors = [color_map[c] for c in categories]

        bars = ax.bar(categories, values, color=colors, edgecolor="#444444", linewidth=0.8)
        ax.set_title(label.capitalize(), fontweight="semibold")
        ax.set_ylim(0, max_count * 1.14)
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", rotation=24)

        before_total = before_neg[idx] + before_pos[idx]
        after_total = after_neg[idx] + after_pos[idx]
        before_rate = (before_pos[idx] / before_total) if before_total else 0.0
        after_rate = (after_pos[idx] / after_total) if after_total else 0.0
        ax.text(
            0.02,
            0.98,
            f"Pos. rate: {before_rate:.1%} -> {after_rate:.1%}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color="#2F3A4A",
            bbox=dict(facecolor="#F6F7F9", edgecolor="#D4D9E0", boxstyle="round,pad=0.25"),
        )

        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:,}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1E2A36",
            )

    axes[0].set_ylabel("Sample Count")
    
    # Save figure
    smote_vis_path = output_dir / "smote_impact_visualization.png"
    plt.savefig(smote_vis_path, dpi=300, bbox_inches='tight')
    print(f"✓ SMOTE visualization saved to: {smote_vis_path}")
    plt.close()
    
    return smote_vis_path


def generate_confusion_matrix_visualizations(output_dir="results/research_comparison"):
    """Generate confusion matrix heatmaps for ALL models"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Generating true 3×3 confusion matrices for ALL models...")
    
    model_artifacts_dir = Path("results/modular_multimodel/model_artifacts")
    if not model_artifacts_dir.exists():
        print("⚠ Model artifacts directory not found - skipping confusion matrix visualization")
        return None
    
    # Collect all models
    model_dirs = sorted([d for d in model_artifacts_dir.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print("⚠ No model directories found")
        return None
    
    num_models = len(model_dirs)
    cols = 3
    rows = (num_models + cols - 1) // cols  # Ceil division for grid layout
    
    confusion_fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape((rows, cols))
    
    confusion_fig.suptitle(f'Confusion Matrices: All {num_models} Models', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    best_fold_map = _load_best_fold_map()
    n_labels = len(LABEL_COLUMNS)
    success_count = 0
    for model_idx, model_dir in enumerate(model_dirs):
        model_name = model_dir.name
        
        try:
            fold_dir = _get_model_fold_dir(model_dir, best_fold_map)
            if fold_dir is None:
                print(f"  ⚠ {model_name}: no fold directories")
                continue

            pred_file = fold_dir / "predictions.npy"
            label_file = fold_dir / "labels.npy"
            if not pred_file.exists() or not label_file.exists():
                print(f"  ⚠ {model_name}: missing predictions.npy/labels.npy")
                continue

            y_pred = _to_binary_label_matrix(np.load(pred_file), n_labels)
            y_true = _to_binary_label_matrix(np.load(label_file), n_labels)
            if y_true.shape != y_pred.shape:
                print(f"  ⚠ {model_name}: shape mismatch {y_true.shape} vs {y_pred.shape}")
                continue

            cm = _build_3x3_label_confusion(y_true, y_pred, n_labels)
            metrics = compute_metrics(y_true, y_pred)

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[model_idx],
                cbar=True,
                xticklabels=LABEL_COLUMNS,
                yticklabels=LABEL_COLUMNS,
                vmin=0,
                linewidths=0.5,
                linecolor='black',
                square=True,
            )
            axes[model_idx].set_title(
                f"{model_name}\n"
                f"Acc-Micro={metrics.get('accuracy_micro', 0.0):.3f}, "
                f"Acc-Macro={metrics.get('accuracy_macro', 0.0):.3f}",
                fontsize=10,
                fontweight='bold',
            )
            axes[model_idx].set_ylabel('True Label', fontweight='bold', fontsize=9)
            axes[model_idx].set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
            success_count += 1
            print(f"  ✓ {model_name}")
        except Exception as e:
            print(f"  ⚠ {model_name}: {e}")
    
    # Remove unused subplots
    for idx in range(num_models, len(axes)):
        confusion_fig.delaxes(axes[idx])
    
    if success_count > 0:
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrices_all_models.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrices for {success_count} models saved to: {cm_path}")
        plt.close()
        return cm_path
    else:
        plt.close()
        print("✗ No confusion matrices could be generated")
        return None


def generate_per_label_confusion_matrices(output_dir="results/research_comparison"):
    """Generate per-label 2x2 confusion matrices styled for presentation.

    Output: one figure per model (best fold), with 3 panels (one per label).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating per-label 2×2 confusion matrices...")
    
    model_artifacts_dir = Path("results/modular_multimodel/model_artifacts")
    if not model_artifacts_dir.exists():
        print("⚠ Model artifacts directory not found")
        return
    
    model_dirs = sorted([d for d in model_artifacts_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        print("⚠ No model directories found")
        return
    
    labels = LABEL_COLUMNS
    num_labels = len(labels)
    best_fold_map = _load_best_fold_map()
    
    # Create subdirectory for confusion matrices
    cm_out_dir = output_dir / "per_label_confusion_matrices"
    cm_out_dir.mkdir(parents=True, exist_ok=True)
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        fig = None
        try:
            fold_path = _get_model_fold_dir(model_dir, best_fold_map)
            if fold_path is None:
                print(f"  ⚠ {model_name}: no fold directory found")
                continue

            pred_file = fold_path / "predictions.npy"
            label_file = fold_path / "labels.npy"
            if not pred_file.exists() or not label_file.exists():
                print(f"  ⚠ {model_name}: predictions.npy/labels.npy not found in {fold_path.name}")
                continue

            y_pred = _to_binary_label_matrix(np.load(pred_file), num_labels)
            y_true = _to_binary_label_matrix(np.load(label_file), num_labels)
            if y_true.shape != y_pred.shape:
                print(f"  ⚠ {model_name}: shape mismatch {y_true.shape} vs {y_pred.shape}")
                continue
            fig, axes = plt.subplots(1, num_labels, figsize=(4.2 * num_labels + 1.2, 4.8))
            if num_labels == 1:
                axes = [axes]

            fold_num = fold_path.name.replace("fold_", "")

            for idx, label in enumerate(labels):
                ax = axes[idx]
                # Matrix layout with labels=[1, 0]: [[TP, FN], [FP, TN]]
                cm_label = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[1, 0])
                tp = int(cm_label[0, 0])
                fn = int(cm_label[0, 1])
                fp = int(cm_label[1, 0])
                tn = int(cm_label[1, 1])

                annot = np.array(
                    [[f"{cm_label[r, c]}" for c in range(2)] for r in range(2)]
                )

                sns.heatmap(
                    cm_label,
                    annot=annot,
                    fmt='',
                    cmap='Blues',
                    cbar=False,
                    linewidths=1.0,
                    linecolor='white',
                    square=True,
                    ax=ax,
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                )

                ax.set_xticklabels(['Pred: Positive', 'Pred: Negative'], fontsize=9)
                ax.set_yticklabels(['True: Positive', 'True: Negative'], fontsize=9, rotation=0)
                ax.tick_params(top=False, bottom=True, left=True, right=False)
                ax.set_xlabel('Predicted label', fontsize=9, fontweight='bold')
                ax.set_ylabel('True label', fontsize=9, fontweight='bold')
                ax.set_title(f"{label.capitalize()}", fontsize=12, pad=8)

            plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.98])
            cm_file = cm_out_dir / f"confusion_matrix_3labels_{model_name}_{fold_path.name}.png"
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ {model_name}: {cm_file}")
            plt.close()
        except Exception as e:
            print(f"  ⚠ {model_name}: {str(e)[:100]}")
            if fig is not None:
                plt.close(fig)
    
    print(f"✓ Per-label confusion matrices saved to: {cm_out_dir}")
    return cm_out_dir


def generate_model_comparison_visualizations(comparison_results, output_dir="results/research_comparison"):
    """Generate model performance comparison visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not comparison_results:
        print("⚠ No comparison results to visualize")
        return
    
    # Sort by F1-macro
    sorted_results = sorted(comparison_results, key=lambda x: x.get('f1_macro_mean', 0), reverse=True)
    
    # Prepare data
    models = [_short_model_name(r['model']) for r in sorted_results]
    f1_macro = [r.get('f1_macro_mean', 0) for r in sorted_results]
    f1_micro = [r.get('f1_micro_mean', 0) for r in sorted_results]
    hamming_loss = [r.get('hamming_loss_mean', 0) for r in sorted_results]
    subset_acc = [r.get('subset_accuracy_mean', 0) for r in sorted_results]

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    # Figure 1: Horizontal dumbbell chart for F1 comparison
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    for i in range(len(models)):
        ax.plot(
            [f1_macro[i], f1_micro[i]],
            [y[i], y[i]],
            color="#AAB4C2",
            linewidth=2,
            zorder=1,
        )

    ax.scatter(f1_macro, y, color="#1F5A8A", s=46, label="F1-Macro", zorder=3)
    ax.scatter(f1_micro, y, color="#2A9D8F", s=46, label="F1-Micro", zorder=3)

    for i in range(len(models)):
        ax.text(f1_macro[i] - 0.008, y[i] + 0.16, f"{f1_macro[i]:.3f}", ha="right", va="center", fontsize=8)
        ax.text(f1_micro[i] + 0.008, y[i] - 0.16, f"{f1_micro[i]:.3f}", ha="left", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0.45, 1.00)
    ax.set_xlabel("Score")
    ax.set_title("Model Ranking by F1-Macro and F1-Micro")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True)
    ax.invert_yaxis()

    plt.tight_layout()
    fig_path = output_dir / "model_f1_comparison.png"
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    print(f"✓ F1 comparison saved to: {fig_path}")
    plt.close()

    # Figure 2: Hamming loss and subset accuracy in paired publication panels
    metric_df = pd.DataFrame({
        "Model": models,
        "Subset Accuracy": subset_acc,
        "Hamming Loss": hamming_loss,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 6.2), sharey=True)

    subset_order = metric_df.sort_values("Subset Accuracy", ascending=True)
    hamming_order = metric_df.sort_values("Hamming Loss", ascending=False)

    bars_sa = ax1.barh(
        subset_order["Model"],
        subset_order["Subset Accuracy"],
        color="#2A9D8F",
        edgecolor="#2C3E50",
        linewidth=0.8,
    )
    ax1.set_title("Subset Accuracy (Higher is Better)")
    ax1.set_xlabel("Subset Accuracy")
    ax1.set_xlim(0.0, 1.0)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    bars_hl = ax2.barh(
        hamming_order["Model"],
        hamming_order["Hamming Loss"],
        color="#B56576",
        edgecolor="#2C3E50",
        linewidth=0.8,
    )
    ax2.set_title("Hamming Loss (Lower is Better)")
    ax2.set_xlabel("Hamming Loss")
    ax2.set_xlim(0.0, max(hamming_loss) * 1.12 if hamming_loss else 1.0)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars_sa:
        val = bar.get_width()
        ax1.text(min(val + 0.012, 0.985), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    for bar in bars_hl:
        val = bar.get_width()
        ax2.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    fig_path = output_dir / "model_multilabel_metrics.png"
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    print(f"✓ Multilabel metrics saved to: {fig_path}")
    plt.close()


def _main() -> None:
    """CLI for generating confusion matrices from saved fold artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix visualizations from saved predictions/labels (no retraining)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/research_comparison",
        help="Directory to save generated confusion matrix figures.",
    )
    parser.add_argument(
        "--all_models_only",
        action="store_true",
        help="Generate only the combined all-models confusion figure.",
    )
    parser.add_argument(
        "--per_model_only",
        action="store_true",
        help="Generate only per-model 3x3 confusion figures.",
    )

    args = parser.parse_args()
    if args.all_models_only and args.per_model_only:
        raise ValueError("Choose at most one of --all_models_only or --per_model_only")

    print("\nGenerating confusion matrices from existing artifacts (no retraining)...")
    if not args.per_model_only:
        generate_confusion_matrix_visualizations(output_dir=args.output_dir)
    if not args.all_models_only:
        generate_per_label_confusion_matrices(output_dir=args.output_dir)


if __name__ == "__main__":
    _main()
