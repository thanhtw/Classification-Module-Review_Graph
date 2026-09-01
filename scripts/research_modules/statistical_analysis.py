"""Paired statistical tests for cross-validation benchmark results."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


BENCHMARK_MODELS = (
    "bert",
    "roberta",
    "linear_svm",
    "logistic_regression",
    "naive_bayes",
    "lstm",
    "bilstm",
)
ANALYSIS_METRICS = (
    "precision_macro",
    "recall_macro",
    "accuracy_macro",
    "f1_macro",
)


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    family_size = len(values)
    for rank, index in enumerate(order):
        candidate = (family_size - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _matched_rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation; positive favors model A."""
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def _bootstrap_mean_difference_ci(
    differences: np.ndarray,
    seed: int,
    samples: int = 10000,
) -> tuple[float, float]:
    """Paired percentile-bootstrap 95% CI for the mean difference."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def generate_statistical_significance_report(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    metric: str = "f1_macro",
    alpha: float = 0.05,
    model_keys: Sequence[str] = BENCHMARK_MODELS,
    seed: int = 42,
) -> Dict[str, object]:
    """Test paired fold-level differences among the seven learned models.

    The primary analysis is a Friedman omnibus test. Pairwise Wilcoxon tests
    are reported with Holm correction across all model pairs. Positive mean
    differences and effect sizes favor ``model_a``.
    """
    required = {"model", "fold", metric}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Missing columns required for significance testing: {sorted(missing)}")

    selected = results_df[results_df["model"].isin(model_keys)][["model", "fold", metric]].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    pivot = selected.pivot_table(index="fold", columns="model", values=metric, aggfunc="mean")
    available_models = [model for model in model_keys if model in pivot.columns]
    if len(available_models) < 3:
        raise ValueError("At least three benchmark models are required for the Friedman test")
    pivot = pivot[available_models].dropna(axis=0, how="any").sort_index()
    if len(pivot) < 2:
        raise ValueError("At least two common completed folds are required for paired testing")

    statistic, omnibus_p = friedmanchisquare(*(pivot[model].to_numpy() for model in available_models))
    if not np.isfinite(statistic) or not np.isfinite(omnibus_p):
        statistic, omnibus_p = 0.0, 1.0
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    average_ranks = ranks.mean(axis=0).sort_values().rename("average_rank").reset_index()

    pairwise_rows = []
    for pair_index, (model_a, model_b) in enumerate(combinations(available_models, 2)):
        values_a = pivot[model_a].to_numpy(dtype=float)
        values_b = pivot[model_b].to_numpy(dtype=float)
        differences = values_a - values_b
        if np.allclose(differences, 0.0):
            test_statistic, raw_p = 0.0, 1.0
        else:
            test = wilcoxon(values_a, values_b, alternative="two-sided", method="auto")
            test_statistic, raw_p = float(test.statistic), float(test.pvalue)
        ci_low, ci_high = _bootstrap_mean_difference_ci(differences, seed + pair_index)
        pairwise_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "folds": int(len(pivot)),
                "mean_a": float(values_a.mean()),
                "mean_b": float(values_b.mean()),
                "mean_difference_a_minus_b": float(differences.mean()),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "wilcoxon_statistic": test_statistic,
                "p_value_raw": raw_p,
                "rank_biserial_effect": _matched_rank_biserial(differences),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df["p_value_holm"] = _holm_adjust(pairwise_df["p_value_raw"].to_numpy())
    pairwise_df["significant_holm"] = pairwise_df["p_value_holm"] < alpha
    pairwise_df = pairwise_df.sort_values(["p_value_holm", "p_value_raw"]).reset_index(drop=True)

    output_path = Path(output_dir) / "statistical_significance"
    output_path.mkdir(parents=True, exist_ok=True)
    common_folds_path = output_path / f"{metric}_common_fold_scores.csv"
    ranks_path = output_path / f"{metric}_average_ranks.csv"
    pairwise_path = output_path / f"{metric}_pairwise_wilcoxon_holm.csv"
    pivot.reset_index().to_csv(common_folds_path, index=False)
    average_ranks.to_csv(ranks_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    report = {
        "primary_metric": metric,
        "alpha": alpha,
        "models": available_models,
        "common_folds": int(len(pivot)),
        "omnibus_test": "Friedman chi-square test",
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(omnibus_p),
        "omnibus_significant": bool(omnibus_p < alpha),
        "post_hoc_test": "paired two-sided Wilcoxon signed-rank",
        "multiplicity_control": "Holm family-wise error correction",
        "effect_size": "matched-pairs rank-biserial correlation",
        "confidence_interval": "paired percentile bootstrap 95% CI of mean difference",
        "interpretation": (
            "At least one model differs beyond expected fold variation."
            if omnibus_p < alpha
            else "The benchmark does not provide sufficient evidence of an overall model difference."
        ),
        "files": {
            "common_fold_scores": str(common_folds_path),
            "average_ranks": str(ranks_path),
            "pairwise_tests": str(pairwise_path),
        },
    }
    report_json = output_path / f"{metric}_statistical_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text_lines = [
        "STATISTICAL SIGNIFICANCE ANALYSIS",
        "=" * 80,
        f"Primary metric: {metric}",
        f"Models: {', '.join(available_models)}",
        f"Common paired folds: {len(pivot)}",
        f"Friedman chi-square: {statistic:.6f}",
        f"Omnibus p-value: {omnibus_p:.6g}",
        f"Decision at alpha={alpha}: {'significant' if omnibus_p < alpha else 'not significant'}",
        report["interpretation"],
        "",
        "Pairwise results use two-sided Wilcoxon signed-rank tests with Holm correction.",
        "Positive differences/effect sizes favor model_a.",
        "",
        pairwise_df.to_string(index=False),
    ]
    (output_path / f"{metric}_statistical_report.txt").write_text("\n".join(text_lines), encoding="utf-8")
    return report


def generate_multi_metric_significance_report(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    metrics: Sequence[str] = ANALYSIS_METRICS,
    alpha: float = 0.05,
    model_keys: Sequence[str] = BENCHMARK_MODELS,
    seed: int = 42,
) -> Dict[str, object]:
    """Run the full paired analysis for precision, recall, accuracy, and F1."""
    metric_reports = []
    omnibus_rows = []
    pairwise_frames = []

    for metric_index, metric in enumerate(metrics):
        report = generate_statistical_significance_report(
            results_df=results_df,
            output_dir=output_dir,
            metric=metric,
            alpha=alpha,
            model_keys=model_keys,
            seed=seed + metric_index * 1000,
        )
        metric_reports.append(report)
        omnibus_rows.append(
            {
                "metric": metric,
                "friedman_statistic": report["friedman_statistic"],
                "p_value_raw": report["friedman_p_value"],
                "common_folds": report["common_folds"],
            }
        )
        pairwise = pd.read_csv(report["files"]["pairwise_tests"])
        pairwise.insert(0, "metric", metric)
        pairwise_frames.append(pairwise)

    output_path = Path(output_dir) / "statistical_significance"
    omnibus_df = pd.DataFrame(omnibus_rows)
    omnibus_df["p_value_holm_across_metrics"] = _holm_adjust(omnibus_df["p_value_raw"].to_numpy())
    omnibus_df["significant_holm_across_metrics"] = omnibus_df["p_value_holm_across_metrics"] < alpha
    omnibus_path = output_path / "all_metrics_friedman_omnibus.csv"
    omnibus_df.to_csv(omnibus_path, index=False)

    all_pairwise_df = pd.concat(pairwise_frames, ignore_index=True)
    all_pairwise_df["p_value_holm_across_all_metrics_and_pairs"] = _holm_adjust(
        all_pairwise_df["p_value_raw"].to_numpy()
    )
    all_pairwise_df["significant_holm_global"] = (
        all_pairwise_df["p_value_holm_across_all_metrics_and_pairs"] < alpha
    )
    all_pairwise_path = output_path / "all_metrics_pairwise_wilcoxon_holm.csv"
    all_pairwise_df.to_csv(all_pairwise_path, index=False)

    combined = {
        "metrics": list(metrics),
        "metric_display_names": {
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "accuracy_macro": "Macro Accuracy",
            "f1_macro": "Macro F1-score",
        },
        "alpha": alpha,
        "models": list(model_keys),
        "omnibus_multiplicity_control": "Holm correction across four Friedman tests",
        "pairwise_multiplicity_control_within_metric": "Holm correction across 21 model pairs",
        "pairwise_multiplicity_control_global": "Holm correction across all 84 metric/model-pair tests",
        "metric_reports": metric_reports,
        "files": {
            "omnibus_summary": str(omnibus_path),
            "all_pairwise_tests": str(all_pairwise_path),
        },
    }
    combined_json_path = output_path / "all_metrics_statistical_report.json"
    combined_json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    combined_text = [
        "MULTI-METRIC STATISTICAL SIGNIFICANCE ANALYSIS",
        "=" * 100,
        "Metrics: Macro Precision, Macro Recall, Macro Accuracy, Macro F1-score",
        "Omnibus p-values use Holm correction across the four metrics.",
        "Pairwise p-values include within-metric Holm correction and a global Holm correction across 84 tests.",
        "",
        "FRIEDMAN OMNIBUS RESULTS",
        omnibus_df.to_string(index=False),
        "",
        "PAIRWISE WILCOXON RESULTS",
        all_pairwise_df.to_string(index=False),
    ]
    (output_path / "all_metrics_statistical_report.txt").write_text(
        "\n".join(combined_text), encoding="utf-8"
    )
    return combined
