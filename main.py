#!/usr/bin/env python
"""Single command-line entry point for the classification research pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CV_MODELS = [
    "linear_svm",
    "logistic_regression",
    "naive_bayes",
    "lstm",
    "bilstm",
    "bert",
    "roberta",
]
LLM_MODELS = ["llm_zero_shot", "llm_few_shot"]
ALL_RESEARCH_MODELS = [*CV_MODELS, *LLM_MODELS]


def _run(script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification research pipeline")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train the seven research models with 10-fold CV")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-smote", action="store_true")
    train.add_argument("--smote-target-ratio", type=float, default=1.0)

    compare = subparsers.add_parser("compare", help="Run all nine models and build the research comparison")
    compare.add_argument("--seed", type=int, default=42)

    llm = subparsers.add_parser("llm", help="Run OpenAI zero-shot and few-shot evaluation")
    llm.add_argument("--seed", type=int, default=42)
    llm.add_argument("--folds", type=int, default=1)

    sensitivity = subparsers.add_parser("sensitivity", help="Run the 10-fold BERT resampling study")
    sensitivity.add_argument("--seed", type=int, default=42)
    sensitivity.add_argument("--models", nargs="+", choices=["bert", "roberta"], default=["bert"])
    sensitivity.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command is None:
        _run("research_comparison.py", ["--models", *ALL_RESEARCH_MODELS, "--n_folds", "10", "--seed", "42"])
    elif args.command == "train":
        command = [
            "--models", *CV_MODELS,
            "--n_folds", "10",
            "--seed", str(args.seed),
            "--smote_target_ratio", str(args.smote_target_ratio),
        ]
        if args.no_smote:
            command.append("--no_smote")
        _run("train.py", command)
    elif args.command == "compare":
        _run(
            "research_comparison.py",
            ["--models", *ALL_RESEARCH_MODELS, "--n_folds", "10", "--seed", str(args.seed)],
        )
    elif args.command == "llm":
        _run("run_llm_only_metrics.py", ["--n_folds", str(args.folds), "--seed", str(args.seed)])
    elif args.command == "sensitivity":
        _run(
            "smote_sensitivity.py",
            ["--models", *args.models, "--seed", str(args.seed), "--epochs", str(args.epochs)],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
