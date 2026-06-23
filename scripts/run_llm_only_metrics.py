"""Run only OpenAI zero-shot/few-shot prediction and generate per-label metrics reports."""

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
scripts_dir = project_root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from research_comparison import run_research_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only llm_zero_shot and llm_few_shot, then generate per-label metrics reports."
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=1,
        help="Number of folds. Default is 1 (holdout only, no 10-fold CV).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_research_comparison(
        n_folds=args.n_folds,
        seed=args.seed,
        selected_model_keys=["llm_zero_shot", "llm_few_shot"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
