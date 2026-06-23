"""Generate fold-level per-label CSV reports for OpenAI LLM artifact folders."""

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
module_dir = project_root / "scripts" / "research_modules"
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

from metrics_analysis import generate_fold_level_per_label_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate fold-level per-label reports for OpenAI llm_zero_shot and llm_few_shot runs."
    )
    parser.add_argument(
        "--artifact-root",
        default="results/modular_multimodel/model_artifacts",
        help="Root directory containing model artifact folders",
    )
    parser.add_argument(
        "--output-dir",
        default="results/research_comparison",
        help="Directory where CSV reports will be written",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llm_zero_shot", "llm_few_shot"],
        help="Model keys to include",
    )
    args = parser.parse_args()

    result = generate_fold_level_per_label_report(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        model_keys=args.models,
    )

    if not result["fold_level_csv"]:
        print("No usable artifact predictions/labels were found.")
        return 1

    print("OpenAI LLM per-label reports generated")
    print(f"  Fold-level CSV: {result['fold_level_csv']}")
    print(f"  Summary CSV:    {result['summary_csv']}")
    print(f"  JSON report:    {result['report_json']}")
    print(f"  Text report:    {result['report_txt']}")
    print(f"  Rows written:   {result['rows']}")
    print(f"  Models:         {', '.join(result['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
