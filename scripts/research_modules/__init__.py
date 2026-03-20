"""Research paper generation modules - Organized by functionality"""

from .report_builders import (
    generate_dataset_report,
    generate_model_configurations,
    generate_training_process_report,
    generate_research_paper_appendix,
)

from .metrics_analysis import (
    calculate_per_label_metrics,
    aggregate_per_label_metrics_across_folds,
    extract_per_label_metrics_from_results,
    generate_per_label_metrics_report,
    generate_multilabel_metrics_report,
)

from .visualizations import (
    load_smote_analysis,
    generate_smote_visualization,
    generate_confusion_matrix_visualizations,
    generate_training_curves,
    generate_comprehensive_heatmaps,
    generate_model_comparison_visualizations,
)

from .table_generators import (
    generate_comprehensive_metrics_report,
    generate_detailed_comparison_table,
)

__all__ = [
    # Report builders
    "generate_dataset_report",
    "generate_model_configurations",
    "generate_training_process_report",
    "generate_research_paper_appendix",
    # Metrics analysis
    "calculate_per_label_metrics",
    "aggregate_per_label_metrics_across_folds",
    "extract_per_label_metrics_from_results",
    "generate_per_label_metrics_report",
    "generate_multilabel_metrics_report",
    # Visualizations
    "load_smote_analysis",
    "generate_smote_visualization",
    "generate_confusion_matrix_visualizations",
    "generate_training_curves",
    "generate_comprehensive_heatmaps",
    "generate_model_comparison_visualizations",
    # Table generators
    "generate_comprehensive_metrics_report",
    "generate_detailed_comparison_table",
]
