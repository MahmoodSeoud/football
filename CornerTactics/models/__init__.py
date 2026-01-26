# Corner Kick Prediction Models
#
# This package contains:
# - baseline_position.py: Position-only model (replicates 7.5 ECTS approach)
# - velocity_model.py: Position + velocity model with ablation study
# - evaluation.py: Shared evaluation utilities

from .evaluation import (
    compute_metrics,
    compare_models,
    bootstrap_auc_ci,
    generate_report,
    load_and_compare_results,
    format_thesis_table
)

__all__ = [
    'compute_metrics',
    'compare_models',
    'bootstrap_auc_ci',
    'generate_report',
    'load_and_compare_results',
    'format_thesis_table'
]
