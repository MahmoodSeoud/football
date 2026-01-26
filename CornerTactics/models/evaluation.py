#!/usr/bin/env python3
"""
Evaluation utilities for corner kick prediction models.
Provides consistent metrics and comparison functions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_pred_proba: np.ndarray) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_pred_proba)),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int(len(y_true) - y_true.sum()),
        'positive_rate': float(y_true.mean()),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])

    return metrics


def compare_models(results1: Dict, results2: Dict,
                  name1: str = 'Model 1', name2: str = 'Model 2') -> Dict:
    """
    Compare two models and compute improvement metrics.

    Args:
        results1: Metrics dict for first model (baseline)
        results2: Metrics dict for second model (test)
        name1: Name for first model
        name2: Name for second model

    Returns:
        Comparison dictionary
    """
    comparison = {
        'baseline': name1,
        'test': name2,
        'baseline_auc': results1.get('auc', 0),
        'test_auc': results2.get('auc', 0),
        'auc_improvement': results2.get('auc', 0) - results1.get('auc', 0),
        'auc_improvement_pct': (results2.get('auc', 0) - results1.get('auc', 0)) * 100,
        'f1_improvement': results2.get('f1', 0) - results1.get('f1', 0),
    }

    # Statistical significance thresholds
    comparison['improvement_significant'] = comparison['auc_improvement'] > 0.02
    comparison['improvement_marginal'] = 0.01 < comparison['auc_improvement'] <= 0.02
    comparison['no_improvement'] = comparison['auc_improvement'] <= 0.01

    return comparison


def bootstrap_auc_ci(y_true: np.ndarray, y_pred_proba: np.ndarray,
                     n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for AUC.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (auc_mean, ci_lower, ci_upper)
    """
    n_samples = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_sample = y_true[indices]
        proba_sample = y_pred_proba[indices]

        # Skip if only one class in sample
        if len(np.unique(y_sample)) < 2:
            continue

        aucs.append(roc_auc_score(y_sample, proba_sample))

    aucs = np.array(aucs)
    alpha = 1 - confidence
    ci_lower = np.percentile(aucs, alpha / 2 * 100)
    ci_upper = np.percentile(aucs, (1 - alpha / 2) * 100)

    return float(aucs.mean()), float(ci_lower), float(ci_upper)


def compute_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                n_bins: int = 10) -> Dict:
    """
    Compute model calibration metrics.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Calibration metrics dictionary
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / bin_counts.sum()
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred[:len(prob_true)]))

    return {
        'ece': float(ece),
        'calibration_curve_true': prob_true.tolist(),
        'calibration_curve_pred': prob_pred.tolist(),
    }


def generate_report(results: Dict, model_name: str = 'Model',
                   output_dir: Optional[str] = None) -> str:
    """
    Generate a text report from evaluation results.

    Args:
        results: Metrics dictionary
        model_name: Name of the model
        output_dir: Optional directory to save report

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"EVALUATION REPORT: {model_name}")
    lines.append("=" * 60)

    lines.append("\nPerformance Metrics:")
    lines.append(f"  AUC:       {results.get('auc', 'N/A'):.4f}")
    lines.append(f"  F1:        {results.get('f1', 'N/A'):.4f}")
    lines.append(f"  Accuracy:  {results.get('accuracy', 'N/A'):.4f}")
    lines.append(f"  Precision: {results.get('precision', 'N/A'):.4f}")
    lines.append(f"  Recall:    {results.get('recall', 'N/A'):.4f}")

    lines.append("\nDataset Statistics:")
    lines.append(f"  Total samples:  {results.get('n_samples', 'N/A')}")
    lines.append(f"  Positive:       {results.get('n_positive', 'N/A')}")
    lines.append(f"  Negative:       {results.get('n_negative', 'N/A')}")
    lines.append(f"  Positive rate:  {results.get('positive_rate', 0):.2%}")

    cm = results.get('confusion_matrix')
    if cm:
        lines.append("\nConfusion Matrix:")
        lines.append(f"  TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
        lines.append(f"  FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")

    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / f"{model_name.lower().replace(' ', '_')}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

    return report


def load_and_compare_results(baseline_file: str, test_file: str) -> Dict:
    """
    Load two result files and compare them.

    Args:
        baseline_file: Path to baseline results JSON
        test_file: Path to test results JSON

    Returns:
        Comparison results
    """
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    with open(test_file, 'r') as f:
        test = json.load(f)

    # Extract best model results
    baseline_auc = 0
    test_auc = 0

    for model in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        if model in baseline:
            baseline_auc = max(baseline_auc, baseline[model].get('auc', 0))
        if model in test:
            test_results = test[model]
            if 'position_velocity' in test_results:
                test_auc = max(test_auc, test_results['position_velocity'].get('auc', 0))
            else:
                test_auc = max(test_auc, test_results.get('auc', 0))

    return compare_models(
        {'auc': baseline_auc},
        {'auc': test_auc},
        'Position-Only',
        'Position+Velocity'
    )


def format_thesis_table(results: Dict) -> str:
    """
    Format results as a LaTeX-compatible table for thesis.

    Args:
        results: Dictionary with model comparison results

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Corner Kick Prediction Performance Comparison}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Feature Set & AUC & F1 & Accuracy & Improvement \\\\")
    lines.append("\\hline")

    # Add rows from results
    for model_type in ['XGBoost', 'RandomForest']:
        if model_type in results:
            model_results = results[model_type]

            pos_only = model_results.get('position_only', {})
            pos_vel = model_results.get('position_velocity', {})
            comparison = model_results.get('comparison', {})

            lines.append(
                f"{model_type} (Pos) & "
                f"{pos_only.get('auc', 0):.3f} & "
                f"{pos_only.get('f1', 0):.3f} & "
                f"{pos_only.get('accuracy', 0):.3f} & "
                f"-- \\\\"
            )
            lines.append(
                f"{model_type} (Pos+Vel) & "
                f"{pos_vel.get('auc', 0):.3f} & "
                f"{pos_vel.get('f1', 0):.3f} & "
                f"{pos_vel.get('accuracy', 0):.3f} & "
                f"{comparison.get('auc_improvement', 0):+.3f} \\\\"
            )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


if __name__ == '__main__':
    # Example usage / test
    print("Evaluation module loaded successfully.")
    print("\nAvailable functions:")
    print("  - compute_metrics(y_true, y_pred, y_pred_proba)")
    print("  - compare_models(results1, results2)")
    print("  - bootstrap_auc_ci(y_true, y_pred_proba)")
    print("  - generate_report(results, model_name)")
    print("  - load_and_compare_results(baseline_file, test_file)")
    print("  - format_thesis_table(results)")
