#!/usr/bin/env python3
"""
Position-only baseline model.
Replicates the 7.5 ECTS approach on GRF synthetic data.
Expected result: AUC ~ 0.50 (random chance)

This validates that static positioning alone cannot predict corner kick outcomes,
matching the findings from the original StatsBomb analysis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_data(filepath):
    """Load feature CSV and split into X, y."""
    df = pd.read_csv(filepath)

    # Separate features and labels
    label_cols = ['episode_id', 'shot_label', 'goal_label']
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].values
    y = df['shot_label'].values

    return X, y, feature_cols, df


def train_and_evaluate(X, y, feature_names, model_name='XGBoost'):
    """Train model and evaluate performance."""

    # Check minimum sample size
    if len(y) < 10:
        print(f"  WARNING: Only {len(y)} samples - results will be unreliable")

    # Train/test split (episodes are independent in GRF)
    # Use stratification only if we have enough samples per class
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    use_stratify = min_class_count >= 2 and len(y) >= 10

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if use_stratify else None
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weight for imbalanced data
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Initialize model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Check if we can compute AUC (need both classes in test set)
    n_classes_test = len(np.unique(y_test))
    if n_classes_test < 2:
        print("  WARNING: Test set has only one class - AUC undefined, using 0.5")
        auc_score = 0.5
    else:
        auc_score = roc_auc_score(y_test, y_pred_proba)

    # Metrics
    results = {
        'model': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc': float(auc_score),
        'train_size': int(len(y_train)),
        'test_size': int(len(y_test)),
        'positive_rate_train': float(y_train.mean()),
        'positive_rate_test': float(y_test.mean()),
    }

    # Cross-validation AUC with stratified folds
    # Adjust number of folds based on sample size
    n_splits = min(5, len(y_train) // 2)  # At least 2 samples per fold
    if n_splits >= 2 and min(np.sum(y_train == 0), np.sum(y_train == 1)) >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        results['cv_auc_mean'] = float(cv_scores.mean())
        results['cv_auc_std'] = float(cv_scores.std())
        results['cv_auc_scores'] = [float(s) for s in cv_scores]
    else:
        results['cv_auc_mean'] = results['auc']  # Fallback to test AUC
        results['cv_auc_std'] = 0.0
        results['cv_auc_scores'] = []
        print("  WARNING: Insufficient samples for cross-validation")

    # Feature importance (for tree models)
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
        results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])
    elif hasattr(model, 'coef_'):
        # For logistic regression, use absolute coefficients
        importance = dict(zip(feature_names, [float(abs(x)) for x in model.coef_[0]]))
        results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm.tolist()

    return results, model, scaler


def permutation_test(X, y, n_permutations=100):
    """
    Permutation test to verify whether there's real signal in the data.
    Train on shuffled labels and compare to real labels.

    If position features have no predictive power, the real AUC should be
    statistically indistinguishable from the shuffled AUC distribution.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model for testing
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Real label performance
    real_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    real_auc = float(real_scores.mean())

    print(f"  Running {n_permutations} permutations...")

    # Shuffled label performances
    shuffled_aucs = []
    for i in range(n_permutations):
        y_shuffled = np.random.permutation(y)
        try:
            scores = cross_val_score(model, X_scaled, y_shuffled, cv=cv, scoring='roc_auc')
            shuffled_aucs.append(float(scores.mean()))
        except Exception:
            # Some permutations might create issues with stratification
            continue

        if (i + 1) % 20 == 0:
            print(f"    Completed {i + 1}/{n_permutations} permutations")

    shuffled_aucs = np.array(shuffled_aucs)

    # Statistical test
    p_value = float(np.mean(shuffled_aucs >= real_auc))
    z_score = float((real_auc - shuffled_aucs.mean()) / (shuffled_aucs.std() + 1e-10))

    return {
        'real_auc': real_auc,
        'shuffled_auc_mean': float(shuffled_aucs.mean()),
        'shuffled_auc_std': float(shuffled_aucs.std()),
        'shuffled_auc_min': float(shuffled_aucs.min()),
        'shuffled_auc_max': float(shuffled_aucs.max()),
        'p_value': p_value,
        'z_score': z_score,
        'significant': p_value < 0.05,
        'n_permutations': len(shuffled_aucs)
    }


def print_summary(results):
    """Print a summary of results."""
    print("\n" + "=" * 60)
    print("POSITION-ONLY BASELINE RESULTS")
    print("=" * 60)

    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        if model_name in results:
            r = results[model_name]
            print(f"\n{model_name}:")
            print(f"  Test AUC:    {r['auc']:.4f}")
            print(f"  CV AUC:      {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f}")
            print(f"  F1 Score:    {r['f1']:.4f}")
            print(f"  Accuracy:    {r['accuracy']:.4f}")

    if 'permutation_test' in results:
        pt = results['permutation_test']
        print(f"\nPermutation Test:")
        print(f"  Real AUC:      {pt['real_auc']:.4f}")
        print(f"  Shuffled AUC:  {pt['shuffled_auc_mean']:.4f} +/- {pt['shuffled_auc_std']:.4f}")
        print(f"  p-value:       {pt['p_value']:.4f}")
        print(f"  Significant:   {pt['significant']} (p < 0.05)")

    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    best_auc = max(results.get(m, {}).get('auc', 0) for m in ['XGBoost', 'RandomForest', 'LogisticRegression'])

    if best_auc < 0.55:
        print("  Position-only features show NO predictive signal (AUC ~ 0.50)")
        print("  This confirms the 7.5 ECTS finding: static positioning alone")
        print("  cannot predict corner kick outcomes.")
    else:
        print(f"  Unexpected: Position features show some signal (AUC = {best_auc:.4f})")
        print("  This differs from the 7.5 ECTS findings. Investigate further.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train position-only baseline model')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to features_position_only.csv')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--permutation_test', action='store_true',
                        help='Run permutation test to verify null hypothesis')
    parser.add_argument('--n_permutations', type=int, default=100,
                        help='Number of permutations for significance test')

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_file}...")
    X, y, feature_names, df = load_data(args.data_file)
    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Positive rate (shots): {y.mean():.2%}")
    print(f"  Class balance: {len(y[y==0])} negative, {len(y[y==1])} positive")

    # Train models
    all_results = {}
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        print(f"\nTraining {model_name}...")
        results, model, scaler = train_and_evaluate(X, y, feature_names, model_name)
        all_results[model_name] = results

        print(f"  Test AUC: {results['auc']:.4f}")
        print(f"  F1:       {results['f1']:.4f}")
        print(f"  CV AUC:   {results['cv_auc_mean']:.4f} +/- {results['cv_auc_std']:.4f}")

    # Permutation test
    if args.permutation_test:
        print(f"\nRunning permutation test ({args.n_permutations} permutations)...")
        perm_results = permutation_test(X, y, n_permutations=args.n_permutations)
        all_results['permutation_test'] = perm_results

        print(f"  Real AUC:     {perm_results['real_auc']:.4f}")
        print(f"  Shuffled AUC: {perm_results['shuffled_auc_mean']:.4f} +/- {perm_results['shuffled_auc_std']:.4f}")
        print(f"  p-value:      {perm_results['p_value']:.4f}")
        print(f"  Significant:  {perm_results['significant']}")

    # Add metadata
    all_results['metadata'] = {
        'data_file': str(args.data_file),
        'n_samples': int(len(y)),
        'n_features': int(len(feature_names)),
        'feature_names': feature_names,
        'positive_rate': float(y.mean())
    }

    # Save results
    results_file = output_path / 'baseline_position_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
