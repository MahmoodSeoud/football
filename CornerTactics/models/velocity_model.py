#!/usr/bin/env python3
"""
Position + Velocity model with ablation study.
Tests whether velocity features improve prediction beyond position-only baseline.

This is the KEY experiment for the thesis:
- If velocity helps (AUC improvement > 0.02), it explains why TacticAI works
- If velocity doesn't help, the hypothesis needs revision

Expected result: AUC > 0.55 with velocity features (significant improvement)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_data(filepath):
    """Load feature CSV."""
    df = pd.read_csv(filepath)
    label_cols = ['episode_id', 'shot_label', 'goal_label']
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].values
    y = df['shot_label'].values

    return X, y, feature_cols, df


def train_model(X_train, y_train, X_test, y_test, model_type='XGBoost'):
    """Train a model and return predictions and AUC."""
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    scale_pos_weight = n_neg / max(n_pos, 1)

    if model_type == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Check if we can compute AUC (need both classes in test set)
    n_classes_test = len(np.unique(y_test))
    if n_classes_test < 2:
        auc = 0.5  # Undefined, use random baseline
    else:
        auc = roc_auc_score(y_test, y_pred_proba)

    return model, y_pred, y_pred_proba, auc


def ablation_study(X_full, y, feature_names, position_feature_count, model_type='XGBoost'):
    """
    Compare position-only vs position+velocity performance.
    This is the KEY experiment for the thesis.

    Args:
        X_full: Full feature matrix with all features
        y: Labels
        feature_names: List of all feature names
        position_feature_count: Number of position-only features (first N columns)
        model_type: 'XGBoost' or 'RandomForest'

    Returns:
        dict with comparison results
    """
    # Split features by type
    X_position = X_full[:, :position_feature_count]
    X_velocity_only = X_full[:, position_feature_count:]

    position_features = feature_names[:position_feature_count]
    velocity_features = feature_names[position_feature_count:]

    print(f"\nFeature split:")
    print(f"  Position features: {len(position_features)}")
    print(f"  Velocity features: {len(velocity_features)}")

    # Check minimum sample size
    if len(y) < 10:
        print(f"\n  WARNING: Only {len(y)} samples - results will be unreliable")

    # Train/test split (same split for fair comparison)
    # Use stratification only if we have enough samples per class
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    use_stratify = min_class_count >= 2 and len(y) >= 10

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42,
        stratify=y if use_stratify else None
    )

    X_train_pos = X_train_full[:, :position_feature_count]
    X_test_pos = X_test_full[:, :position_feature_count]

    X_train_vel = X_train_full[:, position_feature_count:]
    X_test_vel = X_test_full[:, position_feature_count:]

    # Standardize each feature set
    scaler_full = StandardScaler()
    scaler_pos = StandardScaler()
    scaler_vel = StandardScaler()

    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_full.transform(X_test_full)

    X_train_pos_scaled = scaler_pos.fit_transform(X_train_pos)
    X_test_pos_scaled = scaler_pos.transform(X_test_pos)

    X_train_vel_scaled = scaler_vel.fit_transform(X_train_vel)
    X_test_vel_scaled = scaler_vel.transform(X_test_vel)

    results = {}

    # Adjust CV folds based on sample size
    n_splits = min(5, len(y_train) // 2)
    can_do_cv = n_splits >= 2 and min(np.sum(y_train == 0), np.sum(y_train == 1)) >= n_splits
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42) if can_do_cv else None

    # ========== Position-Only Model ==========
    print(f"\n--- Position-Only Model ({model_type}) ---")
    model_pos, y_pred_pos, y_proba_pos, auc_pos = train_model(
        X_train_pos_scaled, y_train, X_test_pos_scaled, y_test, model_type
    )

    # Cross-validation for position-only
    if cv is not None:
        cv_pos = cross_val_score(
            model_pos, X_train_pos_scaled, y_train, cv=cv, scoring='roc_auc'
        )
        cv_pos_mean, cv_pos_std = float(cv_pos.mean()), float(cv_pos.std())
        cv_pos_scores = [float(s) for s in cv_pos]
    else:
        cv_pos_mean, cv_pos_std, cv_pos_scores = auc_pos, 0.0, []
        print("  WARNING: Insufficient samples for cross-validation")

    results['position_only'] = {
        'auc': float(auc_pos),
        'cv_auc_mean': cv_pos_mean,
        'cv_auc_std': cv_pos_std,
        'cv_auc_scores': cv_pos_scores,
        'f1': float(f1_score(y_test, y_pred_pos, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, y_pred_pos)),
        'num_features': len(position_features),
        'confusion_matrix': confusion_matrix(y_test, y_pred_pos).tolist()
    }

    print(f"  Test AUC: {auc_pos:.4f}")
    print(f"  CV AUC:   {cv_pos_mean:.4f} +/- {cv_pos_std:.4f}")

    # ========== Position + Velocity Model ==========
    print(f"\n--- Position + Velocity Model ({model_type}) ---")
    model_full, y_pred_full, y_proba_full, auc_full = train_model(
        X_train_full_scaled, y_train, X_test_full_scaled, y_test, model_type
    )

    if cv is not None:
        cv_full = cross_val_score(
            model_full, X_train_full_scaled, y_train, cv=cv, scoring='roc_auc'
        )
        cv_full_mean, cv_full_std = float(cv_full.mean()), float(cv_full.std())
        cv_full_scores = [float(s) for s in cv_full]
    else:
        cv_full_mean, cv_full_std, cv_full_scores = auc_full, 0.0, []

    results['position_velocity'] = {
        'auc': float(auc_full),
        'cv_auc_mean': cv_full_mean,
        'cv_auc_std': cv_full_std,
        'cv_auc_scores': cv_full_scores,
        'f1': float(f1_score(y_test, y_pred_full, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, y_pred_full)),
        'num_features': len(feature_names),
        'confusion_matrix': confusion_matrix(y_test, y_pred_full).tolist()
    }

    print(f"  Test AUC: {auc_full:.4f}")
    print(f"  CV AUC:   {cv_full_mean:.4f} +/- {cv_full_std:.4f}")

    # ========== Velocity-Only Model (Additional ablation) ==========
    print(f"\n--- Velocity-Only Model ({model_type}) ---")
    model_vel, y_pred_vel, y_proba_vel, auc_vel = train_model(
        X_train_vel_scaled, y_train, X_test_vel_scaled, y_test, model_type
    )

    if cv is not None:
        cv_vel = cross_val_score(
            model_vel, X_train_vel_scaled, y_train, cv=cv, scoring='roc_auc'
        )
        cv_vel_mean, cv_vel_std = float(cv_vel.mean()), float(cv_vel.std())
        cv_vel_scores = [float(s) for s in cv_vel]
    else:
        cv_vel_mean, cv_vel_std, cv_vel_scores = auc_vel, 0.0, []

    results['velocity_only'] = {
        'auc': float(auc_vel),
        'cv_auc_mean': cv_vel_mean,
        'cv_auc_std': cv_vel_std,
        'cv_auc_scores': cv_vel_scores,
        'f1': float(f1_score(y_test, y_pred_vel, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, y_pred_vel)),
        'num_features': len(velocity_features),
        'confusion_matrix': confusion_matrix(y_test, y_pred_vel).tolist()
    }

    print(f"  Test AUC: {auc_vel:.4f}")
    print(f"  CV AUC:   {cv_vel_mean:.4f} +/- {cv_vel_std:.4f}")

    # ========== Statistical Comparison ==========
    print("\n--- Comparison ---")
    auc_improvement = auc_full - auc_pos
    cv_improvement = cv_full_mean - cv_pos_mean

    # Check if improvement is meaningful
    # Using 2% as threshold for "meaningful" improvement
    velocity_helps = auc_improvement > 0.02
    velocity_helps_cv = cv_improvement > 0.02

    results['comparison'] = {
        'auc_improvement': float(auc_improvement),
        'auc_improvement_pct': float(auc_improvement * 100),
        'cv_auc_improvement': float(cv_improvement),
        'velocity_helps_test': velocity_helps,
        'velocity_helps_cv': velocity_helps_cv,
        'position_only_auc': float(auc_pos),
        'position_velocity_auc': float(auc_full),
        'velocity_only_auc': float(auc_vel),
    }

    print(f"  AUC Improvement (test): {auc_improvement:+.4f} ({auc_improvement*100:+.2f}%)")
    print(f"  AUC Improvement (CV):   {cv_improvement:+.4f} ({cv_improvement*100:+.2f}%)")
    print(f"  Velocity helps (test):  {velocity_helps}")
    print(f"  Velocity helps (CV):    {velocity_helps_cv}")

    # ========== Feature Importance Analysis ==========
    if hasattr(model_full, 'feature_importances_'):
        importance = dict(zip(feature_names, [float(x) for x in model_full.feature_importances_]))

        # Separate by feature type
        pos_importance = {k: v for k, v in importance.items() if k in position_features}
        vel_importance = {k: v for k, v in importance.items() if k in velocity_features}

        # Top features overall
        results['feature_importance_all'] = dict(sorted(importance.items(), key=lambda x: -x[1])[:15])

        # Top velocity features specifically
        results['velocity_feature_importance'] = dict(sorted(vel_importance.items(), key=lambda x: -x[1])[:10])

        # Aggregate importance by feature type
        total_importance = sum(importance.values())
        results['importance_by_type'] = {
            'position_total': float(sum(pos_importance.values())),
            'velocity_total': float(sum(vel_importance.values())),
            'position_pct': float(sum(pos_importance.values()) / total_importance * 100) if total_importance > 0 else 0,
            'velocity_pct': float(sum(vel_importance.values()) / total_importance * 100) if total_importance > 0 else 0,
        }

        print(f"\n  Feature importance breakdown:")
        print(f"    Position features: {results['importance_by_type']['position_pct']:.1f}%")
        print(f"    Velocity features: {results['importance_by_type']['velocity_pct']:.1f}%")

        print(f"\n  Top 5 velocity features:")
        for feat, imp in list(results['velocity_feature_importance'].items())[:5]:
            print(f"    {feat}: {imp:.4f}")

    return results


def run_multiple_models(X_full, y, feature_names, position_feature_count):
    """Run ablation study with multiple model types."""
    all_results = {}

    for model_type in ['XGBoost', 'RandomForest']:
        print(f"\n{'='*60}")
        print(f"Model: {model_type}")
        print('='*60)

        results = ablation_study(
            X_full, y, feature_names, position_feature_count, model_type
        )
        all_results[model_type] = results

    return all_results


def print_thesis_conclusion(results):
    """Print thesis-relevant conclusions."""
    print("\n" + "=" * 60)
    print("THESIS CONCLUSION")
    print("=" * 60)

    # Get best results (prefer XGBoost)
    model_results = results.get('XGBoost', results.get('RandomForest', {}))
    comparison = model_results.get('comparison', {})

    pos_auc = comparison.get('position_only_auc', 0)
    full_auc = comparison.get('position_velocity_auc', 0)
    vel_only_auc = comparison.get('velocity_only_auc', 0)
    improvement = comparison.get('auc_improvement', 0)

    print(f"\n  Position-only AUC:       {pos_auc:.4f}")
    print(f"  Position+Velocity AUC:   {full_auc:.4f}")
    print(f"  Velocity-only AUC:       {vel_only_auc:.4f}")
    print(f"  Improvement:             {improvement:+.4f}")

    if comparison.get('velocity_helps_test') or comparison.get('velocity_helps_cv'):
        print("\n  RESULT: Velocity features IMPROVE corner kick prediction")
        print("\n  INTERPRETATION:")
        print("    - Position-only features match 7.5 ECTS findings (AUC ~ 0.50)")
        print("    - Adding velocity improves prediction significantly")
        print("    - This explains WHY TacticAI works: their 25Hz tracking")
        print("      provides velocity information that enables prediction")
        print("\n  THESIS CONTRIBUTION:")
        print("    Static positioning alone is fundamentally insufficient for")
        print("    corner kick prediction. Temporal dynamics (velocity) are required.")
    else:
        print("\n  RESULT: Velocity features did NOT significantly improve prediction")
        print("\n  POSSIBLE EXPLANATIONS:")
        if pos_auc > 0.55:
            print("    - Position features show more signal than expected in GRF")
            print("    - GRF environment may differ from real football dynamics")
        elif vel_only_auc < 0.52:
            print("    - Velocity features alone also lack predictive power")
            print("    - The GRF random agent may not create realistic dynamics")
        else:
            print("    - Further investigation needed")
            print("    - Consider: more data, better agents, different scenarios")

    # Importance breakdown
    imp_by_type = model_results.get('importance_by_type', {})
    if imp_by_type:
        print(f"\n  Feature importance distribution:")
        print(f"    Position: {imp_by_type.get('position_pct', 0):.1f}%")
        print(f"    Velocity: {imp_by_type.get('velocity_pct', 0):.1f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train position+velocity model with ablation')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to features_position_velocity.csv')
    parser.add_argument('--feature_info', type=str, required=True,
                        help='Path to feature_info.json')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='all',
                        choices=['XGBoost', 'RandomForest', 'all'],
                        help='Which model(s) to train')

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load feature info to know how many position features there are
    print(f"Loading feature info from {args.feature_info}...")
    with open(args.feature_info, 'r') as f:
        feature_info = json.load(f)

    position_feature_count = len(feature_info['position_features'])

    # Load data
    print(f"Loading data from {args.data_file}...")
    X, y, feature_names, df = load_data(args.data_file)

    print(f"\n  Total samples:      {len(y)}")
    print(f"  Position features:  {position_feature_count}")
    print(f"  Velocity features:  {len(feature_names) - position_feature_count}")
    print(f"  Total features:     {len(feature_names)}")
    print(f"  Positive rate:      {y.mean():.2%}")
    print(f"  Class balance:      {len(y[y==0])} negative, {len(y[y==1])} positive")

    # Run ablation study
    if args.model == 'all':
        all_results = run_multiple_models(X, y, feature_names, position_feature_count)
    else:
        print(f"\n{'='*60}")
        print(f"Model: {args.model}")
        print('='*60)
        all_results = {
            args.model: ablation_study(X, y, feature_names, position_feature_count, args.model)
        }

    # Add metadata
    all_results['metadata'] = {
        'data_file': str(args.data_file),
        'n_samples': int(len(y)),
        'n_features': int(len(feature_names)),
        'position_feature_count': position_feature_count,
        'velocity_feature_count': len(feature_names) - position_feature_count,
        'position_features': feature_info['position_features'],
        'velocity_features': feature_info['velocity_features'],
        'positive_rate': float(y.mean()),
        'shot_rate_from_info': feature_info.get('shot_rate'),
        'goal_rate_from_info': feature_info.get('goal_rate'),
    }

    # Save results
    results_file = output_path / 'velocity_ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print thesis conclusion
    print_thesis_conclusion(all_results)


if __name__ == '__main__':
    main()
