# Corner Kick Prediction: Velocity Features Experiment

**Author:** Mahmood Mohammed Seoud
**Date:** January 2026
**Institution:** IT University of Copenhagen

## Executive Summary

This experiment tested whether velocity/temporal features improve corner kick outcome prediction compared to position-only features. Using 10,002 corner kicks from realistic 11v11 Google Research Football simulations, we found that **velocity features improve prediction by 4% AUC**, supporting the hypothesis that static positioning alone is insufficient for corner kick prediction.

## Data Collection

### Source
- **Environment:** Google Research Football `11_vs_11_stochastic` scenario
- **Collection method:** Detected natural corner kicks during full matches
- **Normalization:** All corners normalized so attacking team attacks toward x=+1

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total corners | 10,002 |
| Matches played | 12,979 |
| Corners per match | 0.77 |
| Goal rate | 7.7% |
| Shot rate | 16.8% |

### Outcome Distribution
| Outcome | Count | Percentage |
|---------|-------|------------|
| Clearance | 8,236 | 82.3% |
| Shot missed | 915 | 9.1% |
| Goal | 770 | 7.7% |
| Foul | 80 | 0.8% |
| Out of play | 1 | <0.1% |

## Features

### Position Features (20)
Static spatial features extracted at moment of corner delivery:
- Zone counts: `attacking_in_box`, `defending_in_box`, `attacking_near_goal`, `defending_near_goal`
- Distances: `attacking_to_goal_dist_mean/min`, `defending_to_goal_dist_mean/min`
- Goalkeeper: `keeper_x`, `keeper_y`, `keeper_distance_to_goal`
- Density: `attacking_density`, `defending_density`, `defending_depth`
- Numerical: `numerical_advantage`, `attacker_defender_ratio`
- Ball: `ball_x`, `ball_y`

### Velocity Features (17)
Temporal/dynamic features capturing movement:
- Speeds: `attacking_speed_mean/max/std`, `defending_speed_mean/max/std`
- Direction: `attacking_toward_goal_mean/max`, `defending_toward_goal_mean`
- Movement counts: `attackers_moving_forward`, `defenders_moving_forward`
- Ball: `ball_speed`, `ball_vx`, `ball_vy`
- Goalkeeper: `keeper_speed`, `keeper_moving_toward_ball`
- Relative: `speed_differential`

## Results

### Model Performance (5-fold Cross-Validation)

| Model | Position-Only | Position+Velocity | Improvement |
|-------|---------------|-------------------|-------------|
| XGBoost | 0.571 | 0.594 | +2.3% |
| RandomForest | 0.573 | **0.613** | **+4.0%** |

### Test Set Performance

| Model | Position-Only | Position+Velocity | Improvement |
|-------|---------------|-------------------|-------------|
| XGBoost | 0.591 | 0.638 | +4.7% |
| RandomForest | 0.588 | **0.655** | **+6.7%** |

### Feature Importance (RandomForest, Position+Velocity model)

**Top 10 Features:**
| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | ball_speed | 8.4% | Velocity |
| 2 | ball_vx | 7.7% | Velocity |
| 3 | ball_vy | 6.1% | Velocity |
| 4 | attacking_speed_std | 4.3% | Velocity |
| 5 | attacking_toward_goal_mean | 4.2% | Velocity |
| 6 | ball_y | 4.0% | Position |
| 7 | defending_depth | 3.8% | Position |
| 8 | keeper_speed | 3.6% | Velocity |
| 9 | attacking_density | 3.5% | Position |
| 10 | defending_to_goal_dist_mean | 3.4% | Position |

**Aggregate Importance by Feature Type:**
- Position features: 44%
- Velocity features: **56%**

## Key Findings

### 1. Velocity Features Improve Prediction
- Adding velocity features improves AUC by **4.0%** (CV) to **6.7%** (test)
- Velocity features account for **56% of total feature importance**
- Ball velocity (`ball_speed`, `ball_vx`, `ball_vy`) are the top 3 most important features

### 2. Position-Only Performance Above Random
- Position-only AUC: 0.57 (vs expected ~0.50)
- This differs from the 7.5 ECTS StatsBomb result (AUC ≈ 0.50)
- Possible explanations:
  - GRF has more consistent/predictable positioning than real football
  - Larger sample size (10k vs smaller StatsBomb dataset)
  - Different feature engineering

### 3. Ball Dynamics Are Critical
- The ball's velocity at delivery is highly predictive
- `ball_speed` alone has 8.4% importance
- This aligns with the intuition that delivery quality matters

### 4. Player Movement Patterns Matter
- `attacking_speed_std` (variation in attacker speeds) is important
- Suggests coordinated vs. chaotic attacking movements are distinguishable
- `attacking_toward_goal_mean` captures attacking intent

## Implications for Thesis

### Hypothesis Validation
| Claim | Evidence | Status |
|-------|----------|--------|
| Position-only yields ~random performance | AUC = 0.57 (slightly above random) | Partial |
| Velocity features enable prediction | AUC improvement of 4-6% | **Confirmed** |
| This explains TacticAI's success | Velocity features are most important | **Supported** |

### Why Position-Only Performs Better Than Expected
The GRF simulation may have:
1. More deterministic physics than real football
2. Less tactical variation than professional teams
3. Consistent AI behavior patterns that position captures

### Thesis Narrative
The 7.5 ECTS work found that static freeze-frame data from StatsBomb yielded AUC ≈ 0.50. This experiment shows:

1. Even with some positional signal (AUC 0.57), **velocity adds significant predictive power** (AUC 0.61)
2. **56% of predictive signal comes from velocity features**, not position
3. Ball velocity at delivery is the single most important factor
4. This supports the hypothesis that TacticAI's 25Hz tracking data succeeds because it captures temporal dynamics

## Reproduction

### Data Collection
```bash
# Collect corners from 11v11 matches
python grf/scripts/collect_corners_11v11.py --num_corners 10000 --output_dir grf/data/raw_11v11

# Merge parallel job outputs
python grf/scripts/merge_corners.py --input_dirs job_1 job_2 job_3 job_4 --output_file corners_merged.json
```

### Feature Extraction
```bash
python grf/scripts/extract_features.py \
    --input_file grf/data/raw_11v11/corners_merged_10k.json \
    --output_dir grf/data/processed_11v11 \
    --use_frame delivery
```

### Model Training
```bash
python models/velocity_model.py \
    --data_file grf/data/processed_11v11/features_position_velocity.csv \
    --feature_info grf/data/processed_11v11/feature_info.json \
    --output_dir results_11v11
```

## Files

```
CornerTactics/
├── grf/
│   ├── scripts/
│   │   ├── collect_corners_11v11.py    # Data collection from 11v11 matches
│   │   ├── extract_features.py          # Feature engineering
│   │   └── merge_corners.py             # Merge parallel job outputs
│   ├── data/
│   │   ├── raw_11v11/                   # Raw corner episode JSONs
│   │   └── processed_11v11/             # Feature CSVs
│   └── slurm/
│       └── collect_11v11.slurm          # HPC job script
├── models/
│   ├── baseline_position.py             # Position-only baseline
│   └── velocity_model.py                # Position vs velocity comparison
├── results_11v11/                       # Model outputs
├── FINDINGS.md                          # This document
└── CLAUDE.md                            # Development notes
```
