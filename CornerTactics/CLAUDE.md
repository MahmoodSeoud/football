# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis project investigating whether velocity/temporal features improve corner kick outcome prediction in football. The project uses Google Research Football (GRF) simulator to generate synthetic data with full player tracking (positions + velocities).

**Hypothesis:** Position-only features yield AUC ≈ 0.50 (random), while adding velocity features should achieve AUC > 0.55.

## Environment Setup

### Why Conda Instead of Apptainer/Docker

The original plan (in `/home/mseo/football/plan.md`) called for Docker containers converted to Apptainer for HPC use. This approach was abandoned due to:

1. **Podman UID/GID issues**: The HPC has Podman but no `/etc/subuid` entry for the user, causing "potentially insufficient UIDs or GIDs available in user namespace" errors during image pulls/builds
2. **Incomplete Apptainer installation**: The `/opt/itu/easybuild/software/Apptainer/` directories exist but are empty - the EasyBuild installation was never completed
3. **No Docker on HPC**: Docker is not available on HPC login/compute nodes for security reasons

### Working Environment: `grf_new` conda environment

```bash
# Activate
conda activate grf_new

# Key packages: Python 3.8, gfootball (compiled from source), SDL2, Boost 1.85.0, CMake 3.26
```

### GRF Build Modifications

The gfootball engine in `/home/mseo/football/` was compiled with these changes to `third_party/gfootball_engine/CMakeLists.txt`:
- Line 1: `VERSION 3.4` → `VERSION 3.5` (CMake compatibility)
- Line 39: `GLVND` → `LEGACY` (system OpenGL compatibility)

EGL headers manually installed at `$CONDA_PREFIX/include/EGL/` from Khronos registry.

Build script (`gfootball/build_game_engine.sh`) modified to skip rebuild if `libgame.so` exists.

## Commands

### Data Collection
```bash
conda activate grf_new
cd /home/mseo/football/CornerTactics
python grf/scripts/collect_corners.py --num_episodes 1000 --output_dir grf/data/raw --save_every 100
```

### SLURM Job Submission
```bash
sbatch grf/slurm/collect_data.slurm
sbatch grf/slurm/train_model.slurm
squeue -u $USER  # Monitor jobs
tail -f logs/grf-collect-<jobid>.err  # Watch progress bar
```

### HPC Partitions
- **acltr**: GPU partition with A100s, 7-day limit (use this for data collection)
- **scavenge**: Default partition, 1-day limit, preemptible
- **cores**: CPU-only partition

### GPU Specification
```bash
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1              # Request 1 GPU (any type)
#SBATCH --gres=gpu:a100:1         # Request specific GPU type
```

Available GPU nodes on acltr: cn3, cn5, cn6, cn7, cn18

## Architecture

### Data Pipeline
1. **Collection** (`grf/scripts/collect_corners.py`): Runs GRF `academy_corner` scenario, captures per-frame player/ball positions and velocities, labels outcomes (goal/shot/clearance)
2. **Feature Extraction** (`grf/scripts/extract_features.py`): Creates position-only and position+velocity feature sets
3. **Modeling** (`models/`): XGBoost/RandomForest classifiers comparing feature sets

### Data Format
Episodes stored as JSON with structure:
```python
{
    'frames': [{'ball_position': [x,y,z], 'ball_velocity': [vx,vy,vz],
                'left_team_positions': [[x,y],...], 'left_team_velocities': [[vx,vy],...],
                'right_team_positions': [...], 'right_team_velocities': [...],
                'game_mode': int, 'score': [l,r], 'step': int}, ...],
    'actions': [int, ...],
    'rewards': [float, ...],
    'outcome': {'shot_occurred': bool, 'goal_scored': bool, 'outcome': str},
    'metadata': {...}
}
```

### Key Directories
- `grf/data/raw/`: Collected episode JSONs
- `grf/data/processed/`: Feature CSVs (position-only and position+velocity)
- `results/`: Model evaluation outputs

## Implementation Status

- **Phase 1 (Environment Setup)**: ✅ Complete - using conda instead of containers
- **Phase 2 (HPC Container Setup)**: ⏭️ Skipped - not needed with conda approach
- **Phase 3 (Data Collection)**: 🔲 Ready to run large-scale collection (10k+ episodes needed)
- **Phase 4 (Model Training)**: ✅ Complete - all model scripts implemented

## Model Training (Phase 4)

### Position-Only Baseline
Replicates the 7.5 ECTS approach using static position features only.
```bash
python models/baseline_position.py \
    --data_file grf/data/processed/features_position_only.csv \
    --output_dir results \
    --permutation_test
```

### Velocity Ablation Study
Compares position-only vs position+velocity performance (KEY EXPERIMENT).
```bash
python models/velocity_model.py \
    --data_file grf/data/processed/features_position_velocity.csv \
    --feature_info grf/data/processed/feature_info.json \
    --output_dir results
```

### Model Files
- `models/baseline_position.py`: Position-only baseline (XGBoost, RF, LogReg)
- `models/velocity_model.py`: Ablation comparing position vs position+velocity
- `models/evaluation.py`: Shared metrics and comparison utilities

### Expected Results
- Position-only: AUC ~ 0.50 (confirms 7.5 ECTS findings)
- Position+Velocity: AUC > 0.55 (validates velocity hypothesis)
