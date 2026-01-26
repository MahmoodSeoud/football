# Corner Kick Prediction with Google Research Football

## Environment Setup (Phase 1 - Completed)

This project uses Google Research Football (GRF) to generate synthetic corner kick data for prediction modeling.

### HPC Setup (ITU HPC)

Due to container runtime limitations on the HPC (Podman UID/GID issues, incomplete Apptainer installation), we use a **conda environment** instead of Docker/Apptainer containers.

#### Conda Environment: `grf_new`

The environment was created with the following key dependencies:
- Python 3.8
- gfootball (compiled from source)
- SDL2, SDL2_image, SDL2_ttf, SDL2_gfx
- Boost 1.85.0
- CMake 3.26

#### Activation

```bash
conda activate grf_new
```

#### Build Notes

The gfootball engine was compiled with the following modifications to `/home/mseo/football/third_party/gfootball_engine/CMakeLists.txt`:
1. Changed `cmake_minimum_required(VERSION 3.4)` to `VERSION 3.5` for CMake 4.x compatibility
2. Changed `set(OpenGL_GL_PREFERENCE GLVND)` to `LEGACY` for system OpenGL compatibility

EGL headers were manually downloaded from Khronos registry to `$CONDA_PREFIX/include/EGL/`.

The build script was modified to skip rebuilding if the library exists.

### Running Data Collection

```bash
conda activate grf_new
cd /home/mseo/football/CornerTactics
python grf/scripts/collect_corners.py --num_episodes 100 --output_dir grf/data/raw
```

### Project Structure

```
CornerTactics/
├── grf/
│   ├── docker/
│   │   └── Dockerfile              # Reference only (not used on HPC)
│   ├── scripts/
│   │   ├── collect_corners.py      # Main data collection script
│   │   ├── extract_features.py     # Feature engineering (Phase 3)
│   │   └── utils.py                # Helper functions
│   ├── slurm/
│   │   ├── collect_data.slurm      # Data collection job
│   │   └── train_model.slurm       # Model training job
│   └── data/
│       ├── raw/                    # Raw episode data
│       ├── processed/              # Extracted features
│       └── splits/                 # Train/val/test splits
├── models/
│   ├── baseline_position.py        # Position-only model
│   ├── velocity_model.py           # Position + velocity model
│   └── evaluation.py               # Metrics and comparison
├── notebooks/
│   └── analysis.ipynb              # Results visualization
├── results/
└── README.md
```
