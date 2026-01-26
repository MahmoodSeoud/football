#!/bin/bash
#SBATCH --job-name=grf-corner-train
#SBATCH --output=logs/grf-train-%j.out
#SBATCH --error=logs/grf-train-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=scavenge

# Create directories
mkdir -p /home/$USER/football/CornerTactics/logs
mkdir -p /home/$USER/football/CornerTactics/results

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

# Paths
PROJECT_DIR="/home/$USER/football/CornerTactics"
DATA_DIR="$PROJECT_DIR/grf/data"
RESULTS_DIR="$PROJECT_DIR/results"

# Initialize conda
source /home/$USER/.bashrc
conda activate grf_new

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

cd $PROJECT_DIR

# Step 1: Extract features from raw data
echo ""
echo "=== Step 1: Feature Extraction ==="
EPISODES_FILE=$(ls -t $DATA_DIR/raw/corner_episodes_n*.json 2>/dev/null | head -1)

if [ -z "$EPISODES_FILE" ]; then
    echo "ERROR: No episode files found in $DATA_DIR/raw/"
    exit 1
fi

echo "Using episodes file: $EPISODES_FILE"
python grf/scripts/extract_features.py \
    --input_file "$EPISODES_FILE" \
    --output_dir $DATA_DIR/processed

# Check if feature extraction succeeded
if [ ! -f "$DATA_DIR/processed/features_position_only.csv" ]; then
    echo "ERROR: Feature extraction failed"
    exit 1
fi

# Step 2: Train position-only baseline
echo ""
echo "=== Step 2: Position-Only Baseline ==="
python models/baseline_position.py \
    --data_file $DATA_DIR/processed/features_position_only.csv \
    --output_dir $RESULTS_DIR \
    --permutation_test

# Step 3: Train position+velocity model and compare
echo ""
echo "=== Step 3: Position+Velocity Ablation ==="
python models/velocity_model.py \
    --data_file $DATA_DIR/processed/features_position_velocity.csv \
    --feature_info $DATA_DIR/processed/feature_info.json \
    --output_dir $RESULTS_DIR

echo ""
echo "End time: $(date)"
echo "Training complete!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "=== Summary ==="
if [ -f "$RESULTS_DIR/baseline_position_results.json" ]; then
    echo "Position-only results:"
    cat $RESULTS_DIR/baseline_position_results.json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"  XGBoost AUC: {d.get('XGBoost',{}).get('auc','N/A')}\")"
fi
if [ -f "$RESULTS_DIR/velocity_ablation_results.json" ]; then
    echo "Velocity ablation results:"
    cat $RESULTS_DIR/velocity_ablation_results.json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"  Position-only AUC: {d.get('position_only',{}).get('auc','N/A')}\"); print(f\"  Position+Velocity AUC: {d.get('position_velocity',{}).get('auc','N/A')}\")"
fi
