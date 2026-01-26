#!/bin/bash
#SBATCH --job-name=grf-corner-collect
#SBATCH --output=logs/grf-collect-%j.out
#SBATCH --error=logs/grf-collect-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1

# Create log directory if needed
mkdir -p /home/$USER/football/CornerTactics/logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

# Set paths
PROJECT_DIR="/home/$USER/football/CornerTactics"
OUTPUT_DIR="$PROJECT_DIR/grf/data/raw"

# Number of episodes to collect
NUM_EPISODES=${1:-10000}

# Initialize conda
source /home/$USER/.bashrc
conda activate grf_new

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Start virtual framebuffer for headless rendering
Xvfb :99 -screen 0 1024x768x24 &> /dev/null &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

# Run data collection
cd $PROJECT_DIR
python grf/scripts/collect_corners.py \
    --num_episodes $NUM_EPISODES \
    --output_dir $OUTPUT_DIR \
    --save_every 500

# Cleanup
kill $XVFB_PID 2>/dev/null

echo "End time: $(date)"
echo "Data collection complete!"
echo "Data saved to: $OUTPUT_DIR"
