#!/bin/bash
#SBATCH --job-name=grf-11v11-corners
#SBATCH --output=logs/grf-11v11-%j.out
#SBATCH --error=logs/grf-11v11-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --partition=cores
#SBATCH --array=1-4

# Parallel collection: Run 4 jobs each collecting 2,500 corners
# Total: 10,000 corners
# Time estimate: 20 corners took 13 min = 38s/corner
# 2,500 corners × 38s = 26.4 hours per job
# Using 72 hours for safety buffer

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)"

conda activate grf_new

cd /home/$USER/football/CornerTactics

# Each array task collects 2,500 corners
CORNERS_PER_JOB=2500
OUTPUT_DIR="grf/data/raw_11v11/job_${SLURM_ARRAY_TASK_ID}"

python grf/scripts/collect_corners_11v11.py \
    --num_corners $CORNERS_PER_JOB \
    --output_dir "$OUTPUT_DIR" \
    --save_every 250

echo "End time: $(date)"
echo "Collection complete for job ${SLURM_ARRAY_TASK_ID}!"
