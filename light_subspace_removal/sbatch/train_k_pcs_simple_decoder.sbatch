#!/bin/bash
#SBATCH --job-name=chm_cv
#SBATCH --partition=preempt        # or your CPU partition
#SBATCH --gres=gpu:1           # drop if CPU-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-23%24
#SBATCH --output=logs/light_subspace_removal/%x_%A_%a.out
#SBATCH --error=logs/light_subspace_removal/%x_%A_%a.err

# ---- env setup (edit for your cluster) ----
# module load cuda/12.1
# mamba activate your_env

cd /home/kdoherty/light_subspace_removal

mkdir -p results/light_subspace_removal

# Configuration parameters
TOTAL_CONFIGS=50     # Total number of k/fold combinations
TOTAL_JOBS=24        # Number of parallel jobs
OUT_SIZE=224         # supervision size; must be a 2^n multiple of inferred token grid

echo "Running job ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS} for ${TOTAL_CONFIGS} total configs"

export PYTHONUNBUFFERED=1

python -u light_subspace_removal/scripts/train_k_pcs_simple_decoder.py \
  --job_id ${SLURM_ARRAY_TASK_ID} \
  --total_jobs ${TOTAL_JOBS} \
  --total_configs ${TOTAL_CONFIGS} \
  --outdir results/light_subspace_removal/simple_decoder \
  --epochs 50 \
  --batch_size 32 \
  --out_size ${OUT_SIZE}