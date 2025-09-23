#!/bin/bash
#SBATCH --job-name=chm_cv
#SBATCH --partition=russ-lab         # Switch to general partition
#SBATCH --gres=gpu:A6000:1         # Request A6000 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # Increase memory for A6000
#SBATCH --time=24:00:00            # Increase time for larger dataset
#SBATCH --array=0-15%16              # Max 8 array jobs
#SBATCH --output=/home/kdoherty/logs/light_subspace_removal/%x_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/light_subspace_removal/%x_%A_%a.err

cd /home/kdoherty/light-stable-semantics

source ~/.bashrc
mamba activate light-stable

# Configuration parameters
TOTAL_CONFIGS=330    # 11 k values × 5 spatial folds × 3 temporal holdouts × 2 models
TOTAL_JOBS=16         # Number of parallel jobs
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