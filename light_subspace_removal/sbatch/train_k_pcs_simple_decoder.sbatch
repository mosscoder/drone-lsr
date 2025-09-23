#!/bin/bash
#SBATCH --job-name=chm_cv_vpct_spatial
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/kdoherty/logs/light_subspace_removal/%x_%A_%a.out
#SBATCH --error=/home/kdoherty/logs/light_subspace_removal/%x_%A_%a.err

set -euo pipefail

cd /home/kdoherty/light-stable-semantics

# Env
source ~/.bashrc
mamba activate light-stable

# Threading (helps the SVD step on CPU)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_JIT=0
export PYTHONUNBUFFERED=1
# Optional: reduce CUDA OOM fragmentation on long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Grid parameters (spatial-only CV with variance targets)
TOTAL_CONFIGS=60      # 6 var levels × 5 folds × 2 model_configs
TOTAL_JOBS=8          # matches --array 0-7
OUT_SIZE=224          # supervision size (must be integer multiple of token grid)

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Job ${SLURM_ARRAY_TASK_ID}/${TOTAL_JOBS} over ${TOTAL_CONFIGS} configs"

# Run
python -u light_subspace_removal/scripts/train_vpct_spatial_cv.py \
  --job_id ${SLURM_ARRAY_TASK_ID} \
  --total_jobs ${TOTAL_JOBS} \
  --total_configs ${TOTAL_CONFIGS} \
  --outdir results/light_subspace_removal \
  --epochs 50 \
  --batch_size 32 \
  --out_size ${OUT_SIZE}