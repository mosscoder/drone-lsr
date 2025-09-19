#!/bin/bash
#SBATCH --job-name=chm_cv
#SBATCH --partition=preempt        # or your CPU partition
#SBATCH --gres=gpu:1           # drop if CPU-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-49%24
#SBATCH --output=logs/light_subspace_removal/%x_%A_%a.out
#SBATCH --error=logs/light_subspace_removal/%x_%A_%a.err

# ---- env setup (edit for your cluster) ----
# module load cuda/12.1
# mamba activate your_env

cd /home/kdoherty/light_subspace_removal

mkdir -p results/light_subspace_removal

# Sweep parameters
KS=(0 1 2 3 5 6 7 8 16 32)
FOLDS=(0 1 2 3 4)
OUT_SIZE=224         # supervision size; must be a 2^n multiple of inferred token grid

# Map array index -> (fold, k)
IDX=${SLURM_ARRAY_TASK_ID}
NUM_K=${#KS[@]}
FOLD=${FOLDS[$(( IDX / NUM_K ))]}
K=${KS[$(( IDX % NUM_K ))]}

echo "Running fold=${FOLD}  k=${K}  out_size=${OUT_SIZE}"

python -u light_subspace_removal/scripts/train_k_pcs.py \
  --fold ${FOLD} \
  --k ${K} \
  --outdir results/light_subspace_removal \
  --epochs 50 \
  --batch_size 32 \
  --out_size ${OUT_SIZE}