#!/bin/bash
#SBATCH --job-name=hw2_act
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
# One array task per (dataset size, training seed): 3 sizes x 3 seeds.
# Each task trains ACT and evaluates it on the 100 held-out eval seeds.
set -euo pipefail
source slurm/env.sh
DATASET=${DATASET:-baseline}
SIZES=(100 50 25)
N=${SIZES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
RUN=act_${DATASET}_n${N}_s${SEED}
VIDEOS=$([ "$SEED" -eq 0 ] && echo 2 || echo 0)

python code/act/train.py --config configs/act_base.yaml --dataset "$DATASET" \
  --n_demos "$N" --seed "$SEED" --run_name "$RUN"
python code/eval_policy.py --run "$RUN" --videos "$VIDEOS"
