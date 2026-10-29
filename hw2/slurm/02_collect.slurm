#!/bin/bash
#SBATCH --job-name=hw2_collect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Collects the baseline demonstrations and produces the Part 1 figures/videos.
# DATASET / NOISE can be overridden, e.g. `NOISE=0.3 DATASET=noisy0.3 sbatch ...`.
set -euo pipefail
source slurm/env.sh
DATASET=${DATASET:-baseline}
NOISE=${NOISE:-0.0}

python code/collect_demos.py --ckpt "$HW2_DATA/teacher/checkpoints" \
  --name "$DATASET" --noise_std "$NOISE"
python code/inspect_demos.py --name "$DATASET"
