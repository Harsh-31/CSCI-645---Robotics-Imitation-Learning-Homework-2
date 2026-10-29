#!/bin/bash
#SBATCH --job-name=hw2_teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Trains the PPO teacher, then evaluates every checkpoint and the zero/random
# reference policies on the held-out eval seeds.
set -euo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
source slurm/env.sh

python code/train_teacher.py --seed 0 --out "$HW2_DATA/teacher"
python code/eval_teacher.py --policy teacher --ckpt "$HW2_DATA/teacher/checkpoints" --all_checkpoints
python code/eval_teacher.py --policy zero
python code/eval_teacher.py --policy random
python code/render_videos.py --rollout "$(ls -d $HW2_DATA/rollouts/teacher_*.npz | sort | tail -1)" --n 1 --prefix teacher
