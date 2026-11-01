#!/bin/bash
# Sourced by every job script: activates the env and sets shared variables.
# Override any of these before `sbatch` (sbatch exports your environment).
export HW2_ROOT=${HW2_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # hw2/ directory
export HW2_DATA=${HW2_DATA:-/scratch1/$USER/hw2}
export HW2_VENV=${HW2_VENV:-$HOME/envs/hw2}

source "$HW2_VENV/bin/activate"
export MUJOCO_GL=egl                        # headless rendering (fallback: osmesa)
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # let JAX and PyTorch share the GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}
# Login nodes cap per-user threads; OpenBLAS would otherwise try one per core.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export WANDB_MODE=${WANDB_MODE:-offline}    # `wandb sync $HW2_DATA/wandb/...` later
export WANDB_DIR=$HW2_DATA/wandb
mkdir -p "$HW2_DATA" "$WANDB_DIR" "$HW2_ROOT/logs"
cd "$HW2_ROOT"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
