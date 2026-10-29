#!/bin/bash
# One-time environment setup. Run inside an interactive GPU allocation:
#   salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --account=<acct>
#   bash slurm/setup_carc.sh
set -euo pipefail
export HW2_VENV=${HW2_VENV:-$HOME/envs/hw2}
export HW2_ROOT=${HW2_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}

# uv: standalone Python/venv manager (no conda/module needed).
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv "$HW2_VENV" --python 3.11
source "$HW2_VENV/bin/activate"
uv pip install -r "$HW2_ROOT/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match
uv pip freeze > "$HW2_ROOT/configs/requirements_frozen.txt"

cd "$HW2_ROOT"
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import torch; print('torch cuda:', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
# First env load downloads mujoco_menagerie (robot meshes) into site-packages.
python code/sanity_random_rollout.py
echo "Setup OK. Video: $HW2_ROOT/videos/sanity_random_actions.mp4"
