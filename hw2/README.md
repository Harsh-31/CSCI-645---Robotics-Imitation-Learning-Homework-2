# CSCI 645 HW2: Imitation Learning on PandaPickCube

Experiment: **ACT (state-only)** trained on PPO-teacher demonstrations in
MuJoCo Playground `PandaPickCube`. Ablation: **Option A, dataset quantity**
(25 / 50 / 100 demonstrations, nested subsets, 3 training seeds each).

## Layout

```text
hw2/
├── README.md                 # this file
├── requirements.txt          # pinned versions (configs/requirements_frozen.txt = full freeze)
├── code/
│   ├── common.py             # env, seeds, success definition, rollouts, rendering
│   ├── sanity_random_rollout.py
│   ├── train_teacher.py      # PPO teacher (brax + Playground tuned params)
│   ├── eval_teacher.py       # teacher / zero / random on held-out seeds
│   ├── collect_demos.py      # demonstrations + split file + replay check
│   ├── inspect_demos.py      # Part 1 stats, figures, expert videos
│   ├── act/                  # ACT model, dataset, training, inference
│   ├── eval_policy.py        # ACT student on held-out seeds (+ videos)
│   ├── render_videos.py      # MP4s from stored rollouts
│   └── analyze.py            # tables and plots
├── configs/                  # act_base.yaml, split_<dataset>.json (seeds/subsets)
├── slurm/                    # CARC job scripts
├── results/                  # eval JSON, loss CSVs, summary tables, plots
└── videos/                   # expert and student rollouts
```

Large artifacts (demonstrations, checkpoints, raw rollouts) are written to
`$HW2_DATA` (default `/scratch1/$USER/hw2` on CARC, `hw2/data` locally) and are
not part of the ZIP. They can be regenerated with the commands below.

## Conventions

| Item | Value |
|---|---|
| Environment | `PandaPickCube`, `impl="jax"`, 150 steps × 0.02 s |
| Observation | 66-d state vector from the env (raw; the student z-scores it with training-subset stats) |
| Action | 8-d in [-1, 1]: delta joint-position targets (× 0.04 rad) for 7 arm joints + gripper |
| Success | box centre within 5 cm of the target at the final step, and the episode was not terminated |
| Demo seeds | reset keys `PRNGKey(0), PRNGKey(1), …` (first 100 successes = training pool, next 10 = validation) |
| Eval seeds | reset keys `PRNGKey(10000 … 10099)`, identical for every policy |
| Subsets | nested 25 ⊂ 50 ⊂ 100, fixed permutation (seed 0), listed in `configs/split_baseline.json` |
| ACT training seeds | 0, 1, 2 |

## Reproduce on USC CARC (Discovery)

```bash
# 0) Copy the hw2/ directory to CARC, e.g. ~/csci645-hw2/hw2, then cd into it.
myaccount                                   # find your Slurm account

# 1) One-time setup on a GPU node (~10 min)
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --account=<acct>
bash slurm/setup_carc.sh                    # creates ~/envs/hw2, runs the README sanity rollout
exit

# 2) Whole pipeline: teacher -> demos -> 9 ACT runs (chained with afterok dependencies)
ACCOUNT=<acct> bash slurm/submit_all.sh
squeue -u $USER                             # monitor; logs in hw2/logs/

# 3) Tables and plots (login node is fine)
source slurm/env.sh && python code/analyze.py
```

Run the stages individually:

```bash
source slurm/env.sh
python code/train_teacher.py --seed 0                    # PPO, 20M steps
python code/eval_teacher.py --policy teacher --ckpt $HW2_DATA/teacher/checkpoints --all_checkpoints
python code/eval_teacher.py --policy zero
python code/eval_teacher.py --policy random
python code/collect_demos.py --ckpt $HW2_DATA/teacher/checkpoints --name baseline
python code/inspect_demos.py --name baseline
python code/act/train.py --dataset baseline --n_demos 100 --seed 0
python code/eval_policy.py --run act_baseline_n100_s0 --videos 2
python code/analyze.py
```

Optional Option B (noisy demonstrations):
`DATASET=noisy0.3 NOISE=0.3 sbatch --account=<acct> slurm/02_collect.slurm`, then
`DATASET=noisy0.3 sbatch --account=<acct> slurm/03_act.slurm`.

## Notes and deviations

- Upstream quirk: `PandaPickCube.reset` does not run forward kinematics, so the
  xpos-based parts of the *first* observation are zeros. The teacher was trained
  on this, and we keep it unchanged for the teacher, the demos and the student. The
  initial box position for plots/diagnostics is read from `qpos`.
- brax 0.14.2 cannot reload its own PPO checkpoint config (null initialiser
  kwargs, dict-serialised observation size), so `common.load_teacher_policy`
  rebuilds the network with those entries fixed.
- `jax` is pinned to 0.9.2 because brax 0.14.2 calls `jax.device_put_replicated`,
  which was removed in jax 0.10.
- The teacher uses Playground's tuned PPO hyperparameters for PandaPickCube, with
  `impl="jax"` instead of the default `warp`, and `num_evals=11` so that
  intermediate checkpoints are saved.
- ACT is adapted from https://github.com/tonyzhaozh/act (state-only, no image
  backbone). See the header of `code/act/model.py`.
- Weights & Biases logging is optional (`--wandb`, offline by default on CARC).
  Every curve in the report also comes from the CSV files in `results/`.
