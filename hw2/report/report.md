# CSCI 645 HW2: Imitation Learning for Robot Manipulation

**Name:** … **Environment:** MuJoCo Playground `PandaPickCube` (state observations)
**Algorithm:** ACT (state-only) **Ablation:** Option A, dataset quantity (25 / 50 / 100)

> Fill every `TODO` from the files in `results/`. Convert to PDF with e.g.
> `pandoc report.md -o report.pdf` (or paste into a doc and export).

## 1. Task and demonstration source

- Task: Franka Panda must grasp a cube whose initial position is uniform in a
  40 × 40 cm square and bring it to a target sampled 20–40 cm above the table (no
  orientation target). 150 control steps at 50 Hz.
- Observation (66-d): qpos (16), qvel (15), gripper position (3), gripper
  orientation (6), box orientation (6), box − gripper (3), target − box (3),
  target − box orientation (6), ctrl − qpos (8).
- Action (8-d, [-1, 1]): added to the current joint-position *control targets*
  after scaling by 0.04 (delta position control), so errors accumulate (see Q6).
- Teacher: PPO (brax, Playground's tuned hyperparameters, 20M env steps, seed 0),
  deterministic (mean) actions. Held-out success: **TODO** (`results/eval/teacher_*.json`).
- Teacher-quality curve over checkpoints: **TODO** (optional figure).
- Filtering: only successful episodes are used for training. **TODO**: X of Y
  collected episodes succeeded (`configs/split_baseline.json`).

## 2. Demonstration quality

- Table: `results/demos/baseline_stats.md` (**TODO** paste).
- Figures: `results/demos/baseline_frames.png`, `baseline_trajectories.png`,
  `baseline_coverage.png`, `baseline_actions.png`.
- Video: `videos/expert_baseline_ep*.mp4`.
- Discussion (**TODO**): consistency of the approach and grasp strategy, coverage
  of initial box/target positions (25 vs 100 demos vs eval seeds), errors and
  inefficiencies (hesitation, regrasps, jitter, behaviour after reaching the
  target).

## 3. Policy and training

| Item | Value |
|---|---|
| Inputs | 66-d state, z-scored with training-subset mean/std (std clipped ≥ 1e-2) |
| Outputs | chunk of k = 20 actions (8-d), z-scored; un-normalised and clipped to [-1, 1] |
| Architecture | CVAE encoder (4 layers) + transformer encoder (4) / decoder (4), d = 256, 8 heads, latent 32, 10.6M params |
| Objective | L1(chunk) + 10 · KL, padded steps masked |
| Optimiser | AdamW, lr 1e-4, wd 1e-4, batch 256, grad clip 10, 30k steps (same for all sizes) |
| Execution | query every step, temporal ensembling w_i = exp(−0.01 i) |
| Seeds | training seeds 0, 1, 2; eval seeds 10000–10099 |

- Loss curves: `results/loss_curves.png` (**TODO**).
- Changes vs upstream ACT: listed in `code/act/model.py` header.

## 4. Baseline results

- Success over 100 held-out episodes: **TODO** (mean ± std over 3 seeds).
- Videos: `videos/act_baseline_n100_s0_success_*.mp4`, `..._failure_*.mp4`.

## 5. Data ablation

`results/summary.md` (**TODO** paste) and `results/success_vs_demos.png`.

| Condition | Number of demonstrations | Teacher success rate | Student success rate | Evaluation episodes |
|---|---:|---:|---:|---:|
| Baseline | 100 | TODO | TODO | 100 × 3 seeds |
| Ablation | 25 | TODO | TODO | 100 × 3 seeds |
| (Ablation) | 50 | TODO | TODO | 100 × 3 seeds |
| Reference: zero / random actions | – | – | TODO | 100 |

Discussion (**TODO**): effect size vs seed variance and CI; relation to
coverage of initial conditions; whether validation L1 predicts closed-loop success.

## 6. Failure analysis

- Automatic categories: `results/failure_modes.png` (never reached box, reached
  but no lift, lifted but missed target, reached then lost, out of bounds).
- **TODO**: pick one mode, show a video/frames, and explain it. Candidates:
  compounding error / covariate shift (delta control integrates small errors;
  the student drifts to states that are not in the demos, worse with 25 demos);
  grasp timing errors; jitter (compare `mean_action_jitter` of teacher and student).

## 7. Reproducibility and reflection

- Commands: `README.md`. Seeds/subsets: `configs/split_baseline.json`. Config:
  `configs/act_base.yaml`. Versions: `configs/requirements_frozen.txt`.
- Hardware: **TODO** (GPU model and driver from the `meta` field of any results JSON).
- Improvements with more time: **TODO**.

## 8. Citations and LLM use

- MuJoCo Playground (Zakka et al., 2025), MuJoCo / MJX, Brax PPO.
- ACT: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware", RSS 2023; code github.com/tonyzhaozh/act.
- LLM use: **TODO** (describe honestly).
