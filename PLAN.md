# HW2 Project Plan — Imitation Learning on PandaPickCube (USC CARC)

**Deadline:** Fri Oct 30, 11:59 PM PT. Today is Wed Oct 28, so we have about 2.5 days. The plan is built around that: use the cheapest correct choices first and add extras only if there's time left.

---

## 1. Key decisions (made up front to save time)

| Decision | Choice | Why |
|---|---|---|
| Environment | `PandaPickCube` (MuJoCo Playground, `impl="jax"`) | Recommended. The PPO config and reference video code already exist. |
| Observations | State-based (the env's `state.obs` vector) | Required. Fast to train, no rendering needed in the training loop. |
| Teacher | **PPO (RL teacher)** via Playground's `learning/train_jax_ppo.py` (or reuse the HW1 checkpoint if you have one) | Playground ships tuned PPO params for PandaPickCube. A scripted expert in joint-delta action space would need IK, which is slower to build. |
| IL algorithm | **ACT (state-only)**, PyTorch, adapted from [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | Single forward pass, so eval is fast. L1 + KL loss is simple. With state input we drop the ResNet backbone. Diffusion Policy is the fallback (see §10). |
| Ablation | **Option A: dataset quantity, 25 / 50 / 100 demos** (nested subsets) | No extra data collection, cleanest comparison, all runs fit in a few short GPU jobs. Option B (noisy demos) is the stretch goal. |
| Rigor add-on | 3 training seeds per condition, same 50 to 100 eval seeds | Cheap for state-based ACT. Gives error bars, which strengthens the analysis grade. |

---

## 2. USC CARC setup (Discovery cluster)

### 2.1 Storage layout
- Code: `/home1/$USER/csci645-hw2/` (small, backed up)
- Data, checkpoints, videos, wandb: `/scratch1/$USER/hw2/` (large, fast, **not backed up**, so copy final results back)
- Conda env: `/project/<account>/$USER/envs/hw2` or `/scratch1/$USER/envs/hw2`. Avoid filling the home quota.

### 2.2 Environment install (on a GPU node, not the login node)
```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --account=<your_account>
module purge
module load conda            # or install micromamba in /home1 if conda module is unavailable
conda create -p /scratch1/$USER/envs/hw2 python=3.11 -y
conda activate /scratch1/$USER/envs/hw2

pip install -U "jax[cuda12]"          # bundles its own CUDA, no CUDA module needed
pip install playground                 # MuJoCo Playground (pins mujoco / mujoco-mjx / brax)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install mediapy matplotlib pandas wandb einops tqdm

# sanity checks
python -c "import jax; print(jax.devices())"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
MUJOCO_GL=egl python code/sanity_random_rollout.py   # README's random-action script
pip freeze > configs/requirements_frozen.txt
```
**Verify:** `myaccount` shows your Slurm account. `sinfo -p gpu` (or CARC's `noderes`) shows the available GPU types (A100/A40/L40S/V100). Any of them works here.

### 2.3 Env vars every job should set
```bash
export MUJOCO_GL=egl                       # headless rendering on GPU nodes (fallback: osmesa, slow)
export XLA_PYTHON_CLIENT_PREALLOCATE=false # let JAX and PyTorch share one GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export WANDB_DIR=/scratch1/$USER/hw2/wandb # or WANDB_MODE=offline, then `wandb sync` later
```

### 2.4 Slurm templates (in `slurm/`)
- `train_teacher.slurm`: 1 GPU, ~1 to 2 h
- `collect.slurm`: 1 GPU, ~15 min
- `train_act.slurm`: **job array** over `{25,50,100} × {seed 0,1,2}` = 9 tasks, ~30 to 60 min each
- `eval.slurm`: job array matching the above, plus video rendering

Submit early because queue waits on `gpu` can be long. Use `--time` close to the real need, since shorter jobs get scheduled sooner. Develop and debug on CPU on the Mac (JAX + MuJoCo run fine on macOS CPU) with tiny settings. Use CARC only for real runs.

---

## 3. Repository / submission layout

```text
LASTNAME_FIRSTNAME_HW2/
├── report.pdf
├── README.md                    # exact reproduce commands (CARC + local)
├── code/
│   ├── common.py                # env loading, seeds, success definition, obs/action specs
│   ├── train_teacher.py         # wraps Playground train_jax_ppo (or loads HW1 ckpt)
│   ├── eval_teacher.py          # teacher success rate on eval seeds
│   ├── collect_demos.py         # roll out teacher, save .npz dataset
│   ├── inspect_demos.py         # stats table, frame grid, demo video
│   ├── act/                     # model.py (adapted ACT), dataset.py, train.py
│   ├── eval_policy.py           # batched student eval on held-out seeds + videos
│   └── analyze.py               # results table, plots
├── configs/                     # teacher.yaml, act_base.yaml, ablation.yaml, seeds.json, requirements_frozen.txt
├── slurm/                       # *.slurm job scripts
├── results/                     # results.csv/json, loss curves, plots
└── videos/                      # expert demo, student success, student failure
```

---

## 4. Phase-by-phase plan

### Phase 0: Setup and environment understanding (Wed evening, ~2 h)
- [ ] Set up the CARC env (§2), run the README random-action script, save the MP4.
- [ ] Print and record: `env.observation_size`, `env.action_size`, `env.dt`, episode length, the action-scaling convention (actions in [-1, 1] mapped to joint-position deltas + gripper; check `env._config`), and the reward terms in `state.metrics`.
- [ ] **Define "success"** in `common.py` and use it everywhere (teacher eval, dataset labels, student eval). For example: cube within X cm of the target position at episode end, or for ≥ N consecutive steps. Base it on the env's own box/target positions, and document the threshold in the report.
- [ ] Fix seed conventions in `configs/seeds.json`:
  - demo collection keys: `PRNGKey(0..99)` (plus extras so failures can be replaced)
  - **held-out eval keys: `PRNGKey(10_000..10_099)`**, disjoint from demo seeds and identical across all conditions
  - training seeds: `{0, 1, 2}`

### Phase 1: Teacher (Wed night, runs overnight)
- [ ] Train PPO with Playground's tuned `PandaPickCube` params (`train_jax_ppo.py --env_name=PandaPickCube`). Save checkpoints at several stages (useful for Option C if you pivot).
- [ ] `eval_teacher.py`: deterministic (mean) actions on the 100 eval seeds and report the success rate. Target ≥ 80 to 90%. If it's low, train longer before collecting.
- **Gotcha:** Playground PPO uses observation normalization. The checkpoint includes normalizer stats, so the inference function handles it. **Store the raw env obs** in the dataset, not the normalized obs.

### Phase 2: Collect and inspect demos (Thu morning, ~2 h)
- [ ] `collect_demos.py`: `jax.vmap` over N envs, `jit` step, deterministic teacher actions, full episode length.
  - Save per step: `obs[t]`, `action[t]` (the exact clipped action passed to `env.step` at `obs[t]`), `reward[t]`, `done[t]`. Save per episode: `seed`, `success`, `length`, and the cube init pose.
  - Format: one `demos_baseline.npz` (padded arrays + `lengths`) or HDF5. Save collection config + git hash alongside.
  - **Alignment check:** assert `obs[t]` is the obs *before* `action[t]` was applied. Replay a few stored action sequences from the same seed and confirm the trajectory matches (determinism check).
- [ ] Decide the filtering policy: **keep successful episodes only** for training. Report the raw teacher success rate and how many were filtered. Collect from more seeds until you have 100 successes.
- [ ] `inspect_demos.py`:
  - Table: #traj, mean ± std length (if episodes don't terminate early, use the *time-to-success* as the "length"), success fraction.
  - Figure: a frame grid (e.g. 4 demos × 6 timesteps) and a plot of end-effector/cube trajectories overlaid across demos.
  - Video: 1 to 3 expert demos (`env.render` → `mediapy.write_video`).
  - Initial-condition coverage: scatter of cube initial xy positions for train vs eval seeds.
  - Notes for discussion: consistency of the grasp approach, hesitations or regrasps, jitter from the PPO policy, idle actions after success.
- [ ] Create nested subsets: `idx_25 ⊂ idx_50 ⊂ idx_100`, with the permutation fixed by seed and saved in `configs/`.

### Phase 3: ACT implementation and baseline training (Thu, ~4 to 6 h incl. debugging)
- [ ] Adapt ACT for state input:
  - Encoder (CVAE, training only): `[CLS, obs_token, action_chunk_tokens]` → latent `z` (dim 32).
  - Decoder: transformer with tokens from `obs` (linear projection) + `z`, and `k` learned query embeddings → `k × action_dim`.
  - Loss: `L1(actions) + β·KL` with `β = 10` (ACT default). Pad chunks past episode end and mask them in the loss.
- [ ] Preprocessing: z-score obs and actions using stats from the **training subset only** (saved with the checkpoint). Clip actions to [-1, 1] at execution.
- [ ] Starting hyperparameters (record all in `configs/act_base.yaml`):
  - chunk size `k = 20` (sweep {10, 20, 50} only if time allows)
  - `d_model = 256`, 4 encoder / 4 decoder layers, 8 heads, dropout 0.1
  - AdamW, lr `1e-4`, weight decay `1e-4`, batch 256
  - Train for a **fixed number of gradient steps** (e.g. 20k to 50k), the same for every dataset size, so the ablation only changes data. Note in the report that smaller datasets therefore see more epochs.
- [ ] Execution in the sim: query the policy every step and use **temporal ensembling** (exponential weights, `m = 0.01`) as in the ACT paper. Simpler alternative: execute the full chunk open-loop and re-query every `k` steps. Pick one and document it.
- [ ] Logging: wandb (train loss, L1, KL, val loss on 10% held-out demos). Also save loss curves as PNG + CSV in `results/` so the report doesn't need logins.

### Phase 4: Evaluation (Thu night)
- [ ] `eval_policy.py`: batched eval with `jax.vmap` over the 50 to 100 eval seeds. At each step, move obs to torch, run batched ACT inference, apply temporal ensembling, convert back to JAX, step. (Small model, so torch-on-CPU is also fine if GPU sharing causes trouble.)
- [ ] Report the success rate and a Wilson 95% CI per run. Report mean ± std across the 3 training seeds per condition.
- [ ] Render videos: ≥ 1 success and ≥ 1 failure per condition, plus a teacher rollout on the same eval seed for side-by-side comparison.

### Phase 5: Ablation runs (Thu, in parallel with Phase 3/4 via Slurm array)
- [ ] Train `{25, 50, 100} × {seed 0, 1, 2}` with an identical config, then evaluate all on the same eval seeds.
- [ ] Results table (required format):

| Condition | # demos | Teacher success | Student success (mean ± std over 3 seeds) | Eval episodes |
|---|---:|---:|---:|---:|
| Baseline | 100 | x% | y ± s % | 100 |
| Ablation | 25 | x% | ... | 100 |
| (optional) | 50 | x% | ... | 100 |

- [ ] Plot: student success vs # demos (with error bars); training/val loss curves per condition.
- [ ] **Stretch (Option B, only if ahead of schedule):** collect 100 demos with Gaussian noise `σ ∈ {0.1, 0.3}` added to teacher actions *before* `env.step`, clipped to [-1, 1]. Store the **executed noisy action** (or the clean teacher action, which is a DART-like variant). Choose one and document it. Report teacher success under noise vs student success.

### Phase 6: Failure analysis (Fri morning)
- [ ] Watch failure videos and categorize the failures over the eval set: e.g. missed grasp, slip while lifting, oscillation/hesitation near the cube, wrong target, timeout. Make a bar chart of failure categories per condition.
- [ ] Relate the failures to class concepts: **compounding errors / covariate shift** (the student drifts into states not covered by the demos, which is worse with 25 demos), **multimodality** (ACT's CVAE vs plain BC), and the effect of action chunking and temporal ensembling on jitter.
- [ ] Optional quick diagnostic: compare cube init positions for student failures against the training-set coverage scatter.

### Phase 7: Report and submission (Fri afternoon/evening, finish by ~8 PM for buffer)
- [ ] `report.pdf` answering Q1 to Q8 in order (see README). Include hardware (CARC GPU model, CUDA driver), library versions (jax, mujoco, mujoco-mjx, playground, brax, torch), seeds, the success definition, and deviations from upstream ACT.
- [ ] Citations: MuJoCo Playground, ACT paper + repo, PPO/Brax, any other code, and the LLM-use statement.
- [ ] `README.md` with exact commands (setup → teacher → collect → train → eval → analyze), plus the Slurm submission commands.
- [ ] Package: code, configs, results, videos (compressed MP4s), report. **Exclude** third-party libs, large datasets/checkpoints (give a regeneration script or a link), and the wandb API key / `.netrc`.
- [ ] Zip as `LASTNAME_FIRSTNAME_HW2.zip`, unzip in a temp dir to verify, run `grep -ri "api_key\|wandb_" .`, then upload to Brightspace.

---

## 5. Timeline (compressed)

| When | Work | CARC jobs running |
|---|---|---|
| **Wed Oct 28, evening** | Phase 0, start teacher training | `train_teacher` (overnight) |
| **Thu Oct 29, morning** | Teacher eval, collect demos, inspect/visualize | `collect` |
| **Thu, midday** | ACT implementation, quick 1-seed run on 100 demos to debug | `train_act` (debug) |
| **Thu, afternoon/evening** | Launch full 9-run array, write eval script | `train_act` array → `eval` array |
| **Thu, night** | Eval, videos; start report skeleton (Q1–Q3 can be written now) | `eval` |
| **Fri Oct 30, morning** | Results table/plots, failure analysis, optional Option B | (optional) noisy collect + train |
| **Fri, afternoon** | Finish report, README, package ZIP | — |
| **Fri, ≤ 11:59 PM** | Submit (aim for ~9 PM) | — |

---

## 6. Things to verify early (they most often cause silent bugs)

1. **obs/action alignment** in the dataset (off-by-one).
2. **Raw vs normalized obs:** the teacher normalizes internally, while the student uses its own normalization.
3. **Same success definition** used for teacher eval, data labels, and student eval.
4. **Eval seeds disjoint** from demo seeds, and identical across conditions.
5. **Action clipping** at execution, both for the student and for noisy teacher actions.
6. **Determinism:** fixed JAX keys, `torch.manual_seed`, and the eval env reset from the saved key list.
7. **JAX + PyTorch GPU memory:** set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. If CUDA libs conflict, run torch inference on CPU during eval.

## 7. Compute budget estimate

| Job | GPU time (est.) |
|---|---|
| PPO teacher | 0.5 to 2 h |
| Demo collection (≤ 200 episodes) | < 15 min |
| ACT training, 9 runs | ~0.5 h each, ~5 GPU-h total (run in parallel as an array) |
| Eval + videos, 9 runs | ~10 min each |
| **Total** | **~8 to 10 GPU-hours** |

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| PPO teacher success is low or training is slow | Train longer or use more envs. Last resort: scripted expert (approach above cube → descend → close gripper → lift → move to target) using the env's mocap/IK helpers if they're available. |
| CARC queue delays | Use short `--time`, request any GPU type (`--gres=gpu:1`), and develop on the Mac CPU in the meantime. |
| EGL rendering fails on a node | `MUJOCO_GL=osmesa` for video rendering only (slow but fine for a few videos). |
| ACT student gets ~0% success | Check normalization, chunk alignment, and temporal ensembling. Try `k=10`, and train longer. A negative result with good analysis is acceptable per the README. |
| Out of time | Drop the 50-demo condition and use 1 training seed. The minimum deliverable is 100 vs 25 demos with ≥ 50 eval episodes each. |

## 9. What the provided GIFs show (and what it means for us)

Both GIFs are 400×300 at 25 fps. Zero actions: 51 frames (≈ 100 env steps). Random actions: 101 frames (= 200 steps rendered every 2nd state).
- **Timestep:** playback fps = `1/env.dt/2` = 25, so the **control dt is 0.02 s** (50 Hz), and 200 steps = 4 s of sim time. Verify the episode length in `env._config`.
- **Scene:** a small **green cube on the floor** in front of and to the side of the robot (it has a shadow and reflection, so it's a physical object). A **translucent red cube floating in mid-air** (no shadow) is the **target pose** marker. The task is pick *and lift/place at the target*, not just grasp.
- **Same initial state in both GIFs** (`PRNGKey(0)`): the cube, target, and arm start at identical positions. The GIFs show no randomization, so we must check the range of cube/target initial positions ourselves (the Phase 2 coverage scatter).
- **Zero actions → the arm holds its pose** (first and last frames are nearly identical, with no sag). This means actions are **relative/delta position-control targets**, not absolute joint positions. If they were absolute, zero would drive the arm to an all-zeros joint configuration.
- **Random actions (resampled every 10 steps in [-1, 1]) → the arm drifts far away** into folded, upward-pointing configurations. Holding a constant action makes the arm keep moving, because the deltas *accumulate*. The cube is never touched, and success is 0%.

**Implications:**
1. **Errors integrate.** Since actions are deltas, a small biased error in the student's actions builds up into large pose drift over time, which is a concrete form of compounding error. Expect this to be a key failure mode, and watch for it in the failure videos.
2. **The obs must expose the current joint/ctrl state** so the student knows where the accumulated target is. Confirm this in Phase 0 by checking what `state.obs` contains.
3. **Smoothness matters.** Jittery teacher actions become jittery motion. ACT chunking + temporal ensembling should help here. Measure action jitter (mean |aₜ − aₜ₋₁|) for teacher vs student as an extra diagnostic.
4. **Add reference rows** to the results table: *zero-action* and *random-action* policies on the same eval seeds (expected ≈ 0%). This confirms the success metric isn't trivially satisfied.
5. **Video framing:** in the default camera the cube is small (~10 px). For failure analysis, render at 640×480 and consider a closer camera, so grasp errors are visible.

## 10. Alternative: Diffusion Policy (if you prefer it over ACT)
Use the state-based 1D conditional U-Net from [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) (lowdim config): `obs_horizon=2`, `pred_horizon=16`, `action_horizon=8`, DDPM with 100 train steps, and DDIM with 10 to 16 steps at inference to keep eval fast. Everything else in this plan stays the same.
