"""Shared utilities: environment, seeds, success definition, rollouts, rendering.

Everything that must be *identical* across teacher evaluation, demonstration
collection and student evaluation lives here (env construction, seed
conventions, the success criterion and per-episode diagnostics).
"""

import json
import os
import pathlib
import subprocess
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jp
import numpy as np

ENV_NAME = "PandaPickCube"
ROOT = pathlib.Path(__file__).resolve().parents[1]  # hw2/
# Large artifacts (demos, checkpoints, rollouts) go to $HW2_DATA, e.g.
# /scratch1/$USER/hw2 on CARC. Defaults to hw2/data for local runs.
DATA_DIR = pathlib.Path(os.environ.get("HW2_DATA", ROOT / "data"))

# ---------------------------------------------------------------------------
# Seed conventions (see configs/seeds.json; kept in code so scripts agree).
# Demo collection uses reset keys PRNGKey(DEMO_SEED_START + i).
# Held-out evaluation uses PRNGKey(EVAL_SEED_START + i): disjoint from demos
# and identical for every policy we evaluate.
# ---------------------------------------------------------------------------
DEMO_SEED_START = 0
EVAL_SEED_START = 10_000
NUM_EVAL_EPISODES = 100

# ---------------------------------------------------------------------------
# Success definition. The environment has no built-in success flag, so we
# define one: at the final step the box centre is within SUCCESS_DIST metres
# of the target (mocap) position. Per-step distances are stored as well, so
# other thresholds can be reported later without re-running anything.
# ---------------------------------------------------------------------------
SUCCESS_DIST = 0.05
# Geometric thresholds used for automatic failure categorisation.
REACH_DIST = 0.02   # gripper site within 2 cm of box centre counts as "reached"
LIFT_HEIGHT = 0.02  # box raised >2 cm above its initial height counts as "lifted"


def load_env():
  """PandaPickCube with the pure-JAX MJX backend (as in the HW README)."""
  from mujoco_playground import registry
  return registry.load(ENV_NAME, config_overrides={"impl": "jax"})


def demo_keys(start: int, n: int) -> jax.Array:
  return jp.stack([jax.random.PRNGKey(s) for s in range(start, start + n)])


def eval_keys(n: int = NUM_EVAL_EPISODES) -> jax.Array:
  return demo_keys(EVAL_SEED_START, n)


def policy_key(reset_key: jax.Array) -> jax.Array:
  """Key for stochastic policy/noise sampling, derived from the reset key."""
  return jax.random.fold_in(reset_key, 1)


# ---------------------------------------------------------------------------
# Policies usable inside JAX rollouts: policy_fn(obs, key) -> action.
# ---------------------------------------------------------------------------
def resolve_checkpoint(path) -> pathlib.Path:
  """Accepts a brax step directory, or a checkpoints/ dir (-> latest step)."""
  path = pathlib.Path(path)
  steps = sorted(p for p in path.iterdir() if p.is_dir() and p.name.isdigit())
  return steps[-1] if steps else path


def load_teacher_policy(ckpt, deterministic: bool = True) -> Callable:
  """PPO teacher from a brax checkpoint. The checkpoint contains the running
  observation normaliser, so the policy consumes *raw* env observations."""
  from brax.training import checkpoint as brax_checkpoint
  from brax.training import networks as brax_networks
  from brax.training.agents.ppo import checkpoint as ppo_checkpoint
  from brax.training.agents.ppo import networks as ppo_networks
  from ml_collections import config_dict

  path = resolve_checkpoint(ckpt)
  # Equivalent to ppo_checkpoint.load_policy, except that null kwargs are
  # dropped: brax 0.14.2 saves `mean_kernel_init_fn: null` and then fails to
  # parse it (KeyError: None) in checkpoint.load_config; it also serialises
  # the flat observation spec as {"shape": [66], ...}, restored here as an int.
  raw = json.loads((path / "ppo_network_config.json").read_text())
  if isinstance(raw["observation_size"], dict) and "shape" in raw["observation_size"]:
    raw["observation_size"] = int(raw["observation_size"]["shape"][-1])
  kw = {k: v for k, v in raw["network_factory_kwargs"].items() if v is not None}
  for k in ["policy_network_kernel_init_fn", "value_network_kernel_init_fn",
            "mean_kernel_init_fn"]:
    if k in kw:
      kw[k] = brax_networks.KERNEL_INITIALIZER[kw[k]]
  if "activation" in kw:
    kw["activation"] = brax_networks.ACTIVATION[kw["activation"]]
  raw["network_factory_kwargs"] = kw
  network = brax_checkpoint.get_network(config_dict.create(**raw),
                                        ppo_networks.make_ppo_networks)
  params = ppo_checkpoint.load(path)
  inference = ppo_networks.make_inference_fn(network)(
      params, deterministic=deterministic)
  return lambda obs, key: inference(obs, key)[0]


def zero_policy(action_size: int) -> Callable:
  return lambda obs, key: jp.zeros(action_size)


def random_policy(action_size: int) -> Callable:
  """I.i.d. uniform actions in [-1, 1] (a lower-bound reference)."""
  return lambda obs, key: jax.random.uniform(key, (action_size,), minval=-1.0,
                                             maxval=1.0)


# ---------------------------------------------------------------------------
# Per-step quantities recorded alongside obs/actions.
# ---------------------------------------------------------------------------
def box_pos_from_qpos(env, qpos):
  """Box position from the free joint's qpos. Needed right after reset: the env
  does not run forward kinematics in reset(), so data.xpos is still all zeros
  there (and so are the xpos-based parts of the very first observation)."""
  adr = int(env._obj_qposadr)
  return qpos[..., adr:adr + 3]


def step_info(env, state) -> Dict[str, jax.Array]:
  box_pos = state.data.xpos[env._obj_body]
  gripper_pos = state.data.site_xpos[env._gripper_site]
  return {
      "qpos": state.data.qpos,
      "box_pos": box_pos,
      "gripper_pos": gripper_pos,
      "box_target_dist": jp.linalg.norm(state.info["target_pos"] - box_pos),
      "gripper_box_dist": jp.linalg.norm(box_pos - gripper_pos),
  }


def make_jax_rollout(env, policy_fn: Callable, episode_length: int,
                     noise_std: float = 0.0):
  """Builds a jitted, vmapped rollout for a JAX policy (teacher / baselines).

  policy_fn(obs, key) -> action in [-1, 1]. If noise_std > 0, Gaussian noise is
  added to the policy action *before* env.step and the result is clipped to
  [-1, 1]; both the clean and the executed action are returned.

  Returned arrays have shape [N, T, ...]. obs[t] is the observation from which
  action[t] was selected; step-t diagnostics (box_target_dist, ...) describe the
  state *after* executing action[t]. qpos has T+1 entries (incl. initial state).
  """

  def single(reset_key):
    state = env.reset(reset_key)

    def body(carry, _):
      state, key, alive = carry
      key, k_pol, k_noise = jax.random.split(key, 3)
      a_policy = policy_fn(state.obs, k_pol)
      a_exec = a_policy
      if noise_std > 0:
        a_exec = a_policy + noise_std * jax.random.normal(k_noise, a_policy.shape)
      a_exec = jp.clip(a_exec, -1.0, 1.0)
      nstate = env.step(state, a_exec)
      info = step_info(env, nstate)
      out = {
          "obs": state.obs,
          "action": a_exec,
          "action_policy": a_policy,
          "reward": nstate.reward,
          "alive": alive,
          **{k: v for k, v in info.items() if k != "qpos"},
          "qpos_next": info["qpos"],
      }
      alive = alive & (nstate.done == 0)
      return (nstate, key, alive), out

    init = (state, policy_key(reset_key), jp.array(True))
    _, traj = jax.lax.scan(body, init, None, length=episode_length)
    traj["qpos"] = jp.concatenate([state.data.qpos[None], traj.pop("qpos_next")])
    traj["target_pos"] = state.info["target_pos"]
    traj["box_init_pos"] = box_pos_from_qpos(env, state.data.qpos)
    return traj

  return jax.jit(jax.vmap(single))


def python_rollout(env, act_fn: Callable, reset_keys: jax.Array,
                   episode_length: int):
  """Rollout for non-JAX (PyTorch) policies: act_fn(obs_np [N, D], t) -> np
  actions [N, A]. Records the same arrays as make_jax_rollout."""
  import functools
  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  info_fn = jax.jit(jax.vmap(functools.partial(step_info, env)))

  state = reset(reset_keys)
  init = jax.device_get(info_fn(state))
  target_pos = np.asarray(state.info["target_pos"])
  alive = np.ones(reset_keys.shape[0], bool)
  rec = {k: [] for k in ["obs", "action", "reward", "alive", "box_pos",
                         "gripper_pos", "box_target_dist", "gripper_box_dist"]}
  qpos = [init["qpos"]]
  for t in range(episode_length):
    obs = np.asarray(state.obs)
    action = np.clip(np.asarray(act_fn(obs, t), np.float32), -1.0, 1.0)
    state = step(state, jp.asarray(action))
    info = jax.device_get(info_fn(state))
    rec["obs"].append(obs)
    rec["action"].append(action)
    rec["reward"].append(np.asarray(state.reward))
    rec["alive"].append(alive.copy())
    for k in ["box_pos", "gripper_pos", "box_target_dist", "gripper_box_dist"]:
      rec[k].append(info[k])
    qpos.append(info["qpos"])
    alive &= np.asarray(state.done) == 0
  traj = {k: np.stack(v, axis=1) for k, v in rec.items()}
  traj["qpos"] = np.stack(qpos, axis=1)
  traj["target_pos"] = target_pos
  traj["box_init_pos"] = np.asarray(box_pos_from_qpos(env, init["qpos"]))
  return traj


def summarize_episodes(traj: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  """Per-episode labels/diagnostics from batched rollout arrays [N, T, ...]."""
  alive = np.asarray(traj["alive"])
  length = alive.sum(1)  # steps executed before termination (<= T)
  idx = np.arange(len(length))
  last = np.maximum(length - 1, 0)
  d = np.asarray(traj["box_target_dist"])
  terminated = length < alive.shape[1]
  final_dist = d[idx, last]
  success = (final_dist < SUCCESS_DIST) & ~terminated
  # First step at which the box is within SUCCESS_DIST of the target (or -1).
  within = (d < SUCCESS_DIST) & alive
  time_to_success = np.where(within.any(1), within.argmax(1) + 1, -1)
  box_z = np.asarray(traj["box_pos"])[..., 2]
  box_z0 = np.asarray(traj["box_init_pos"])[:, 2]
  lift = np.where(alive, box_z, -np.inf).max(1) - box_z0
  a = np.asarray(traj["action"])
  jitter = np.abs(np.diff(a, axis=1)).mean(axis=(1, 2))
  return {
      "length": length,
      "terminated": terminated,
      "success": success,
      "final_box_target_dist": final_dist,
      "min_box_target_dist": np.where(alive, d, np.inf).min(1),
      "min_gripper_box_dist": np.where(alive, traj["gripper_box_dist"], np.inf).min(1),
      "max_box_lift": lift,
      "time_to_success": time_to_success,
      "action_jitter": jitter,
      "return": (np.asarray(traj["reward"]) * alive).sum(1),
  }


def failure_category(ep: Dict[str, float]) -> str:
  """Coarse, automatic failure label; confirm by watching the videos."""
  if ep["success"]:
    return "success"
  if ep["terminated"]:
    return "box_out_of_bounds"
  if ep["min_gripper_box_dist"] > REACH_DIST:
    return "never_reached_box"
  if ep["max_box_lift"] < LIFT_HEIGHT:
    return "reached_no_lift"
  if ep["min_box_target_dist"] < SUCCESS_DIST:
    return "reached_target_then_lost"
  return "lifted_missed_target"


# ---------------------------------------------------------------------------
# Rendering from stored qpos (+ target pose) so that any saved episode can be
# rendered later, offline, without re-simulating.
# ---------------------------------------------------------------------------
def render_episode(env, qpos: np.ndarray, target_pos: np.ndarray,
                   height: int = 480, width: int = 640, stride: int = 2,
                   camera: Optional[str] = None):
  import mujoco
  model = env.mj_model
  renderer = mujoco.Renderer(model, height=height, width=width)
  data = mujoco.MjData(model)
  mocap_id = env._mocap_target
  frames = []
  for q in np.asarray(qpos)[::stride]:
    data.qpos[:] = q
    data.qvel[:] = 0
    data.mocap_pos[mocap_id] = target_pos
    data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera if camera is not None else -1)
    frames.append(renderer.render())
  renderer.close()
  return frames


def write_video(path, frames, env, stride: int = 2):
  import shutil
  import mediapy as media
  if shutil.which("ffmpeg") is None:
    # No system ffmpeg on CARC nodes: use the static binary from imageio-ffmpeg.
    import imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(path), frames, fps=round(1.0 / env.dt / stride))


# ---------------------------------------------------------------------------
# Reproducibility helpers.
# ---------------------------------------------------------------------------
def run_metadata() -> Dict[str, str]:
  import importlib.metadata as md
  meta = {}
  for pkg in ["jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "brax",
              "flax", "torch", "numpy"]:
    try:
      meta[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
      pass
  meta["jax_devices"] = str(jax.devices())
  try:
    meta["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    meta["gpu"] = "none"
  meta["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
  meta["hostname"] = os.uname().nodename
  return meta


def save_json(path, obj):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  def default(o):
    if isinstance(o, (np.integer,)):
      return int(o)
    if isinstance(o, (np.floating,)):
      return float(o)
    if isinstance(o, np.ndarray):
      return o.tolist()
    return str(o)

  path.write_text(json.dumps(obj, indent=2, default=default))


def wilson_ci(k: int, n: int, z: float = 1.96):
  if n == 0:
    return (0.0, 0.0)
  p = k / n
  denom = 1 + z**2 / n
  centre = (p + z**2 / (2 * n)) / denom
  half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
  return (max(0.0, centre - half), min(1.0, centre + half))
