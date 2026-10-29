"""Collect PPO-teacher demonstrations for PandaPickCube.

Rolls out the deterministic teacher on reset keys PRNGKey(0), PRNGKey(1), ...
until `n_train + n_val` successful episodes exist. *All* episodes (successful
and failed) are stored with success labels; the split file lists which ones
are used for training:
  train pool : the first n_train successful episodes in seed order (baseline)
  val        : the next n_val successful episodes (val loss only, never trained on)
  subsets    : nested random subsets of the train pool (25 c 50 c 100)

Stored per step: obs[t], action[t] (the exact clipped action passed to
env.step at obs[t]), action_policy[t] (teacher output before noise), reward,
alive mask and geometric diagnostics; qpos for rendering.

Usage:
  python code/collect_demos.py --ckpt $HW2_DATA/teacher/checkpoints --name baseline
  python code/collect_demos.py --ckpt ... --name noisy0.3 --noise_std 0.3   # Option B
"""

import argparse
import time

import common
import jax
import jax.numpy as jp
import numpy as np


def verify_replay(env, data, n=4):
  """Replay stored actions open-loop from the same reset key and compare qpos.
  Checks (a) obs/action alignment and (b) simulator determinism."""

  def replay(key, actions):
    state = env.reset(key)

    def body(s, a):
      s = env.step(s, a)
      return s, (s.obs, s.data.qpos)

    _, (obs, qpos) = jax.lax.scan(body, state, actions)
    return state.obs, obs, qpos

  replay = jax.jit(jax.vmap(replay))
  keys = jp.stack([jax.random.PRNGKey(int(s)) for s in data["seed"][:n]])
  obs0, obs_next, qpos = jax.device_get(replay(keys, jp.asarray(data["action"][:n])))
  obs_err = np.abs(obs0 - data["obs"][:n, 0]).max()
  # obs[t+1] stored must equal the obs produced by executing action[t].
  T = data["obs"].shape[1]
  align_err = np.abs(obs_next[:, :T - 1] - data["obs"][:n, 1:]).max()
  qpos_err = np.abs(qpos - data["qpos"][:n, 1:]).max()
  print(f"[verify] initial obs err {obs_err:.2e}, obs alignment err "
        f"{align_err:.2e}, replay qpos err {qpos_err:.2e}")
  return {"initial_obs_err": obs_err, "obs_alignment_err": align_err,
          "replay_qpos_err": qpos_err}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", required=True)
  p.add_argument("--name", default="baseline")
  p.add_argument("--n_train", type=int, default=100)
  p.add_argument("--n_val", type=int, default=10)
  p.add_argument("--batch", type=int, default=64)
  p.add_argument("--max_episodes", type=int, default=1024)
  p.add_argument("--noise_std", type=float, default=0.0)
  p.add_argument("--subset_sizes", default="25,50,100")
  p.add_argument("--subset_seed", type=int, default=0)
  args = p.parse_args()

  env = common.load_env()
  T = env._config.episode_length
  policy = common.load_teacher_policy(args.ckpt, deterministic=True)
  rollout = common.make_jax_rollout(env, policy, T, noise_std=args.noise_std)

  need = args.n_train + args.n_val
  chunks, seed, n_success = [], common.DEMO_SEED_START, 0
  t0 = time.time()
  while n_success < need and seed - common.DEMO_SEED_START < args.max_episodes:
    seeds = np.arange(seed, seed + args.batch)
    keys = jp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    traj = jax.device_get(rollout(keys))
    summ = common.summarize_episodes(traj)
    chunks.append({**traj, **summ, "seed": seeds})
    n_success += int(summ["success"].sum())
    seed += args.batch
    print(f"seeds <{seed}: {n_success} successes "
          f"({time.time() - t0:.0f}s)", flush=True)
  if n_success < need:
    raise RuntimeError(f"Only {n_success} successes in {args.max_episodes} "
                       "episodes; train the teacher longer.")

  data = {k: np.concatenate([np.asarray(c[k]) for c in chunks])
          for k in chunks[0]}
  succ_idx = np.flatnonzero(data["success"])
  train_idx = succ_idx[:args.n_train]
  val_idx = succ_idx[args.n_train:need]
  # Teacher statistics are reported over every episode attempted up to and
  # including the one that produced the last needed success.
  n_attempted = int(val_idx[-1]) + 1
  rng = np.random.default_rng(args.subset_seed)
  perm = rng.permutation(train_idx)
  subsets = {int(s): sorted(perm[:int(s)].tolist())
             for s in args.subset_sizes.split(",")}

  out = common.DATA_DIR / "demos" / f"{args.name}.npz"
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out, **data)
  verify = verify_replay(env, {k: data[k][train_idx] for k in
                               ["seed", "obs", "action", "qpos"]})

  split = {
      "dataset": args.name,
      "file": str(out),
      "teacher_checkpoint": str(common.resolve_checkpoint(args.ckpt)),
      "noise_std": args.noise_std,
      "action_clip": [-1.0, 1.0],
      "n_attempted": n_attempted,
      "n_success_attempted": int(data["success"][:n_attempted].sum()),
      "teacher_success_rate_collection": float(data["success"][:n_attempted].mean()),
      "train_idx": train_idx.tolist(),
      "val_idx": val_idx.tolist(),
      "subsets": subsets,
      "subset_seed": args.subset_seed,
      "verify": verify,
      "obs_dim": int(data["obs"].shape[-1]),
      "action_dim": int(data["action"].shape[-1]),
      "episode_length": T,
      "meta": common.run_metadata(),
  }
  common.save_json(common.ROOT / "configs" / f"split_{args.name}.json", split)
  print(f"Saved {len(data['seed'])} episodes to {out}; teacher success during "
        f"collection {split['teacher_success_rate_collection']:.2%} "
        f"({split['n_success_attempted']}/{n_attempted}).")


if __name__ == "__main__":
  main()
