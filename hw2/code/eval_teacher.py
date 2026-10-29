"""Evaluate JAX policies (PPO teacher, zero-action, random) on held-out seeds.

Uses exactly the same eval reset keys, success definition and per-episode
diagnostics as the student evaluation (eval_policy.py).

Usage:
  python code/eval_teacher.py --policy teacher --ckpt $HW2_DATA/teacher/checkpoints
  python code/eval_teacher.py --policy teacher --ckpt .../checkpoints --all_checkpoints
  python code/eval_teacher.py --policy zero
  python code/eval_teacher.py --policy random
"""

import argparse
import time

import common
import jax
import numpy as np


def evaluate(env, policy_fn, n_episodes, noise_std=0.0):
  rollout = common.make_jax_rollout(env, policy_fn, env._config.episode_length,
                                    noise_std=noise_std)
  traj = jax.device_get(rollout(common.eval_keys(n_episodes)))
  return traj, common.summarize_episodes(traj)


def report(name, summ, extra=None):
  n = len(summ["success"])
  k = int(summ["success"].sum())
  lo, hi = common.wilson_ci(k, n)
  cats = [common.failure_category({key: v[i] for key, v in summ.items()})
          for i in range(n)]
  res = {
      "name": name,
      "eval_episodes": n,
      "successes": k,
      "success_rate": k / n,
      "success_ci95": [lo, hi],
      "mean_final_box_target_dist": float(summ["final_box_target_dist"].mean()),
      "mean_return": float(summ["return"].mean()),
      "mean_action_jitter": float(summ["action_jitter"].mean()),
      "failure_categories": {c: cats.count(c) for c in sorted(set(cats))},
      "per_episode": {k_: v for k_, v in summ.items()},
      "eval_seed_start": common.EVAL_SEED_START,
      "success_dist": common.SUCCESS_DIST,
      **(extra or {}),
  }
  print(f"{name:>30s}: success {k}/{n} = {k / n:.2%} "
        f"(95% CI {lo:.2f}-{hi:.2f}), categories {res['failure_categories']}")
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--policy", choices=["teacher", "zero", "random"],
                 default="teacher")
  p.add_argument("--ckpt", help="brax checkpoint dir (step dir or checkpoints/)")
  p.add_argument("--all_checkpoints", action="store_true",
                 help="Evaluate every saved step (teacher-quality curve).")
  p.add_argument("--noise_std", type=float, default=0.0,
                 help="Gaussian action noise during execution (Option B).")
  p.add_argument("--n_episodes", type=int, default=common.NUM_EVAL_EPISODES)
  p.add_argument("--out_dir", default=str(common.ROOT / "results" / "eval"))
  args = p.parse_args()

  env = common.load_env()
  out_dir = common.pathlib.Path(args.out_dir)
  roll_dir = common.DATA_DIR / "rollouts"
  roll_dir.mkdir(parents=True, exist_ok=True)

  if args.policy == "teacher":
    root = common.pathlib.Path(args.ckpt)
    if args.all_checkpoints:
      ckpts = sorted(p for p in root.iterdir() if p.name.isdigit())
    else:
      ckpts = [common.resolve_checkpoint(root)]
    jobs = [(f"teacher_{c.name}" + (f"_noise{args.noise_std}" if args.noise_std else ""),
             common.load_teacher_policy(c), {"checkpoint": str(c)}) for c in ckpts]
  elif args.policy == "zero":
    jobs = [("zero_action", common.zero_policy(env.action_size), {})]
  else:
    jobs = [("random_action", common.random_policy(env.action_size), {})]

  for name, policy_fn, extra in jobs:
    t0 = time.time()
    traj, summ = evaluate(env, policy_fn, args.n_episodes, args.noise_std)
    extra = {**extra, "noise_std": args.noise_std,
             "eval_time_s": time.time() - t0, "meta": common.run_metadata()}
    common.save_json(out_dir / f"{name}.json", report(name, summ, extra))
    np.savez_compressed(roll_dir / f"{name}.npz", qpos=traj["qpos"],
                        target_pos=traj["target_pos"],
                        success=summ["success"])


if __name__ == "__main__":
  main()
