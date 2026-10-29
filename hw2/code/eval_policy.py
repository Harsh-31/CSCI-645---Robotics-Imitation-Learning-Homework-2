"""Evaluate a trained ACT student on the held-out evaluation seeds.

Same reset keys (PRNGKey(10000 + i)), success definition and diagnostics as
eval_teacher.py. Never replays training trajectories: every episode is a fresh
closed-loop rollout from a held-out initial condition.

Usage:
  python code/eval_policy.py --run act_baseline_n100_s0 [--videos 2]
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from act.inference import ACTRunner, load_act  # noqa: E402
from eval_teacher import report  # noqa: E402
from render_videos import render_examples  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True, help="run name under $HW2_DATA/act/")
  p.add_argument("--n_episodes", type=int, default=common.NUM_EVAL_EPISODES)
  p.add_argument("--execution", default=None,
                 help="override config: temporal_ensemble | chunk")
  p.add_argument("--tag", default="", help="suffix for result names")
  p.add_argument("--videos", type=int, default=2,
                 help="successes and failures to render each (0 = none)")
  p.add_argument("--out_dir", default=str(common.ROOT / "results" / "eval"))
  args = p.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, ckpt = load_act(common.DATA_DIR / "act" / args.run / "model.pt", device)
  ecfg = ckpt["cfg"]["eval"]
  execution = args.execution or ecfg["execution"]
  torch.manual_seed(0)

  env = common.load_env()
  T = env._config.episode_length
  runner = ACTRunner(model, ckpt, args.n_episodes, T, device,
                     execution=execution, ensemble_m=ecfg["ensemble_m"],
                     exec_horizon=ecfg["exec_horizon"])
  t0 = time.time()
  traj = common.python_rollout(env, runner, common.eval_keys(args.n_episodes), T)
  summ = common.summarize_episodes(traj)

  name = args.run + (f"_{args.tag}" if args.tag else "")
  if execution != "temporal_ensemble" and not args.tag:
    name += f"_{execution}"
  train_args = ckpt["args"]
  res = report(name, summ, {
      "run": args.run, "dataset": train_args["dataset"],
      "n_demos": train_args["n_demos"], "train_seed": train_args["seed"],
      "execution": execution, "eval_time_s": time.time() - t0,
      "meta": common.run_metadata()})
  common.save_json(pathlib.Path(args.out_dir) / f"{name}.json", res)

  roll = common.DATA_DIR / "rollouts" / f"{name}.npz"
  roll.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(roll, qpos=traj["qpos"], target_pos=traj["target_pos"],
                      success=summ["success"], action=traj["action"],
                      box_target_dist=traj["box_target_dist"],
                      gripper_box_dist=traj["gripper_box_dist"])
  if args.videos:
    render_examples(env, roll, name, args.videos)


if __name__ == "__main__":
  main()
