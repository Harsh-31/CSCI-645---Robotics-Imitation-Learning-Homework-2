"""Render MP4s from stored rollouts (qpos + target), no re-simulation needed.

Needs MUJOCO_GL=egl on headless GPU nodes (osmesa as a slow fallback).

Usage:
  python code/render_videos.py --rollout $HW2_DATA/rollouts/act_baseline_n100_s0.npz --n 2
  python code/render_videos.py --rollout $HW2_DATA/demos/baseline.npz --episodes 0,1,2 --prefix expert
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

VIDEO_DIR = common.ROOT / "videos"


def render_one(env, qpos, target_pos, path, **kw):
  frames = common.render_episode(env, qpos, target_pos, **kw)
  common.write_video(path, frames, env)
  print(f"wrote {path}")


def render_examples(env, rollout_path, name, n=2, **kw):
  """Renders the first n successful and first n failed episodes."""
  d = np.load(rollout_path)
  succ = np.asarray(d["success"]).astype(bool)
  for label, idx in [("success", np.flatnonzero(succ)[:n]),
                     ("failure", np.flatnonzero(~succ)[:n])]:
    for i in idx:
      render_one(env, d["qpos"][i], d["target_pos"][i],
                 VIDEO_DIR / f"{name}_{label}_ep{i:03d}.mp4", **kw)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--rollout", required=True, help=".npz with qpos, target_pos")
  p.add_argument("--n", type=int, default=2)
  p.add_argument("--episodes", default=None, help="comma-separated indices")
  p.add_argument("--prefix", default=None)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--width", type=int, default=640)
  args = p.parse_args()

  env = common.load_env()
  path = pathlib.Path(args.rollout)
  prefix = args.prefix or path.stem
  kw = {"height": args.height, "width": args.width}
  if args.episodes:
    d = np.load(path)
    for i in map(int, args.episodes.split(",")):
      tag = "success" if d["success"][i] else "failure"
      render_one(env, d["qpos"][i], d["target_pos"][i],
                 VIDEO_DIR / f"{prefix}_{tag}_ep{i:03d}.mp4", **kw)
  else:
    render_examples(env, path, prefix, args.n, **kw)


if __name__ == "__main__":
  main()
