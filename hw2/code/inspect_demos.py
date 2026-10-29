"""Inspect and visualise a demonstration dataset (Part 1 of the HW).

Outputs (results/demos/ and videos/):
  <name>_stats.json / .md   counts, length mean/std, success fraction, ...
  <name>_frames.png         frame grid of several demonstrations
  <name>_trajectories.png   gripper/box paths, box height and box-target distance
  <name>_coverage.png       initial box/target positions: train subsets vs eval seeds
  <name>_actions.png        action time series across demos (consistency)
  expert_<name>_*.mp4       demonstration videos

Usage:
  python code/inspect_demos.py --name baseline [--no_render]
"""

import argparse
import pathlib
import sys

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from act.dataset import load_split  # noqa: E402
from render_videos import render_one  # noqa: E402


def mean_std(x):
  x = np.asarray(x, float)
  return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
          "max": float(x.max())}


def stats_block(d, idx):
  tts = d["time_to_success"][idx]
  return {
      "n_trajectories": int(len(idx)),
      "episode_length_steps": mean_std(d["length"][idx]),
      "success_fraction": float(d["success"][idx].mean()),
      "time_to_success_steps": mean_std(tts[tts > 0]) if (tts > 0).any() else None,
      "return": mean_std(d["return"][idx]),
      "final_box_target_dist_m": mean_std(d["final_box_target_dist"][idx]),
      "action_jitter": mean_std(d["action_jitter"][idx]),
      "terminated_out_of_bounds": int(d["terminated"][idx].sum()),
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--name", default="baseline")
  p.add_argument("--no_render", action="store_true")
  p.add_argument("--n_videos", type=int, default=3)
  args = p.parse_args()

  split = load_split(common.ROOT / "configs" / f"split_{args.name}.json")
  d = dict(np.load(split["file"]))
  out = common.ROOT / "results" / "demos"
  out.mkdir(parents=True, exist_ok=True)
  n_att = split["n_attempted"]
  train = np.asarray(split["train_idx"])
  dt = 0.02

  stats = {
      "dataset": args.name,
      "all_attempted_episodes": stats_block(d, np.arange(n_att)),
      "training_pool": stats_block(d, train),
      "teacher_success_rate_collection": split["teacher_success_rate_collection"],
      "filtering": "only successful episodes are used for training",
      "subsets": {k: len(v) for k, v in split["subsets"].items()},
  }
  common.save_json(out / f"{args.name}_stats.json", stats)
  a, t = stats["all_attempted_episodes"], stats["training_pool"]
  tts = t["time_to_success_steps"]
  md = [
      "| Set | # traj | length (steps) mean ± std | time-to-success (steps) mean ± std | success fraction |",
      "|---|---:|---:|---:|---:|",
      f"| All collected | {a['n_trajectories']} | {a['episode_length_steps']['mean']:.1f} ± "
      f"{a['episode_length_steps']['std']:.1f} | – | {a['success_fraction']:.2%} |",
      f"| Training pool | {t['n_trajectories']} | {t['episode_length_steps']['mean']:.1f} ± "
      f"{t['episode_length_steps']['std']:.1f} | {tts['mean']:.1f} ± {tts['std']:.1f} | "
      f"{t['success_fraction']:.2%} |",
  ]
  (out / f"{args.name}_stats.md").write_text("\n".join(md) + "\n")
  print("\n".join(md))

  # --- Trajectory overlays ---------------------------------------------------
  fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
  for i in train:
    L = d["length"][i]
    ax[0].plot(d["gripper_pos"][i, :L, 0], d["gripper_pos"][i, :L, 1], "C0-",
               alpha=0.25, lw=0.8)
    ax[0].plot(d["box_pos"][i, :L, 0], d["box_pos"][i, :L, 1], "C1-", alpha=0.25,
               lw=0.8)
    ax[0].plot(*d["target_pos"][i, :2], "rx", ms=3, alpha=0.5)
    time = np.arange(L) * dt
    ax[1].plot(time, d["box_pos"][i, :L, 2], "C1-", alpha=0.3, lw=0.8)
    ax[1].plot(time, d["gripper_pos"][i, :L, 2], "C0-", alpha=0.15, lw=0.8)
    ax[2].plot(time, d["box_target_dist"][i, :L], "k-", alpha=0.25, lw=0.8)
  ax[0].set(title="Top-down paths (blue gripper, orange box, x target)",
            xlabel="x [m]", ylabel="y [m]", aspect="equal")
  ax[1].set(title="Height (orange box, blue gripper)", xlabel="time [s]",
            ylabel="z [m]")
  ax[2].axhline(common.SUCCESS_DIST, color="r", ls="--", label="success threshold")
  ax[2].set(title="Box-target distance", xlabel="time [s]", ylabel="[m]")
  ax[2].legend()
  fig.tight_layout()
  fig.savefig(out / f"{args.name}_trajectories.png", dpi=150)
  plt.close(fig)

  # --- Initial-condition coverage -------------------------------------------
  env = common.load_env()
  ev = jax.device_get(jax.jit(jax.vmap(env.reset))(common.eval_keys()))
  ev_box = np.asarray(common.box_pos_from_qpos(env, ev.data.qpos))
  ev_tgt = np.asarray(ev.info["target_pos"])
  fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
  sizes = sorted(split["subsets"], key=int)
  for j, (title, arr, ev_arr) in enumerate([
      ("Initial box position", d["box_init_pos"], ev_box),
      ("Target position", d["target_pos"], ev_tgt)]):
    ax[j].scatter(ev_arr[:, 0], ev_arr[:, 1], marker="x", c="gray", s=18,
                  label=f"eval seeds (n={len(ev_arr)})")
    for s, c in zip(sizes[::-1], ["C0", "C2", "C3"]):
      idx = split["subsets"][s]
      ax[j].scatter(arr[idx, 0], arr[idx, 1], s=14, c=c, label=f"train n={s}")
    ax[j].set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
  ax[0].legend(fontsize=8, loc="best")
  fig.tight_layout()
  fig.savefig(out / f"{args.name}_coverage.png", dpi=150)
  plt.close(fig)

  # --- Actions across demos --------------------------------------------------
  A = d["action"].shape[-1]
  fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
  for j, axj in enumerate(axes.flat[:A]):
    for i in train[:20]:
      L = d["length"][i]
      axj.plot(np.arange(L) * dt, d["action"][i, :L, j], alpha=0.4, lw=0.8)
    axj.set(title=f"action[{j}]" + (" (gripper)" if j == A - 1 else ""),
            ylim=(-1.05, 1.05))
  for axj in axes[-1]:
    axj.set_xlabel("time [s]")
  fig.suptitle("Teacher actions for 20 training demonstrations")
  fig.tight_layout()
  fig.savefig(out / f"{args.name}_actions.png", dpi=150)
  plt.close(fig)

  if args.no_render:
    return

  # --- Frame grid ------------------------------------------------------------
  demos = train[:4]
  times = [0, 25, 50, 75, 100, 149]
  fig, axes = plt.subplots(len(demos), len(times),
                           figsize=(2.6 * len(times), 2.0 * len(demos)))
  for r, i in enumerate(demos):
    frames = common.render_episode(env, d["qpos"][i][times], d["target_pos"][i],
                                   height=240, width=320, stride=1)
    for c, (f, tt) in enumerate(zip(frames, times)):
      axes[r, c].imshow(f)
      axes[r, c].set_xticks([])
      axes[r, c].set_yticks([])
      if r == 0:
        axes[r, c].set_title(f"t = {tt * dt:.1f} s")
    axes[r, 0].set_ylabel(f"demo {i} (seed {d['seed'][i]})", fontsize=8)
  fig.tight_layout()
  fig.savefig(out / f"{args.name}_frames.png", dpi=130)
  plt.close(fig)

  for i in train[:args.n_videos]:
    render_one(env, d["qpos"][i], d["target_pos"][i],
               common.ROOT / "videos" / f"expert_{args.name}_ep{i:03d}.mp4")


if __name__ == "__main__":
  main()
