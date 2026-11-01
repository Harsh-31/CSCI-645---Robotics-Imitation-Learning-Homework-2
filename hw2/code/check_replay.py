"""Diagnose the open-loop replay check from collect_demos.py.

Replays stored actions from the same reset keys and reports, per step, how far
the replayed state is from the stored one. Two cases can be told apart:
  * misaligned obs/actions: error is large from the first steps on;
  * numerical divergence: error is ~0 at first and only grows after contact
    (grasping is chaotic, so tiny float differences from a different XLA
    program, e.g. another batch size, get amplified).
Replaying with the *same batch size and seeds* as collection reuses the same
program and should reproduce the data almost exactly.

Usage:
  python code/check_replay.py --name baseline --batch 64
"""

import argparse

import common
import jax
import jax.numpy as jp
import numpy as np


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--name", default="baseline")
  p.add_argument("--batch", type=int, default=64,
                 help="replay the first `batch` episodes together (64 = the "
                      "collection batch size)")
  args = p.parse_args()

  d = np.load(common.DATA_DIR / "demos" / f"{args.name}.npz")
  env = common.load_env()
  n = args.batch

  def replay(key, actions):
    state = env.reset(key)

    def body(s, a):
      s = env.step(s, a)
      return s, (s.obs, s.data.qpos)

    _, (obs, qpos) = jax.lax.scan(body, state, actions)
    return obs, qpos

  keys = jp.stack([jax.random.PRNGKey(int(s)) for s in d["seed"][:n]])
  obs_next, qpos = jax.device_get(
      jax.jit(jax.vmap(replay))(keys, jp.asarray(d["action"][:n])))

  T = d["obs"].shape[1]
  # obs_next[:, t] is the obs after executing action[t] == stored obs[t + 1].
  obs_err = np.abs(obs_next[:, :T - 1] - d["obs"][:n, 1:]).max(-1)  # [n, T-1]
  qpos_err = np.abs(qpos - d["qpos"][:n, 1:]).max(-1)               # [n, T]

  print(f"batch {n}: max obs err {obs_err.max():.2e}, max qpos err "
        f"{qpos_err.max():.2e}")
  print("median over episodes of max qpos err at step t:")
  for t in [0, 1, 2, 5, 10, 20, 30, 40, 60, 100, 149]:
    print(f"  t={t:3d}: qpos {np.median(qpos_err[:, t]):.2e}  "
          f"obs {np.median(obs_err[:, min(t, T - 2)]):.2e}")

  first = np.where((qpos_err > 1e-3).any(1), (qpos_err > 1e-3).argmax(1), -1)
  contact = np.where((d["gripper_box_dist"][:n] < common.REACH_DIST).any(1),
                     (d["gripper_box_dist"][:n] < common.REACH_DIST).argmax(1), -1)
  diverged = first >= 0
  print(f"episodes whose qpos diverges by >1e-3: {diverged.sum()}/{n}")
  if diverged.any():
    print(f"  first divergence step (median): {np.median(first[diverged]):.0f}; "
          f"first gripper-box contact step (median): "
          f"{np.median(contact[diverged]):.0f}")
    print(f"  diverged before contact: {(first[diverged] < contact[diverged]).sum()}")


if __name__ == "__main__":
  main()
