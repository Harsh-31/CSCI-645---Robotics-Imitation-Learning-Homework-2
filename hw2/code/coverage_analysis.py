"""Does student failure depend on how far an eval initial condition is from
the training demonstrations? (Q6: coverage / covariate shift.)

For every eval episode and dataset size n we compute the distance from its
initial box xy position to the nearest training demo's initial box xy position
(within the n-demo subset), then relate it to the student's per-episode success
pooled over the 3 training seeds. Initial conditions are recomputed from the
reset keys (cheap, no simulation), so this runs on a laptop CPU.

Usage:
  python code/coverage_analysis.py
"""

import json
import pathlib
import sys

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

RES = common.ROOT / "results"
SIZES = [25, 50, 100]
SEEDS = [0, 1, 2]
BINS = [0.0, 0.02, 0.04, 0.06, 1.0]  # nearest-demo distance bins [m]


def initial_conditions(env, keys):
  states = jax.device_get(jax.jit(jax.vmap(env.reset))(keys))
  box = np.asarray(common.box_pos_from_qpos(env, states.data.qpos))[:, :2]
  return box, np.asarray(states.info["target_pos"])


def rank(x):
  r = np.empty(len(x))
  r[np.argsort(x, kind="stable")] = np.arange(len(x))
  return r


def main():
  env = common.load_env()
  split = json.loads((common.ROOT / "configs" / "split_baseline.json").read_text())
  n_demo = max(split["train_idx"] + split["val_idx"]) + 1
  # Demo episode i was collected from reset key PRNGKey(DEMO_SEED_START + i).
  demo_box, _ = initial_conditions(env, common.demo_keys(common.DEMO_SEED_START, n_demo))
  eval_box, _ = initial_conditions(env, common.eval_keys())

  out, fig = {}, plt.figure(figsize=(10, 4))
  ax_s, ax_n = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
  centres = 0.5 * (np.array(BINS[:-1]) + np.minimum(np.array(BINS[1:]), 0.08))
  for c, n in zip(["C3", "C2", "C0"], SIZES):
    train_box = demo_box[split["subsets"][str(n)]]
    nn = np.linalg.norm(eval_box[:, None] - train_box[None], axis=-1).min(1)
    succ, never = [], []
    for s in SEEDS:
      pe = json.loads((RES / "eval" / f"act_baseline_n{n}_s{s}.json").read_text())["per_episode"]
      succ.append(np.asarray(pe["success"], float))
      eps = [{k: v[i] for k, v in pe.items()} for i in range(len(pe["success"]))]
      never.append(np.array([common.failure_category(e) == "never_reached_box"
                             for e in eps], float))
    succ, never = np.mean(succ, 0), np.mean(never, 0)  # per eval episode
    idx = np.digitize(nn, BINS[1:-1])
    rows = []
    for b in range(len(BINS) - 1):
      m = idx == b
      rows.append({"bin_m": [BINS[b], BINS[b + 1]], "n_episodes": int(m.sum()),
                   "success": float(succ[m].mean()) if m.any() else None,
                   "never_reached_box": float(never[m].mean()) if m.any() else None})
    rho = float(np.corrcoef(rank(nn), rank(succ))[0, 1])
    out[n] = {"mean_nn_dist_m": float(nn.mean()), "spearman_nn_vs_success": rho,
              "bins": rows}
    ok = [r["n_episodes"] >= 5 for r in rows]
    x = centres[ok]
    ax_s.plot(x, [r["success"] for r, k in zip(rows, ok) if k], "o-", c=c, label=f"n={n}")
    ax_n.plot(x, [r["never_reached_box"] for r, k in zip(rows, ok) if k], "o-", c=c,
              label=f"n={n}")

  for a, t in [(ax_s, "Student success"), (ax_n, "'Never reached box' failures")]:
    a.set(xlabel="eval box start: distance to nearest training demo [m]",
          ylabel="fraction of episodes (3 seeds pooled)", title=t, ylim=(-0.02, 1.02))
    a.set_xticks(centres, ["<2 cm", "2-4 cm", "4-6 cm", ">6 cm"])
    a.legend()
  fig.tight_layout()
  fig.savefig(RES / "coverage_vs_success.png", dpi=150)
  common.save_json(RES / "coverage_analysis.json", out)

  lines = ["| Demos | Mean distance to nearest demo | Spearman ρ (distance vs success) | "
           "Success <2 cm | 2–4 cm | 4–6 cm | >6 cm |", "|---:|---:|---:|---:|---:|---:|---:|"]
  for n in SIZES:
    o = out[n]
    cells = [f"{r['success']:.0%} (n={r['n_episodes']})" if r["n_episodes"] else "–"
             for r in o["bins"]]
    lines.append(f"| {n} | {o['mean_nn_dist_m'] * 100:.1f} cm | "
                 f"{o['spearman_nn_vs_success']:+.2f} | " + " | ".join(cells) + " |")
  (RES / "coverage_analysis.md").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
