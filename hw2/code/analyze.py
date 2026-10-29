"""Aggregate evaluation results into tables and plots for the report.

Reads results/eval/*.json and results/train/*/loss.csv and writes:
  results/summary.csv, results/summary.md     per-run and per-condition tables
  results/success_vs_demos.png                student success vs # demos
  results/loss_curves.png                     train / val losses per condition
  results/failure_modes.png                   failure categories per condition

Usage:
  python code/analyze.py [--dataset baseline]
"""

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

RES = common.ROOT / "results"


def load_evals():
  rows = []
  for f in sorted((RES / "eval").glob("*.json")):
    r = json.loads(f.read_text())
    rows.append({
        "name": r["name"],
        "kind": "student" if "run" in r else "reference",
        "dataset": r.get("dataset", ""),
        "n_demos": r.get("n_demos", np.nan),
        "train_seed": r.get("train_seed", np.nan),
        "execution": r.get("execution", ""),
        "success_rate": r["success_rate"],
        "successes": r["successes"],
        "eval_episodes": r["eval_episodes"],
        "ci_lo": r["success_ci95"][0],
        "ci_hi": r["success_ci95"][1],
        "mean_final_dist": r["mean_final_box_target_dist"],
        "mean_jitter": r["mean_action_jitter"],
        "failure_categories": r["failure_categories"],
        "noise_std": r.get("noise_std", 0.0),
    })
  return pd.DataFrame(rows)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--dataset", default="baseline")
  p.add_argument("--execution", default="temporal_ensemble")
  args = p.parse_args()

  df = load_evals()
  df.drop(columns=["failure_categories"]).to_csv(RES / "summary_runs.csv",
                                                  index=False)
  ref = df[df.kind == "reference"]
  teachers = ref[ref.name.str.startswith("teacher_") & (ref.noise_std == 0)]
  teacher = teachers.sort_values("name").iloc[-1] if len(teachers) else None
  split = json.loads((common.ROOT / "configs" /
                      f"split_{args.dataset}.json").read_text())

  st = df[(df.kind == "student") & (df.dataset == args.dataset) &
          (df.execution == args.execution)]
  agg = st.groupby("n_demos").agg(
      n_seeds=("success_rate", "size"),
      student_success_mean=("success_rate", "mean"),
      student_success_std=("success_rate", "std"),
      pooled_successes=("successes", "sum"),
      pooled_episodes=("eval_episodes", "sum"),
      eval_episodes_per_run=("eval_episodes", "first"),
      mean_jitter=("mean_jitter", "mean"),
  ).reset_index().sort_values("n_demos", ascending=False)
  agg.to_csv(RES / "summary_conditions.csv", index=False)

  t_rate = teacher.success_rate if teacher is not None else np.nan
  lines = [
      f"Teacher: `{teacher['name'] if teacher is not None else 'n/a'}`, held-out "
      f"success {t_rate:.0%}; success during collection "
      f"{split['teacher_success_rate_collection']:.0%}. Student success: mean ± "
      "std over training seeds (pooled Wilson 95% CI).",
      "",
      "| Condition | Number of demonstrations | Teacher success rate | "
      "Student success rate | Evaluation episodes |",
      "|---|---:|---:|---:|---:|",
  ]
  max_n = agg.n_demos.max() if len(agg) else None
  for _, r in agg.iterrows():
    lo, hi = common.wilson_ci(int(r.pooled_successes), int(r.pooled_episodes))
    cond = "Baseline" if r.n_demos == max_n else "Ablation"
    std = 0.0 if np.isnan(r.student_success_std) else r.student_success_std
    lines.append(
        f"| {cond} | {int(r.n_demos)} | {t_rate:.0%} | "
        f"{r.student_success_mean:.1%} ± {std:.1%} [{lo:.2f}, {hi:.2f}] "
        f"({int(r.n_seeds)} seeds) | {int(r.eval_episodes_per_run)} per seed |")
  for _, r in ref.iterrows():
    lines.append(f"| Reference: {r['name']} | – | – | {r.success_rate:.1%} | "
                 f"{r.eval_episodes} |")
  (RES / "summary.md").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))

  # --- Success vs # demos ---------------------------------------------------
  if len(st):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(st.n_demos, st.success_rate, c="C0", alpha=0.5, label="per seed")
    ax.errorbar(agg.n_demos, agg.student_success_mean,
                yerr=agg.student_success_std.fillna(0), fmt="o-", c="C0",
                capsize=4, label="mean ± std")
    if teacher is not None:
      ax.axhline(t_rate, color="k", ls="--", label="teacher")
    ax.set(xlabel="# demonstrations", ylabel="success rate", ylim=(-0.02, 1.02),
           xticks=sorted(agg.n_demos), title="ACT student success (held-out seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "success_vs_demos.png", dpi=150)
    plt.close(fig)

  # --- Loss curves ----------------------------------------------------------
  runs = sorted((RES / "train").glob(f"act_{args.dataset}_n*"))
  if runs:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = {}
    for r in runs:
      n = int(r.name.split("_n")[1].split("_")[0])
      c = colors.setdefault(n, f"C{len(colors)}")
      L = pd.read_csv(r / "loss.csv")
      tr, va = L[L.split == "train"], L[L.split == "val"]
      lbl = f"n={n}" if r.name.endswith("_s0") else None
      ax[0].plot(tr.step, tr.loss.rolling(10, min_periods=1).mean(), c=c,
                 alpha=0.7, label=lbl)
      ax[1].plot(va.step, va.l1_prior, c=c, alpha=0.7, label=lbl)
    ax[0].set(title="Train loss (L1 + beta KL, smoothed)", xlabel="step",
              yscale="log")
    ax[1].set(title="Validation L1 with z = 0 (held-out demos)", xlabel="step",
              yscale="log")
    for a in ax:
      a.legend()
    fig.tight_layout()
    fig.savefig(RES / "loss_curves.png", dpi=150)
    plt.close(fig)

  # --- Failure categories ----------------------------------------------------
  cats = {}
  for _, r in st.iterrows():
    for c, k in r.failure_categories.items():
      cats.setdefault(int(r.n_demos), {}).setdefault(c, 0)
      cats[int(r.n_demos)][c] += k
  if cats:
    cdf = pd.DataFrame(cats).fillna(0).T.sort_index()
    cdf = cdf.div(cdf.sum(1), axis=0)
    ax = cdf.plot(kind="bar", stacked=True, figsize=(7, 4), colormap="tab10")
    ax.set(xlabel="# demonstrations", ylabel="fraction of eval episodes",
           title="Outcome categories (pooled over seeds)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(RES / "failure_modes.png", dpi=150)
    plt.close()


if __name__ == "__main__":
  main()
