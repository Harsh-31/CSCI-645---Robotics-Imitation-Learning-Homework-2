"""Train the PPO teacher for PandaPickCube.

Follows MuJoCo Playground's learning/train_jax_ppo.py: tuned hyperparameters
from mujoco_playground.config.manipulation_params.brax_ppo_config, the brax PPO
trainer and the Playground training wrapper. Differences from upstream:
  * impl="jax" (pure-JAX MJX) as in the HW README, instead of the default warp;
  * more evaluations (--num_evals) so that intermediate checkpoints exist;
  * progress is written to a CSV file (plus optional wandb).

Usage:
  python code/train_teacher.py --seed 0 [--num_timesteps 20000000]
"""

import argparse
import csv
import functools
import time

import common
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--num_timesteps", type=int, default=None,
                 help="Override tuned value (20M for PandaPickCube).")
  p.add_argument("--num_evals", type=int, default=11,
                 help="Evaluations (and checkpoints) during training.")
  p.add_argument("--out", default=str(common.DATA_DIR / "teacher"))
  p.add_argument("--wandb", action="store_true")
  # Overrides for quick CPU smoke tests only.
  p.add_argument("--num_envs", type=int, default=None)
  p.add_argument("--batch_size", type=int, default=None)
  p.add_argument("--num_minibatches", type=int, default=None)
  p.add_argument("--num_eval_envs", type=int, default=None)
  args = p.parse_args()

  out = common.pathlib.Path(args.out).resolve()
  ckpt_dir = out / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  env = common.load_env()
  eval_env = common.load_env()

  ppo_params = manipulation_params.brax_ppo_config(common.ENV_NAME, "jax")
  ppo_params.num_evals = args.num_evals
  for k in ["num_timesteps", "num_envs", "batch_size", "num_minibatches",
            "num_eval_envs"]:
    if getattr(args, k) is not None:
      ppo_params[k] = getattr(args, k)

  training_params = ppo_params.to_dict()
  network_kwargs = training_params.pop("network_factory")
  network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                      **network_kwargs)

  common.save_json(out / "config.json", {
      "env": common.ENV_NAME, "impl": "jax", "seed": args.seed,
      "ppo": ppo_params.to_dict(), "meta": common.run_metadata()})

  if args.wandb:
    import wandb
    wandb.init(project="csci645-hw2", name=f"teacher_s{args.seed}",
               config=ppo_params.to_dict())

  log_path = out / "progress.csv"
  log_f = open(log_path, "w", newline="")
  writer = None
  t0 = time.time()

  def progress(step, metrics):
    nonlocal writer
    row = {"step": step, "wall_time_s": round(time.time() - t0, 1)}
    row.update({k: float(v) for k, v in metrics.items()})
    if writer is None:
      writer = csv.DictWriter(log_f, fieldnames=list(row.keys()),
                              extrasaction="ignore")
      writer.writeheader()
    writer.writerow(row)
    log_f.flush()
    print(f"[{row['wall_time_s']:7.1f}s] step {step:>10d}  "
          f"eval reward {row.get('eval/episode_reward', float('nan')):.2f}",
          flush=True)
    if args.wandb:
      wandb.log(row, step=step)

  _, params, _ = ppo.train(
      environment=env,
      eval_env=eval_env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      network_factory=network_factory,
      progress_fn=progress,
      seed=args.seed,
      save_checkpoint_path=str(ckpt_dir),
      **training_params,
  )
  log_f.close()
  print(f"Done in {time.time() - t0:.0f}s. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
  main()
