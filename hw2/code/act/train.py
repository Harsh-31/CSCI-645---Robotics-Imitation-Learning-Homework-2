"""Train an ACT policy on a demonstration subset.

Usage:
  python code/act/train.py --config configs/act_base.yaml --dataset baseline \
      --n_demos 100 --seed 0
The run name defaults to act_<dataset>_n<n_demos>_s<seed>.
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from act.dataset import ChunkSampler, Normalizer, load_episodes, load_split  # noqa: E402
from act.model import ACTPolicy, act_loss  # noqa: E402


def evaluate_loss(model, sampler, kl_weight, n=2048):
  """Val loss with posterior z (training objective) and L1 with z=0 (test-time)."""
  model.eval()
  idx = torch.arange(min(n, len(sampler)), device=sampler.device)
  with torch.no_grad():
    obs, act, pad = sampler.get(idx)
    pred, dist = model(obs, act, pad)
    loss, parts = act_loss(pred, act, pad, dist, kl_weight)
    pred0, _ = model(obs)
    valid = (~pad).unsqueeze(-1).float()
    l1_prior = ((pred0 - act).abs() * valid).sum() / (valid.sum() * act.shape[-1])
  model.train()
  return {"loss": loss.item(), **parts, "l1_prior": l1_prior.item()}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--config", default=str(common.ROOT / "configs" / "act_base.yaml"))
  p.add_argument("--dataset", default="baseline")
  p.add_argument("--n_demos", type=int, default=100)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run_name", default=None)
  p.add_argument("--num_steps", type=int, default=None, help="override config")
  p.add_argument("--action_key", default="action",
                 help="'action' (executed) or 'action_policy' (clean teacher)")
  p.add_argument("--wandb", action="store_true")
  args = p.parse_args()

  cfg = yaml.safe_load(open(args.config))
  if args.num_steps is not None:
    cfg["train"]["num_steps"] = args.num_steps
  run = args.run_name or f"act_{args.dataset}_n{args.n_demos}_s{args.seed}"
  ckpt_dir = common.DATA_DIR / "act" / run
  log_dir = common.ROOT / "results" / "train" / run
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  split = load_split(common.ROOT / "configs" / f"split_{args.dataset}.json")
  train_idx = split["subsets"][str(args.n_demos)]
  obs, act, lengths = load_episodes(split["file"], train_idx, args.action_key)
  v_obs, v_act, v_len = load_episodes(split["file"], split["val_idx"], args.action_key)

  valid = np.arange(obs.shape[1])[None] < lengths[:, None]
  obs_norm = Normalizer.fit(obs[valid])
  act_norm = Normalizer.fit(act[valid])

  mcfg, tcfg = cfg["model"], cfg["train"]
  k = mcfg["chunk_size"]
  train_s = ChunkSampler(obs, act, lengths, k, obs_norm, act_norm, device)
  val_s = ChunkSampler(v_obs, v_act, v_len, k, obs_norm, act_norm, device)

  model = ACTPolicy(obs.shape[-1], act.shape[-1], **mcfg).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                          weight_decay=tcfg["weight_decay"])
  n_params = sum(p_.numel() for p_ in model.parameters())
  print(f"{run}: {len(train_idx)} demos, {len(train_s)} samples, "
        f"{n_params / 1e6:.2f}M params, device {device}", flush=True)

  if args.wandb:
    import wandb
    wandb.init(project="csci645-hw2", name=run, config={**cfg, **vars(args)})

  gen = torch.Generator(device=device).manual_seed(args.seed)
  log_f = open(log_dir / "loss.csv", "w", newline="")
  writer = csv.writer(log_f)
  writer.writerow(["step", "split", "loss", "l1", "kl", "l1_prior", "lr"])
  t0 = time.time()
  for step in range(1, tcfg["num_steps"] + 1):
    o, a, pad = train_s.sample(tcfg["batch_size"], gen)
    pred, dist = model(o, a, pad)
    loss, parts = act_loss(pred, a, pad, dist, tcfg["kl_weight"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if tcfg.get("grad_clip"):
      torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
    opt.step()

    if step % tcfg["log_every"] == 0:
      writer.writerow([step, "train", loss.item(), parts["l1"], parts["kl"], "",
                       tcfg["lr"]])
    if step % tcfg["val_every"] == 0 or step == tcfg["num_steps"]:
      v = evaluate_loss(model, val_s, tcfg["kl_weight"])
      writer.writerow([step, "val", v["loss"], v["l1"], v["kl"], v["l1_prior"],
                       tcfg["lr"]])
      log_f.flush()
      print(f"[{time.time() - t0:6.0f}s] step {step:6d} train {loss.item():.4f} "
            f"val {v['loss']:.4f} val_l1_prior {v['l1_prior']:.4f}", flush=True)
      if args.wandb:
        wandb.log({"train/loss": loss.item(), **{f"val/{k_}": x for k_, x in v.items()}},
                  step=step)
  log_f.close()

  torch.save({
      "model": model.state_dict(),
      "model_cfg": mcfg,
      "obs_dim": obs.shape[-1],
      "action_dim": act.shape[-1],
      "obs_norm": obs_norm.state_dict(),
      "act_norm": act_norm.state_dict(),
      "train_idx": train_idx,
      "args": vars(args),
      "cfg": cfg,
  }, ckpt_dir / "model.pt")
  common.save_json(log_dir / "run.json", {
      "run": run, "args": vars(args), "cfg": cfg, "n_train_samples": len(train_s),
      "n_params": n_params, "train_time_s": time.time() - t0,
      "meta": common.run_metadata()})
  print(f"Saved {ckpt_dir / 'model.pt'}")


if __name__ == "__main__":
  main()
