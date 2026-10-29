"""Batched ACT execution in the simulator.

Two execution modes (ACT paper, Sec. 3.3):
  temporal_ensemble: query every step; the action for step t is a weighted
    average of all chunks that cover t, w_i = exp(-m * i) with i = 0 the
    *oldest* prediction (as in the reference implementation).
  chunk: query every `exec_horizon` steps and execute that chunk open-loop.
Observations are normalised with the training-set stats; predicted actions are
un-normalised and clipped to [-1, 1] before env.step.
"""

import numpy as np
import torch

from act.dataset import Normalizer
from act.model import ACTPolicy


def load_act(path, device):
  ckpt = torch.load(path, map_location=device, weights_only=False)
  model = ACTPolicy(ckpt["obs_dim"], ckpt["action_dim"], **ckpt["model_cfg"])
  model.load_state_dict(ckpt["model"])
  model.to(device).eval()
  return model, ckpt


class ACTRunner:

  def __init__(self, model, ckpt, batch_size, episode_length, device,
               execution="temporal_ensemble", ensemble_m=0.01, exec_horizon=None):
    self.model, self.device = model, device
    self.k = model.chunk_size
    self.obs_mean, self.obs_std = Normalizer.from_state_dict(ckpt["obs_norm"]).torch(device)
    self.act_mean, self.act_std = Normalizer.from_state_dict(ckpt["act_norm"]).torch(device)
    self.execution, self.m = execution, ensemble_m
    self.exec_horizon = exec_horizon or self.k
    T, A = episode_length, model.action_dim
    if execution == "temporal_ensemble":
      # buffer[b, t_query, t_exec] = action predicted at t_query for t_exec
      self.buffer = torch.zeros(batch_size, T, T + self.k, A, device=device)
    self.chunk = None

  @torch.no_grad()
  def predict_chunk(self, obs):
    o = (torch.tensor(obs, device=self.device) - self.obs_mean) / self.obs_std
    pred, _ = self.model(o)
    return pred * self.act_std + self.act_mean  # [B, k, A]

  def __call__(self, obs, t):
    if self.execution == "temporal_ensemble":
      self.buffer[:, t, t:t + self.k] = self.predict_chunk(obs)
      first = max(0, t - self.k + 1)
      cands = self.buffer[:, first:t + 1, t]  # oldest first
      w = torch.exp(-self.m * torch.arange(cands.shape[1], device=self.device))
      action = (cands * (w / w.sum())[None, :, None]).sum(1)
    else:
      if t % self.exec_horizon == 0:
        self.chunk = self.predict_chunk(obs)
      action = self.chunk[:, t % self.exec_horizon]
    return action.clamp(-1, 1).cpu().numpy()
