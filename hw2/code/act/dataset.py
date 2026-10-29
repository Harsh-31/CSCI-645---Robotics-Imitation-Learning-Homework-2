"""Demonstration dataset -> (obs_t, action chunk a_{t:t+k}, pad mask) samples.

The whole dataset fits in GPU memory (100 episodes x 150 steps), so samples are
drawn by indexing pre-built tensors instead of using a DataLoader.
"""

import json

import numpy as np
import torch


class Normalizer:
  """Per-dimension z-scoring with stats from the *training subset only*.
  std is clipped below at 1e-2 (as in ACT) so constant dims stay finite."""

  def __init__(self, mean, std):
    self.mean = np.asarray(mean, np.float32)
    self.std = np.asarray(std, np.float32)

  @classmethod
  def fit(cls, x, min_std=1e-2):
    return cls(x.mean(0), np.clip(x.std(0), min_std, None))

  def state_dict(self):
    return {"mean": self.mean.tolist(), "std": self.std.tolist()}

  @classmethod
  def from_state_dict(cls, d):
    return cls(d["mean"], d["std"])

  def torch(self, device):
    return (torch.as_tensor(self.mean, device=device),
            torch.as_tensor(self.std, device=device))


def load_split(split_path):
  with open(split_path) as f:
    return json.load(f)


def load_episodes(demo_path, idx, action_key="action"):
  """Returns obs [N,T,D], actions [N,T,A], lengths [N] for episodes `idx`."""
  d = np.load(demo_path)
  idx = np.asarray(idx)
  return (d["obs"][idx].astype(np.float32), d[action_key][idx].astype(np.float32),
          d["alive"][idx].sum(1).astype(np.int64))


class ChunkSampler:
  """Uniformly samples (episode, t) with t < length; chunks are padded past
  the end of the episode and the padding is masked out of the loss."""

  def __init__(self, obs, actions, lengths, chunk_size, obs_norm, act_norm,
               device):
    n, T, _ = obs.shape
    k = chunk_size
    obs = (obs - obs_norm.mean) / obs_norm.std
    actions = (actions - act_norm.mean) / act_norm.std
    a_pad = np.zeros((n, T + k, actions.shape[-1]), np.float32)
    a_pad[:, :T] = actions
    valid = np.arange(T + k)[None] < lengths[:, None]
    self.obs = torch.as_tensor(obs, device=device)
    self.actions = torch.as_tensor(a_pad, device=device)
    self.is_pad = torch.as_tensor(~valid, device=device)
    pairs = [(e, t) for e in range(n) for t in range(int(lengths[e]))]
    self.pairs = torch.as_tensor(pairs, device=device)
    self.offsets = torch.arange(k, device=device)
    self.device = device

  def __len__(self):
    return len(self.pairs)

  def get(self, pair_idx):
    e, t = self.pairs[pair_idx, 0], self.pairs[pair_idx, 1]
    tt = t[:, None] + self.offsets[None]
    return (self.obs[e, t], self.actions[e[:, None], tt],
            self.is_pad[e[:, None], tt])

  def sample(self, batch_size, generator=None):
    i = torch.randint(len(self.pairs), (batch_size,), device=self.device,
                      generator=generator)
    return self.get(i)
