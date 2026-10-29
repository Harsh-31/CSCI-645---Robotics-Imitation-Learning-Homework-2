"""Action Chunking with Transformers (ACT), state-only variant.

Adapted from the reference implementation by Tony Z. Zhao et al.
(https://github.com/tonyzhaozh/act, detr/models/detr_vae.py; paper
arXiv:2304.13705). Changes for this homework:
  * No image backbone / camera tokens: the observation is the 66-d
    PandaPickCube state vector, projected to a single token.
  * Uses torch.nn.Transformer{Encoder,Decoder} layers (batch_first) instead of
    the DETR transformer copy; fixed sinusoidal position embeddings for the
    CVAE encoder input and learned embeddings for the decoder inputs.
  * The CVAE encoder conditions on the observation (ACT uses joint positions).
Training objective (as in ACT): L1(action chunk) + beta * KL(q(z|o,a) || N(0,I)).
At test time z = 0 (the prior mean).
"""

import math

import torch
import torch.nn as nn


def sinusoid_table(n_pos: int, dim: int) -> torch.Tensor:
  pos = torch.arange(n_pos).unsqueeze(1)
  div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
  table = torch.zeros(n_pos, dim)
  table[:, 0::2] = torch.sin(pos * div)
  table[:, 1::2] = torch.cos(pos * div)
  return table


class ACTPolicy(nn.Module):

  def __init__(self, obs_dim: int, action_dim: int, chunk_size: int = 20,
               d_model: int = 256, n_heads: int = 8, n_enc_layers: int = 4,
               n_dec_layers: int = 4, n_cvae_layers: int = 4,
               dim_feedforward: int = 1024, dropout: float = 0.1,
               latent_dim: int = 32):
    super().__init__()
    self.obs_dim, self.action_dim = obs_dim, action_dim
    self.chunk_size, self.latent_dim = chunk_size, latent_dim

    def enc_layer():
      return nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward,
                                        dropout, batch_first=True)

    # --- CVAE encoder q(z | obs, action chunk) (training only) ---
    self.cvae_encoder = nn.TransformerEncoder(enc_layer(), n_cvae_layers)
    self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model))
    self.cvae_obs_proj = nn.Linear(obs_dim, d_model)
    self.cvae_action_proj = nn.Linear(action_dim, d_model)
    self.latent_proj = nn.Linear(d_model, 2 * latent_dim)  # -> mu, logvar
    self.register_buffer("cvae_pos", sinusoid_table(2 + chunk_size, d_model)[None])

    # --- Policy: transformer encoder over [z token, obs token] ---
    self.latent_out_proj = nn.Linear(latent_dim, d_model)
    self.obs_proj = nn.Linear(obs_dim, d_model)
    self.enc_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
    self.encoder = nn.TransformerEncoder(enc_layer(), n_enc_layers)

    # --- Decoder: chunk_size learned queries attend to the encoder memory ---
    dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward,
                                           dropout, batch_first=True)
    self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers)
    self.query_embed = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
    self.action_head = nn.Linear(d_model, action_dim)

  def encode(self, obs, actions, is_pad):
    b = obs.shape[0]
    tokens = torch.cat([
        self.cls_embed.expand(b, -1, -1),
        self.cvae_obs_proj(obs)[:, None],
        self.cvae_action_proj(actions),
    ], dim=1) + self.cvae_pos
    pad = torch.cat([torch.zeros(b, 2, dtype=torch.bool, device=obs.device),
                     is_pad], dim=1)
    h = self.cvae_encoder(tokens, src_key_padding_mask=pad)
    mu, logvar = self.latent_proj(h[:, 0]).chunk(2, dim=-1)
    return mu, logvar

  def forward(self, obs, actions=None, is_pad=None):
    """obs [B, obs_dim]; actions [B, k, action_dim] (normalised); is_pad [B, k].
    Returns predicted chunk [B, k, action_dim] and (mu, logvar) or None."""
    b = obs.shape[0]
    if actions is not None:
      mu, logvar = self.encode(obs, actions, is_pad)
      z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
      dist = (mu, logvar)
    else:
      z = torch.zeros(b, self.latent_dim, device=obs.device)
      dist = None
    memory = torch.stack([self.latent_out_proj(z), self.obs_proj(obs)], dim=1)
    memory = self.encoder(memory + self.enc_pos)
    h = self.decoder(self.query_embed.expand(b, -1, -1), memory)
    return self.action_head(h), dist


def act_loss(pred, target, is_pad, dist, kl_weight: float):
  valid = (~is_pad).unsqueeze(-1).float()
  l1 = (torch.abs(pred - target) * valid).sum() / (valid.sum() * pred.shape[-1])
  mu, logvar = dist
  kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
  return l1 + kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}
