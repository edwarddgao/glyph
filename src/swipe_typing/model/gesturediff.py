"""Diffusion gesture generator: DDPM over the whole trajectory at once.

Why this shape of model, given what the earlier arms measured:

* Joint generation, not autoregressive. v1 and v2 both failed on
  accumulation — mean-seeking steps compound into a path too short to
  reach the last letter, sampled steps compound into a random walk. A
  diffusion model denoises all 64 points simultaneously, so there is no
  integration to drift.
* No mode-averaging. The VAE arms (v3/v4) reach the letters but their
  reconstruction loss still pulls each sample toward the conditional mean,
  which is why their dwell fraction sits at half of real. Diffusion models
  the distribution rather than its mean, so dwell — a genuinely multi-modal
  quantity, since users pause at different letters on different attempts —
  is expressible.
* Conditioning is the prototype, supplied as extra channels at every step
  (and every denoising step), which is what keeps the geometry anchored
  without the analytic generator's rigidity.

Duration is not diffused: it is a solved subproblem (the CLC power law
fitted in minjerk.py predicts it to ~1s RMSE), and spending model capacity
on it would only dilute the part that matters, which is where the samples
bunch in time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .. import features
from ..layout import KeyboardLayout
from ..schema import Swipe
from .gesturegen import prototype


@dataclass
class DiffConfig:
    n_points: int = features.N_POINTS
    d_model: int = 128
    n_blocks: int = 8
    timesteps: int = 1000
    dropout: float = 0.1


def cosine_alphas(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule; cumulative alphas."""
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = f / f[0]
    return (a[1:] / a[:-1]).clamp(1e-5, 0.999).cumprod(0)


class FiLMBlock(nn.Module):
    def __init__(self, d: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, d)
        self.film = nn.Linear(d, 2 * d)
        self.out = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.norm(self.conv(x))
        scale, shift = self.film(t_emb)[..., None].chunk(2, dim=1)
        h = torch.nn.functional.gelu(h * (1 + scale) + shift)
        return x + self.out(self.drop(h))


class GestureDiffusion(nn.Module):
    """Predicts the noise added to a trajectory, given the word's prototype."""

    def __init__(self, cfg: DiffConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.inp = nn.Conv1d(4, d, 1)          # noisy xy + prototype xy
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([
            FiLMBlock(d, 2 ** (i % 4), cfg.dropout) for i in range(cfg.n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, d), nn.GELU(),
                                 nn.Conv1d(d, 2, 1))
        self.register_buffer("acp", cosine_alphas(cfg.timesteps))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def t_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.cfg.d_model // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device)
                          / half)
        ang = t.float()[:, None] * freqs[None]
        return self.t_mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

    def forward(self, x_t, proto, t):
        """x_t, proto: (B, n, 2); t: (B,) int. Returns predicted noise."""
        h = self.inp(torch.cat([x_t, proto], dim=-1).transpose(1, 2))
        emb = self.t_embed(t)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h).transpose(1, 2)

    def loss(self, x0, proto):
        B = x0.size(0)
        t = torch.randint(0, self.cfg.timesteps, (B,), device=x0.device)
        a = self.acp[t][:, None, None]
        noise = torch.randn_like(x0)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        return (self(x_t, proto, t) - noise).pow(2).mean()

    @torch.no_grad()
    def sample(self, proto, steps: int = 50, eta: float = 0.0,
               generator: torch.Generator | None = None):
        """DDIM sampling. ``eta``=0 is deterministic given the initial noise."""
        B, n, _ = proto.shape
        dev = proto.device
        x = torch.randn(B, n, 2, device=dev, generator=generator)
        ts = torch.linspace(self.cfg.timesteps - 1, 0, steps).long().to(dev)
        for i, t in enumerate(ts):
            a = self.acp[t]
            eps = self(x, proto, t.repeat(B))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-3, 3)
            if i + 1 < len(ts):
                a_prev = self.acp[ts[i + 1]]
                sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).sqrt()
                x = (a_prev.sqrt() * x0
                     + (1 - a_prev - sigma**2).clamp_min(0).sqrt() * eps)
                if eta > 0:
                    x = x + sigma * torch.randn(x.shape, device=dev,
                                                generator=generator)
            else:
                x = x0
        return x


@torch.no_grad()
def sample_swipes(model: GestureDiffusion, words: list[str],
                  layout: KeyboardLayout, device: torch.device, aspect: float,
                  duration_model, steps: int = 50, eta: float = 0.0,
                  batch_size: int = 512, seed: int = 0) -> list[Swipe]:
    """Sample trajectories; durations come from the fitted CLC law."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = model.cfg.n_points
    out: list[Swipe] = []
    for s in range(0, len(words), batch_size):
        chunk = words[s:s + batch_size]
        proto = torch.from_numpy(
            np.stack([prototype(w, layout, n) for w in chunk])).to(device)
        pts = model.sample(proto, steps=steps, eta=eta,
                           generator=g).cpu().numpy()
        for i, w in enumerate(chunk):
            seg = np.linalg.norm(
                np.diff(np.stack([layout.center(c) for c in w]) *
                        [duration_model.aspect, 1.0], axis=0), axis=1) \
                if len(w) > 1 else np.zeros(1)
            dur = max(float(duration_model.m * (seg ** duration_model.n).sum()
                            * np.exp(rng.normal(0, duration_model.log_sigma))),
                      80.0)
            t = np.linspace(0.0, dur, n)
            out.append(Swipe(word=w, x=pts[i, :, 0], y=pts[i, :, 1],
                             t=np.round(t).astype(np.int32), aspect=aspect,
                             session="diffusion", source="diffusion"))
    return out
