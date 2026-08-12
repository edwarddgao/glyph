"""WordGesture-GAN (Chu et al., CHI 2023), implemented to the paper's spec.

This exists as a *control*, not as a favoured design. The other arms in this
repo are my own architectures informed by the paper's conditioning idea, so
"learned generators underperform the analytic one here" cannot be separated
from "I built mine badly" without the published method itself in the
comparison.

Followed as specified in Sections 3-4:

  representation   n = 128 points per gesture as (x, y, t); x/y normalized
                   to [-1, 1] across the keyboard, t in seconds since the
                   previous point (their choice, so the temporal scale
                   matches the spatial one)
  prototype        straight lines between letter centroids, n - k points
                   distributed uniformly between key centres
  encoder          MLP 384-192-96-48 -> (mu, logvar) at 32 dims, Leaky ReLU
  generator        4x BiLSTM (35->32->32->32->32), Linear 32->3, Tanh;
                   the latent is repeated along the sequence and
                   concatenated with the prototype (BicycleGAN-style)
  discriminator    MLP 384-192-96-48-24-1, Leaky ReLU, spectral norm on
                   every layer, unconditional (the paper found this beat a
                   conditional critic)
  objective        WGAN, 5 critic steps per generator step, plus
                   feature matching (1), reconstruction (5), latent
                   recovery (0.5), KLD (0.05); two BicycleGAN cycles, with
                   the encoder frozen inside the latent-recovery term
  optimizer        Adam, lr 2e-4, batch 512

Deliberate deviations, both forced by this repo rather than by taste: the
corpus is FUTO's (917k gestures) rather than their 38k, and gestures enter
through the same canonical pipeline as every other generator here so the
quality oracle compares like with like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

from ..layout import KeyboardLayout
from ..schema import Swipe

N_POINTS = 128
Z_DIM = 32


@dataclass
class WGGConfig:
    n_points: int = N_POINTS
    z_dim: int = Z_DIM
    lstm_hidden: int = 16          # bidirectional -> 32 out, per Figure 3
    lstm_layers: int = 4
    lambda_feat: float = 1.0
    lambda_rec: float = 5.0
    lambda_lat: float = 0.5
    lambda_kld: float = 0.05
    critic_steps: int = 5


def word_prototype(word: str, layout: KeyboardLayout,
                   n: int = N_POINTS) -> np.ndarray:
    """(n, 3) prototype: normalized xy polyline through key centres + dt.

    Their construction: k key centres as anchors, the remaining n - k points
    distributed uniformly between consecutive pairs. The third channel is a
    constant nominal dt, so the generator sees the same channel layout it
    must produce.
    """
    centres = np.stack([layout.center(c) for c in word]).astype(np.float64)
    k = len(centres)
    if k == 1:
        xy = np.repeat(centres, n, axis=0)
    else:
        per = (n - k) // max(k - 1, 1)
        pieces = []
        for i in range(k - 1):
            m = per + 2 if i < k - 2 else n - len(np.concatenate(pieces)) \
                if pieces else per + 2
            seg = np.linspace(centres[i], centres[i + 1], max(m, 2))
            pieces.append(seg[:-1] if i < k - 2 else seg)
        xy = np.concatenate(pieces)[:n]
        if len(xy) < n:
            xy = np.concatenate([xy, np.repeat(xy[-1:], n - len(xy), axis=0)])
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, :2] = xy * 2.0 - 1.0            # canonical [0,1] -> [-1,1]
    out[:, 2] = 1.0 / n
    return out


def encode_gesture(points: np.ndarray, t_ms: np.ndarray,
                   n: int = N_POINTS) -> np.ndarray:
    """(n, 3) real gesture in the paper's representation.

    Uniform subsampling by index with the endpoints kept (their rule), which
    leaves the dwell information in the dt channel rather than in the
    spacing of the coordinates.
    """
    m = len(points)
    idx = (np.linspace(0, m - 1, n).round().astype(int) if m >= n
           else np.linspace(0, m - 1, n))
    if m >= n:
        xy = points[idx]
        t = t_ms[idx].astype(np.float64)
    else:
        xy = np.stack([np.interp(idx, np.arange(m), points[:, c])
                       for c in range(2)], axis=1)
        t = np.interp(idx, np.arange(m), t_ms.astype(np.float64))
    dt = np.diff(t, prepend=t[0]) / 1000.0
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, :2] = xy * 2.0 - 1.0
    out[:, 2] = dt
    return out


class VariationalEncoder(nn.Module):
    def __init__(self, cfg: WGGConfig):
        super().__init__()
        d = cfg.n_points * 3
        self.net = nn.Sequential(
            nn.Linear(d, 192), nn.LeakyReLU(0.2),
            nn.Linear(192, 96), nn.LeakyReLU(0.2),
            nn.Linear(96, 48), nn.LeakyReLU(0.2))
        self.mu = nn.Linear(48, cfg.z_dim)
        self.logvar = nn.Linear(48, cfg.z_dim)

    def forward(self, x):
        h = self.net(x.flatten(1))
        return self.mu(h), self.logvar(h)


class Generator(nn.Module):
    def __init__(self, cfg: WGGConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.lstm_hidden
        self.lstm = nn.LSTM(3 + cfg.z_dim, h, cfg.lstm_layers,
                            batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, 3)

    def forward(self, proto, z):
        n = proto.size(1)
        inp = torch.cat([proto, z[:, None, :].expand(-1, n, -1)], dim=-1)
        h, _ = self.lstm(inp)
        return torch.tanh(self.out(h))


class Discriminator(nn.Module):
    def __init__(self, cfg: WGGConfig):
        super().__init__()
        d = cfg.n_points * 3
        dims = [(d, 192), (192, 96), (96, 48), (48, 24)]
        self.hidden = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Linear(a, b)), nn.LeakyReLU(0.2))
            for a, b in dims])
        self.out = spectral_norm(nn.Linear(24, 1))

    def forward(self, x, features: bool = False):
        h = x.flatten(1)
        feats = []
        for layer in self.hidden:
            h = layer(h)
            feats.append(h)
        score = self.out(h).squeeze(-1)
        return (score, feats) if features else score


class WordGestureGAN(nn.Module):
    def __init__(self, cfg: WGGConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = VariationalEncoder(cfg)
        self.gen = Generator(cfg)
        self.disc = Discriminator(cfg)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def critic_loss(self, real, proto):
        """Eq 1: E[D(G(z,y))] - E[D(x)], z from the prior."""
        with torch.no_grad():
            z = torch.randn(real.size(0), self.cfg.z_dim, device=real.device)
            fake = self.gen(proto, z)
        return self.disc(fake).mean() - self.disc(real).mean()

    def generator_loss(self, real, proto):
        """Eq 2 over both BicycleGAN cycles."""
        cfg = self.cfg
        B = real.size(0)
        dev = real.device

        # cycle 1: z ~ P(z) -> X' -> z'  (latent recovery; encoder frozen)
        z = torch.randn(B, cfg.z_dim, device=dev)
        fake = self.gen(proto, z)
        score_fake, feat_fake = self.disc(fake, features=True)
        with torch.no_grad():
            _, feat_real = self.disc(real, features=True)
        l_adv = -score_fake.mean()
        l_feat = sum((f.mean(0) - r.mean(0)).abs().mean()
                     for f, r in zip(feat_fake, feat_real)) / len(feat_fake)
        for p in self.enc.parameters():
            p.requires_grad_(False)
        z_rec, _ = self.enc(fake)
        for p in self.enc.parameters():
            p.requires_grad_(True)
        l_lat = (z_rec - z).abs().mean()

        # cycle 2: X -> z' -> X'  (reconstruction + KLD)
        mu, logvar = self.enc(real)
        recon = self.gen(proto, self.reparameterize(mu, logvar))
        l_rec = (recon - real).abs().mean()
        l_kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()

        loss = (l_adv + cfg.lambda_feat * l_feat + cfg.lambda_rec * l_rec
                + cfg.lambda_lat * l_lat + cfg.lambda_kld * l_kld)
        return loss, {"adv": float(l_adv.detach()), "feat": float(l_feat.detach()),
                      "rec": float(l_rec.detach()), "lat": float(l_lat.detach()),
                      "kld": float(l_kld.detach())}


@torch.no_grad()
def sample_swipes(model: WordGestureGAN, words: list[str],
                  layout: KeyboardLayout, device: torch.device, aspect: float,
                  batch_size: int = 512, seed: int = 0,
                  min_ms: float = 120.0) -> list[Swipe]:
    """Sample gestures and decode the (x, y, dt) output back to canonical."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = model.cfg.n_points
    out: list[Swipe] = []
    for s in range(0, len(words), batch_size):
        chunk = words[s:s + batch_size]
        proto = torch.from_numpy(
            np.stack([word_prototype(w, layout, n) for w in chunk])).to(device)
        z = torch.randn(len(chunk), model.cfg.z_dim, generator=g).to(device)
        gen = model.gen(proto, z).cpu().numpy().astype(np.float64)
        for i, w in enumerate(chunk):
            xy = (gen[i, :, :2] + 1.0) / 2.0
            dt = np.clip(gen[i, :, 2], 0.0, None)
            t = np.cumsum(dt) * 1000.0
            t = t - t[0]
            if t[-1] < min_ms:      # degenerate duration -> uniform fallback
                t = np.linspace(0.0, min_ms, n)
            out.append(Swipe(word=w, x=xy[:, 0].astype(np.float32),
                             y=xy[:, 1].astype(np.float32),
                             t=np.round(t).astype(np.int32), aspect=aspect,
                             session="wgg", source="wgg", split="train"))
    return out
