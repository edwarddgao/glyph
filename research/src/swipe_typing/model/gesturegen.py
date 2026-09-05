"""Learned gesture generator: a prototype-conditioned CVAE with an
autoregressive trajectory decoder.

What the min-jerk ladder (#56-#62) taught about this design space:

* Decoder training utility lives in per-letter dwell *texture* — where the
  uniform-time samples bunch — not in aggregate kinematic realism. The
  spline profile matched speed histograms and was nearly untrainable; crude
  dead-stop segments trained; randomized texture trained best (85.65 beam
  synthetic-only, the number this model exists to beat).
* Mode-averaged smoothness is the failure mode. A regression decoder under
  L1 predicts the *mean* trajectory given the word — a smooth glide, i.e.
  the spline catastrophe re-learned from data. Hence the autoregressive
  decoder: each step conditions on the previous point, so the model commits
  to one texture per rollout instead of blending them, and L1 on per-step
  displacement makes exact zero steps (dwell) cheap to express.
* WordGesture-GAN's conditioning transfers: a straight-line prototype
  through the key centers, resampled uniformly by arclength, plus a
  Gaussian latent for gesture-level variation. Points are uniform in time
  (the featurizer's native sampling), so the temporal channel is a single
  scalar duration and velocity is encoded as spatial bunching.

The latent must not collapse: a collapsed posterior gives one gesture per
word and re-creates WordGesture-GAN's recall failure at the corpus level.
Free-bits plus previous-point dropout (the decoder sometimes loses its
autoregressive crutch and must read z) keep it alive.
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


@dataclass
class GenConfig:
    n_points: int = features.N_POINTS
    d_model: int = 128
    z_dim: int = 32
    enc_layers: int = 2
    dropout: float = 0.1
    #: P(replace the teacher-forced previous point with the prototype point)
    #: — weakens the AR crutch so z stays informative, and doubles as
    #: exposure-bias training for sampling time.
    prev_dropout: float = 0.15
    #: Mixture components on the next-step displacement (Graves-style
    #: handwriting synthesis). 0 = plain L1 regression, which predicts the
    #: per-step conditional *mean* and therefore rolls out smooth — the
    #: spline failure re-learned from data. A mixture keeps "pause" and
    #: "move fast" as separate modes, which is what dwell texture is.
    mdn_components: int = 0
    #: half-cosine terms in v4's offset field; fewer = smoother. 6 gives a
    #: shortest wiggle of ~10 samples, well above per-knot noise and well
    #: below a letter-to-letter transit.
    n_basis: int = 6


def prototype(word: str, layout: KeyboardLayout,
              n: int = features.N_POINTS) -> np.ndarray:
    """(n, 2) polyline through the word's key centers, uniform in arclength."""
    pts = np.stack([layout.center(c) for c in word]).astype(np.float64)
    if len(pts) == 1:
        return np.repeat(pts.astype(np.float32), n, axis=0)
    return features.resample(pts, None, n=n, mode="arclength")


class GestureVAE(nn.Module):
    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        d, z = cfg.d_model, cfg.z_dim
        # gesture xy + prototype xy + log-duration
        self.encoder = nn.GRU(5, d // 2, cfg.enc_layers, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout)
        self.to_mu = nn.Linear(d, z)
        self.to_logvar = nn.Linear(d, z)
        # prev xy + prototype xy + log-duration + z
        self.decoder = nn.GRU(5 + z, d, 1, batch_first=True)
        self.K = cfg.mdn_components
        # per component: weight, mu(2), log-sigma(2), tanh-correlation
        self.to_delta = nn.Linear(d, 6 * self.K if self.K else 2)
        # z + prototype arclength -> log duration
        self.dur_head = nn.Sequential(
            nn.Linear(z + 1, d), nn.GELU(), nn.Linear(d, 1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _arclen(proto: torch.Tensor) -> torch.Tensor:
        return proto.diff(dim=1).norm(dim=-1).sum(dim=1, keepdim=True)

    def mdn_params(self, h: torch.Tensor):
        """(log_w, mu, sigma, rho) from decoder hidden states."""
        B, T, _ = h.shape
        p = self.to_delta(h).view(B, T, self.K, 6)
        log_w = torch.log_softmax(p[..., 0], dim=-1)
        mu = p[..., 1:3]
        sigma = p[..., 3:5].clamp(-7.0, 4.0).exp()
        rho = torch.tanh(p[..., 5]) * 0.95
        return log_w, mu, sigma, rho

    def mdn_nll(self, h: torch.Tensor, target: torch.Tensor):
        """Negative log likelihood of the true displacement under the mixture."""
        log_w, mu, sigma, rho = self.mdn_params(h)
        d = (target[..., None, :] - mu) / sigma
        one_m = 1.0 - rho**2
        z = d[..., 0] ** 2 + d[..., 1] ** 2 - 2 * rho * d[..., 0] * d[..., 1]
        log_n = (-z / (2 * one_m)
                 - torch.log(2 * np.pi * sigma[..., 0] * sigma[..., 1])
                 - 0.5 * torch.log(one_m))
        return -torch.logsumexp(log_w + log_n, dim=-1).mean()

    def mdn_sample(self, h: torch.Tensor, temperature: float = 1.0):
        """Sample one displacement per step. h is (B, 1, d) during rollout."""
        log_w, mu, sigma, rho = self.mdn_params(h)
        if temperature != 1.0:
            log_w = log_w / temperature
            log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
            sigma = sigma * temperature
        B, T, K = log_w.shape
        k = torch.distributions.Categorical(logits=log_w).sample()
        idx = k[..., None, None].expand(B, T, 1, 2)
        mu_k = mu.gather(2, idx).squeeze(2)
        sig_k = sigma.gather(2, idx).squeeze(2)
        rho_k = rho.gather(2, k[..., None]).squeeze(2)
        e = torch.randn_like(mu_k)
        dx = sig_k[..., 0] * e[..., 0]
        dy = sig_k[..., 1] * (rho_k * e[..., 0]
                              + (1 - rho_k**2).sqrt() * e[..., 1])
        return mu_k + torch.stack([dx, dy], dim=-1)

    def encode(self, gesture, proto, logdur):
        inp = torch.cat(
            [gesture, proto,
             logdur[:, None, None].expand(-1, proto.size(1), 1)], dim=-1)
        h, _ = self.encoder(inp)
        pooled = h.mean(dim=1)
        return self.to_mu(pooled), self.to_logvar(pooled)

    def predict_logdur(self, proto, z):
        return self.dur_head(
            torch.cat([z, self._arclen(proto)], dim=-1)).squeeze(-1)

    def decode_tf(self, gesture, proto, logdur, z,
                  prev_dropout: float | None = None):
        """Teacher-forced next-point prediction. Returns (pred, prev)."""
        p = self.cfg.prev_dropout if prev_dropout is None else prev_dropout
        prev = torch.cat([proto[:, :1], gesture[:, :-1]], dim=1)
        if self.training and p > 0:
            keep = (torch.rand(prev.shape[:2], device=prev.device) > p)
            prev = torch.where(keep[..., None], prev, proto)
        n = proto.size(1)
        inp = torch.cat([
            prev, proto,
            logdur[:, None, None].expand(-1, n, 1),
            z[:, None, :].expand(-1, n, -1),
        ], dim=-1)
        h, _ = self.decoder(inp)
        if self.K:
            return h, prev
        return prev + self.to_delta(h), prev

    def forward(self, gesture, proto, logdur):
        mu, logvar = self.encode(gesture, proto, logdur)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        pred, prev = self.decode_tf(gesture, proto, logdur, z)
        logdur_hat = self.predict_logdur(proto, z)
        return pred, prev, logdur_hat, mu, logvar

    @torch.no_grad()
    def rollout(self, proto: torch.Tensor, z: torch.Tensor,
                temperature: float = 1.0):
        """Autoregressive sampling. Returns (points, logdur)."""
        logdur = self.predict_logdur(proto, z)
        n = proto.size(1)
        prev = proto[:, :1]
        hidden = None
        out = []
        zc = z[:, None, :]
        ld = logdur[:, None, None]
        for t in range(n):
            inp = torch.cat([prev, proto[:, t:t + 1], ld, zc], dim=-1)
            h, hidden = self.decoder(inp, hidden)
            step = (self.mdn_sample(h, temperature) if self.K
                    else self.to_delta(h))
            prev = prev + step
            out.append(prev)
        return torch.cat(out, dim=1), logdur


class WarpGestureVAE(nn.Module):
    """v3: prototype-anchored offsets + a monotone time warp.

    Both v1 and v2 free-run the trajectory step by step, and both fail on
    geometry rather than texture: v1's mean-seeking steps compound into a
    path *shorter* than the polyline through the keys (0.86x), so it cuts
    corners and strands the gesture before the last letter; v2's sampled
    steps compound into a random walk (5-15x, wandering off the keyboard).
    Accumulation is the common cause, so v3 removes it — nothing is
    integrated.

    A gesture is instead parameterized as *where along a curve* each
    uniform-time sample lands:

      curve  = prototype + predicted offset field   (geometry: pinned to the
               keys by construction, free to overshoot and round corners)
      warp   = cumsum(softmax(increments)), from 0 to 1   (texture: flat
               stretches are dwell at a letter, steep stretches are fast
               transit)
      points = curve sampled at the warp positions

    The warp sums to exactly 1, so the gesture always starts on the first
    key and ends on the last — the two failures the eye caught are now
    structurally impossible rather than penalized. Dwell is the natural
    parameterization of the warp, which is what the min-jerk ladder said
    the decoder actually feeds on. Variation comes from z, not from
    per-step noise, so there is no random walk to temper.
    """

    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        d, z = cfg.d_model, cfg.z_dim
        self.K = 0  # no mixture head; kept for interface parity
        self.encoder = nn.GRU(5, d // 2, cfg.enc_layers, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout)
        self.to_mu = nn.Linear(d, z)
        self.to_logvar = nn.Linear(d, z)
        self.decoder = nn.GRU(2 + z + 1, d // 2, 2, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout)
        self.to_offset = nn.Linear(d, 2)
        self.to_warp = nn.Linear(d, 1)
        self.dur_head = nn.Sequential(
            nn.Linear(z + 1, d), nn.GELU(), nn.Linear(d, 1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _arclen(proto):
        return proto.diff(dim=1).norm(dim=-1).sum(dim=1, keepdim=True)

    def encode(self, gesture, proto, logdur):
        inp = torch.cat(
            [gesture, proto,
             logdur[:, None, None].expand(-1, proto.size(1), 1)], dim=-1)
        h, _ = self.encoder(inp)
        return self.to_mu(h.mean(1)), self.to_logvar(h.mean(1))

    def predict_logdur(self, proto, z):
        return self.dur_head(
            torch.cat([z, self._arclen(proto)], dim=-1)).squeeze(-1)

    @staticmethod
    def _sample_curve(curve, w):
        """Linear interpolation of ``curve`` (B,n,2) at warp positions (B,n)."""
        n = curve.size(1)
        idx = w * (n - 1)
        i0 = idx.floor().clamp(0, n - 2).long()
        frac = (idx - i0.float())[..., None]
        g0 = curve.gather(1, i0[..., None].expand(-1, -1, 2))
        g1 = curve.gather(1, (i0 + 1)[..., None].expand(-1, -1, 2))
        return g0 * (1 - frac) + g1 * frac

    def decode(self, proto, z, logdur):
        n = proto.size(1)
        inp = torch.cat([
            proto, z[:, None, :].expand(-1, n, -1),
            logdur[:, None, None].expand(-1, n, 1)], dim=-1)
        h, _ = self.decoder(inp)
        curve = proto + self.to_offset(h)
        inc = torch.softmax(self.to_warp(h).squeeze(-1)[:, 1:], dim=-1)
        w = torch.cat([torch.zeros_like(inc[:, :1]), inc.cumsum(-1)], dim=-1)
        return self._sample_curve(curve, w), w

    def forward(self, gesture, proto, logdur):
        mu, logvar = self.encode(gesture, proto, logdur)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        pred, _ = self.decode(proto, z, logdur)
        return pred, None, self.predict_logdur(proto, z), mu, logvar

    @torch.no_grad()
    def rollout(self, proto, z, temperature: float = 1.0):
        logdur = self.predict_logdur(proto, z)
        pred, _ = self.decode(proto, z, logdur)
        return pred, logdur


def gen_loss(model, gesture, proto, logdur,
             beta: float, free_bits: float = 0.05):
    pred, prev, logdur_hat, mu, logvar = model(gesture, proto, logdur)
    if model.K:
        # `pred` carries decoder hidden states; the target is the true
        # displacement from the (teacher-forced) previous point.
        rec = model.mdn_nll(pred, gesture - prev)
    else:
        rec = (pred - gesture).abs().mean()
    rec_dur = (logdur_hat - logdur).abs().mean()
    kld_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean(dim=0)
    kld = kld_dim.clamp_min(free_bits).sum()
    loss = rec + 0.1 * rec_dur + beta * kld
    return loss, {"rec": float(rec.detach()), "dur": float(rec_dur.detach()),
                  "kld": float(kld_dim.sum().detach())}


class SmoothWarpVAE(WarpGestureVAE):
    """v4: v3 with the offset field restricted to a smooth basis.

    v3 fixed the geometry but buys its path length with abrupt direction
    changes where real gestures curve — its offset is free at all 64 knots,
    so nothing couples neighbours and per-knot noise reads as zig-zag. v4
    keeps everything else and synthesizes the offsets from the first
    ``n_basis`` half-cosines instead, so a wiggle shorter than the basis
    period is not representable at all. Smoothness becomes a property of
    the parameterization rather than something the loss must buy — the same
    reason the analytic min-jerk curve looks smooth.

    The remaining learned quantities are then exactly what min-jerk
    hand-fits: how far the path bows away from the straight line (the
    coefficients, i.e. aiming and overshoot) and how time is distributed
    along it (the warp). v4 is that analytic generator with both fitted
    parts replaced by a conditional model.
    """

    def __init__(self, cfg: GenConfig):
        super().__init__(cfg)
        k = cfg.n_basis
        self.to_offset = nn.Linear(cfg.d_model, 2 * (k + 2))
        s = torch.linspace(0, math.pi, cfg.n_points)
        u = torch.linspace(0, 1, cfg.n_points)
        # Half-sines (vanishing at both ends) plus a linear ramp, so the
        # bow of the path is smooth while the two endpoints stay free —
        # real gestures do not land exactly on the last key centre
        # (end-err 0.071), and a generator that always did would teach the
        # decoder a regularity real data never shows.
        self.register_buffer("basis", torch.cat([
            torch.stack([torch.sin((j + 1) * s) for j in range(k)], 1),
            torch.stack([1 - u, u], 1)], dim=1))

    def decode(self, proto, z, logdur):
        n = proto.size(1)
        inp = torch.cat([
            proto, z[:, None, :].expand(-1, n, -1),
            logdur[:, None, None].expand(-1, n, 1)], dim=-1)
        h, _ = self.decoder(inp)
        coef = self.to_offset(h.mean(1)).view(-1, self.cfg.n_basis + 2, 2)
        curve = proto + torch.einsum("nk,bkc->bnc", self.basis, coef)
        inc = torch.softmax(self.to_warp(h).squeeze(-1)[:, 1:], dim=-1)
        w = torch.cat([torch.zeros_like(inc[:, :1]), inc.cumsum(-1)], dim=-1)
        return self._sample_curve(curve, w), w


def build(cfg: GenConfig, arch: str = "ar"):
    """``ar`` = v1/v2 free-running; ``warp`` = v3; ``smooth`` = v4."""
    if arch == "warp":
        return WarpGestureVAE(cfg)
    if arch == "smooth":
        return SmoothWarpVAE(cfg)
    return GestureVAE(cfg)


@torch.no_grad()
def sample_swipes(model, words: list[str],
                  layout: KeyboardLayout, device: torch.device,
                  aspect: float, temperature: float = 1.0,
                  step_temperature: float = 1.0,
                  batch_size: int = 512, seed: int = 0) -> list[Swipe]:
    """``temperature`` scales the latent; ``step_temperature`` the MDN draw."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = model.cfg.n_points
    out: list[Swipe] = []
    for s in range(0, len(words), batch_size):
        chunk = words[s:s + batch_size]
        proto = torch.from_numpy(
            np.stack([prototype(w, layout, n) for w in chunk])).to(device)
        z = (torch.randn(len(chunk), model.cfg.z_dim, generator=g)
             * temperature).to(device)
        pts, logdur = model.rollout(proto, z, temperature=step_temperature)
        pts = pts.cpu().numpy()
        dur_ms = np.exp(logdur.cpu().numpy().astype(np.float64)) * 1000.0
        for i, w in enumerate(chunk):
            t = np.linspace(0.0, float(np.clip(dur_ms[i], 80.0, 20_000.0)), n)
            out.append(Swipe(
                word=w, x=pts[i, :, 0], y=pts[i, :, 1],
                t=np.round(t).astype(np.int32), aspect=aspect,
                session="gesturegen", source="gesturegen",
            ))
    return out
