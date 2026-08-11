"""Minimum-jerk gesture generator (Quinn & Zhai 2018, as re-implemented in
WordGesture-GAN's baseline).

The generative story: a swipe passes through via points -- the word's key
centers, perturbed by aiming noise -- along the trajectory that minimizes
integrated squared jerk *globally* (Flash & Hogan's via-point formulation):
at rest only at the two ends, position pinned at each interior via, and
derivatives continuous through them. The optimum is a piecewise quintic
whose velocity at a corner is generally non-collinear with either adjacent
segment, so the path rounds corners and glides through letters rather than
parking at each one. Via passage times follow the CLC power law T = m * L^n,
so the only fitted parameters are (m, n), a lognormal duration residual, and
the per-axis aiming noise, all estimated from real gestures by
closed-form/grid least squares. No training loop.

Everything is emitted in canonical coordinates as :class:`Swipe` records, so
synthetic gestures flow through the exact pipeline real ones do. Sampling is
uniform in time at RESAMPLE_HZ: the quintic's bell-shaped speed makes samples
bunch at via points, which is the dwell signal the decoders feed on.

Known signature (paper, Table 6): min-jerk gestures are too smooth -- about
half the jerk of user-drawn ones. That is the realism gap a learned
generator would have to beat.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import features
from .layout import KeyboardLayout
from .schema import Swipe

#: Rest-to-rest quintic: p(tau) = p0 + (p1 - p0) * s(tau). The "segments"
#: profile: each segment starts and ends at rest, so the path is a straight
#: line per segment and the gesture parks at every via point.
def _minjerk_s(tau: np.ndarray) -> np.ndarray:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


#: d^r/dtau^r of tau^i evaluated at tau=1: i!/(i-r)!.
_D_AT_1 = np.array(
    [[np.prod(range(i - r + 1, i + 1)) if i >= r else 0.0 for i in range(6)]
     for r in range(5)]
)


def _minjerk_spline(vias: np.ndarray, seg_s: np.ndarray) -> np.ndarray:
    """(K, 6, 2) quintic coefficients of the global minimum-jerk trajectory.

    Minimizing integrated squared jerk with position pinned at every via,
    rest (v = a = 0) at the two ends, and free interior derivatives gives a
    piecewise quintic with derivatives 1..4 continuous at interior vias --
    6K coefficients against exactly 6K constraints, one dense solve. Each
    piece is parameterized on local tau in [0, 1] for conditioning; the
    chain-rule factor 1/T^r carries the segment durations into the
    continuity rows.
    """
    K = len(seg_s)
    A = np.zeros((6 * K, 6 * K))
    b = np.zeros((6 * K, 2))
    row = 0
    for j in range(K):
        A[row, 6 * j] = 1.0
        b[row] = vias[j]
        row += 1
        A[row, 6 * j:6 * j + 6] = 1.0
        b[row] = vias[j + 1]
        row += 1
    for j in range(K - 1):
        for r in range(1, 5):
            A[row, 6 * j:6 * j + 6] = _D_AT_1[r] / seg_s[j] ** r
            A[row, 6 * (j + 1) + r] = -np.prod(range(1, r + 1)) \
                / seg_s[j + 1] ** r
            row += 1
    A[row, 1] = 1.0
    A[row + 1, 2] = 1.0
    A[row + 2, 6 * (K - 1):] = _D_AT_1[1] / seg_s[-1]
    A[row + 3, 6 * (K - 1):] = _D_AT_1[2] / seg_s[-1] ** 2
    return np.linalg.solve(A, b).reshape(K, 6, 2)


@dataclass
class MinJerkModel:
    """Fitted parameters of the analytic generator."""

    m: float
    n: float
    log_sigma: float          # std of log(T_real / T_pred)
    offset_sigma_x: float     # aiming noise, in key half-widths
    offset_sigma_y: float
    aspect: float             # median device aspect of the fit corpus
    min_segment_ms: float = 80.0
    hz: float = features.RESAMPLE_HZ
    #: "spline" = global via-point minimum jerk (rounds corners, glides
    #: through letters); "segments" = rest-to-rest per segment (straight
    #: lines, full stop at every via). Kept for the ablation.
    profile: str = "spline"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MinJerkModel":
        return cls(**json.loads(Path(path).read_text()))


def _segment_lengths(word: str, layout: KeyboardLayout,
                     aspect: float) -> np.ndarray:
    pts = np.stack([layout.center(c) for c in word]).astype(np.float64)
    pts[:, 0] *= aspect
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def fit(corpus, layout: KeyboardLayout, max_swipes: int = 100_000,
        rng: np.random.Generator | None = None) -> MinJerkModel:
    """Estimate (m, n), duration spread, and aiming noise from real swipes.

    The duration law is linear in m given n, so n is a 1-D grid search with
    closed-form least squares for m at each step. Aiming noise comes from the
    gesture endpoints only: first and last touch are the two via points whose
    correspondence is unambiguous.
    """
    rng = rng or np.random.default_rng(0)
    idx = np.arange(len(corpus))
    if len(idx) > max_swipes:
        idx = rng.choice(idx, size=max_swipes, replace=False)

    sums: dict[float, float] = {}
    durations, feats_cache = [], []
    end_off = []
    kw = features.key_scale(layout.radii)
    grid = np.round(np.arange(0.05, 1.01, 0.05), 2)
    for i in idx:
        word = corpus.words[i]
        if len(word) < 2:
            continue
        aspect = float(corpus.aspects[i])
        if not np.isfinite(aspect) or aspect <= 0:
            aspect = 1.0
        lengths = _segment_lengths(word, layout, aspect)
        t = corpus.times(i)
        dur = float(t[-1] - t[0])
        if dur <= 0:
            continue
        durations.append(dur)
        feats_cache.append([float((lengths**n).sum()) for n in grid])
        pts = corpus.points(i)
        first, last = layout.center(word[0]), layout.center(word[-1])
        end_off.append((pts[0] - first) / kw)
        end_off.append((pts[-1] - last) / kw)

    durations = np.asarray(durations)
    feats_arr = np.asarray(feats_cache)
    best = None
    for j, n in enumerate(grid):
        s = feats_arr[:, j]
        m = float((s * durations).sum() / (s * s).sum())
        rmse = float(np.sqrt(((m * s - durations) ** 2).mean()))
        if best is None or rmse < best[0]:
            pred = np.maximum(m * s, 1.0)
            log_sigma = float(np.std(np.log(durations / pred)))
            best = (rmse, m, float(n), log_sigma)

    end_off = np.asarray(end_off)
    return MinJerkModel(
        m=best[1], n=best[2], log_sigma=best[3],
        offset_sigma_x=float(end_off[:, 0].std()),
        offset_sigma_y=float(end_off[:, 1].std()),
        aspect=float(np.median(corpus.aspects)),
    )


def _via_points(word: str, layout: KeyboardLayout, model: MinJerkModel,
                rng: np.random.Generator) -> np.ndarray:
    vias = np.stack([layout.center(c) for c in word]).astype(np.float64)
    kw = features.key_scale(layout.radii)
    noise = rng.normal(0.0, 1.0, vias.shape)
    noise[:, 0] *= model.offset_sigma_x * kw[0]
    noise[:, 1] *= model.offset_sigma_y * kw[1]
    return vias + noise


def generate(model: MinJerkModel, word: str, layout: KeyboardLayout,
             rng: np.random.Generator) -> Swipe:
    """One synthetic swipe for ``word`` in canonical coordinates."""
    vias = _via_points(word, layout, model, rng)
    seg_vec = np.diff(vias, axis=0)
    seg_len = np.linalg.norm(seg_vec * [model.aspect, 1.0], axis=1)

    seg_ms = np.maximum(model.m * seg_len**model.n, model.min_segment_ms)
    seg_ms *= np.exp(rng.normal(0.0, model.log_sigma))
    total = float(seg_ms.sum()) if len(seg_ms) else model.min_segment_ms
    total = max(total, 4_000.0 / model.hz)  # >= 4 samples for is_plausible
    bounds = np.concatenate([[0.0], np.cumsum(seg_ms)]) if len(seg_ms) \
        else np.array([0.0, total])

    n_out = max(int(round(total * model.hz / 1000.0)) + 1, 4)
    t = np.linspace(0.0, bounds[-1] if len(seg_ms) else total, n_out)
    if len(seg_ms):
        seg_idx = np.clip(np.searchsorted(bounds, t, side="right") - 1,
                          0, len(seg_ms) - 1)
        tau = np.clip((t - bounds[seg_idx]) / seg_ms[seg_idx], 0.0, 1.0)
        if model.profile == "segments":
            pts = vias[seg_idx] + seg_vec[seg_idx] * _minjerk_s(tau)[:, None]
        else:
            coeffs = _minjerk_spline(vias, seg_ms / 1000.0)
            pts = np.einsum("kic,ki->kc", coeffs[seg_idx],
                            tau[:, None] ** np.arange(6))
    else:
        pts = np.repeat(vias[:1], n_out, axis=0)

    return Swipe(
        word=word, x=pts[:, 0].astype(np.float32),
        y=pts[:, 1].astype(np.float32),
        t=np.round(t).astype(np.int32), aspect=model.aspect,
        session="minjerk", source="minjerk",
    )


def generate_many(model: MinJerkModel, words: list[str],
                  layout: KeyboardLayout, seed: int = 0) -> list[Swipe]:
    rng = np.random.default_rng(seed)
    return [generate(model, w, layout, rng) for w in words]
