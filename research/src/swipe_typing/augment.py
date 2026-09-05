"""Co-augmentation of trajectory and layout.

The single most important idea in the FUTO Swipe paper: every geometric
transform is applied *identically* to the swipe trajectory and to the key-center
tensor the model is conditioned on -- the same way an image augmentation must
also move the bounding boxes. Augmenting only the trajectory teaches the model
that a distorted gesture maps to the same word on a fixed layout, which is
false. Moving both teaches it to read gesture shape relative to whatever layout
it is handed, which is what makes the encoder layout-agnostic.

Use ``augment`` for one (trajectory, layout) pair; both come back transformed by
the same affine matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .layout import KeyboardLayout
from .schema import Swipe


@dataclass(slots=True)
class AugmentConfig:
    """Ranges are sampled uniformly; scales are multiplicative."""

    x_scale: tuple[float, float] = (0.85, 1.15)
    y_scale: tuple[float, float] = (0.85, 1.15)
    shear: tuple[float, float] = (-0.10, 0.10)
    rotation_deg: tuple[float, float] = (-8.0, 8.0)
    translate: tuple[float, float] = (-0.05, 0.05)
    flip_x: float = 0.0
    flip_y: float = 0.0
    time_reverse: float = 0.0
    jitter: float = 0.0

    def scaled(self, strength: float) -> "AugmentConfig":
        """Interpolate every range toward identity by ``strength`` in [0, 1]."""
        def s(rng):
            lo, hi = rng
            return (lo * strength, hi * strength)

        return replace(
            self,
            x_scale=(1 + (self.x_scale[0] - 1) * strength,
                     1 + (self.x_scale[1] - 1) * strength),
            y_scale=(1 + (self.y_scale[0] - 1) * strength,
                     1 + (self.y_scale[1] - 1) * strength),
            shear=s(self.shear),
            rotation_deg=s(self.rotation_deg),
            translate=s(self.translate),
            jitter=self.jitter * strength,
        )


DEFAULT = AugmentConfig()


def sample_affine(cfg: AugmentConfig, rng: np.random.Generator,
                  center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Sample a 3x3 homogeneous affine matrix, applied about ``center``."""
    sx = rng.uniform(*cfg.x_scale)
    sy = rng.uniform(*cfg.y_scale)
    if cfg.flip_x and rng.random() < cfg.flip_x:
        sx = -sx
    if cfg.flip_y and rng.random() < cfg.flip_y:
        sy = -sy
    sh = rng.uniform(*cfg.shear)
    th = np.deg2rad(rng.uniform(*cfg.rotation_deg))
    tx = rng.uniform(*cfg.translate)
    ty = rng.uniform(*cfg.translate)

    scale_shear = np.array([[sx, sh, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    rot = np.array([[np.cos(th), -np.sin(th), 0.0],
                    [np.sin(th), np.cos(th), 0.0],
                    [0.0, 0.0, 1.0]])
    cx, cy = center
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    back = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]])
    return back @ rot @ scale_shear @ to_origin


def apply_affine(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous matrix to an (N, 2) array."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (hom @ mat.T)[:, :2].astype(np.float32)


def augment(sw: Swipe, kb: KeyboardLayout, cfg: AugmentConfig = DEFAULT,
            rng: np.random.Generator | None = None) -> tuple[Swipe, KeyboardLayout]:
    """Transform a swipe and its layout by one shared affine.

    Time reversal also reverses the target word: a gesture swiped backwards is
    the gesture for the reversed string, so the label must follow or the model
    is trained on a lie.
    """
    rng = rng or np.random.default_rng()
    mat = sample_affine(cfg, rng)

    pts = apply_affine(sw.points, mat)
    if cfg.jitter:
        pts = pts + rng.normal(0.0, cfg.jitter, size=pts.shape).astype(np.float32)

    centers = apply_affine(kb.centers, mat)
    # Half-extents transform by the linear part only (no translation), and
    # stay positive.
    radii = np.abs(kb.radii @ np.abs(mat[:2, :2]).T).astype(np.float32)

    word, t = sw.word, sw.t
    if cfg.time_reverse and rng.random() < cfg.time_reverse:
        pts = pts[::-1].copy()
        span = int(t[-1]) if len(t) else 0
        t = (span - t[::-1]).astype(np.int32)
        word = word[::-1]

    new_sw = Swipe(
        word=word,
        x=pts[:, 0],
        y=pts[:, 1],
        t=t,
        aspect=sw.aspect,
        session=sw.session,
        source=sw.source,
        split=sw.split,
        sentence=sw.sentence,
        word_idx=sw.word_idx,
        flagged=sw.flagged,
        meta=dict(sw.meta),
    )
    new_kb = KeyboardLayout(kb.name, kb.letters, centers, radii)
    return new_sw, new_kb
