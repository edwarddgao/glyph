"""Trajectory -> fixed-size feature tensor.

Follows the representation described in the FUTO Swipe paper (arXiv 2606.25247):
resample to a uniform rate, interpolate to exactly 64 points, then derive an
8-dim per-point feature vector with a Savitzky-Golay filter.

The 8 channels are position (x, y), velocity (vx, vy), acceleration (ax, ay),
speed, and signed curvature.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .schema import Swipe

N_POINTS = 64
RESAMPLE_HZ = 60.0
N_FEATURES = 8

FEATURE_NAMES = ("x", "y", "vx", "vy", "ax", "ay", "speed", "curvature")

_EPS = 1e-6

#: Curvature is 1/radius, so it diverges wherever the gesture pauses and speed
#: approaches zero -- which happens constantly, at every key corner. Left
#: unbounded it reaches ~1e16 on real data and destroys any feature
#: normalization downstream. A radius of 1e-3 grid-heights is already a hundred
#: times tighter than a key, so anything past that carries no information.
CURVATURE_CLIP = 1e3



def aspect_correct(points: np.ndarray, aspect: float) -> np.ndarray:
    """Scale canonical coords into an isotropic space.

    Canonical x/y each span [0, 1] regardless of the device's real proportions,
    so raw canonical coordinates distort angles and curvature by a per-device
    factor. Multiplying x by ``aspect`` restores physical proportions with the
    grid height as the unit. Sources with an unknown aspect pass through.
    """
    if not np.isfinite(aspect) or aspect <= 0:
        return np.asarray(points, dtype=np.float32)
    out = np.array(points, dtype=np.float32, copy=True)
    out[:, 0] *= aspect
    return out


def _dedupe_time(t: np.ndarray) -> np.ndarray:
    """Make timestamps strictly increasing so interpolation is well posed.

    Touch logs commonly repeat a millisecond across consecutive samples.
    """
    t = np.asarray(t, dtype=np.float64).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-3
    return t


def resample(points: np.ndarray, t: np.ndarray, n: int = N_POINTS,
             mode: str = "time") -> np.ndarray:
    """Interpolate a trajectory to exactly ``n`` points.

    Args:
        mode: ``"time"`` spaces points uniformly in time (what the FUTO paper
            does, and it preserves pause/hesitation cues at key corners);
            ``"arclength"`` spaces them uniformly along the path, which
            normalizes away speed and is the classic SHARK2-style choice.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points.astype(np.float32), n, axis=0)

    if mode == "time":
        u = _dedupe_time(t)
    elif mode == "arclength":
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(step)])
        if u[-1] <= 0:
            return np.repeat(points[:1].astype(np.float32), n, axis=0)
        u = _dedupe_time(u)
    else:
        raise ValueError(f"unknown resample mode {mode!r}")

    grid = np.linspace(u[0], u[-1], n)
    return np.stack(
        [np.interp(grid, u, points[:, 0]), np.interp(grid, u, points[:, 1])],
        axis=1,
    ).astype(np.float32)


def _smooth_derivatives(xy: np.ndarray, dt: float, window: int = 7, poly: int = 2):
    """Position, velocity, acceleration via Savitzky-Golay."""
    n = len(xy)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2:
        # Too short to filter; fall back to plain finite differences.
        pos = xy
        vel = np.gradient(xy, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        return pos, vel, acc
    pos = savgol_filter(xy, window, poly, deriv=0, axis=0)
    vel = savgol_filter(xy, window, poly, deriv=1, delta=dt, axis=0)
    acc = savgol_filter(xy, window, poly, deriv=2, delta=dt, axis=0)
    return pos, vel, acc


def encode(sw: Swipe, n: int = N_POINTS, mode: str = "time",
           correct_aspect: bool = True) -> np.ndarray:
    """Encode one swipe as an ``(n, 8)`` float32 feature tensor."""
    pts = sw.points
    if correct_aspect:
        pts = aspect_correct(pts, sw.aspect)

    xy = resample(pts, sw.t, n=n, mode=mode).astype(np.float64)

    duration_s = max(sw.duration_ms, 1) / 1000.0
    dt = duration_s / max(n - 1, 1) if mode == "time" else 1.0 / max(n - 1, 1)

    pos, vel, acc = _smooth_derivatives(xy, dt)
    speed = np.linalg.norm(vel, axis=1)
    # Signed curvature of a plane curve: (x'y" - y'x") / |v|^3, guarded at both
    # ends -- epsilon in the denominator, then clipped. See CURVATURE_CLIP.
    cross = vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]
    curvature = np.clip(
        cross / (np.power(speed, 3) + _EPS), -CURVATURE_CLIP, CURVATURE_CLIP
    )

    feats = np.column_stack(
        [pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1],
         acc[:, 0], acc[:, 1], speed, curvature]
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


#: Channels of :func:`kinematics` -- ``encode`` minus absolute position.
KINEMATIC_NAMES = ("vx", "vy", "ax", "ay", "speed", "curvature")
N_KINEMATIC = len(KINEMATIC_NAMES)


def key_scale(radii: np.ndarray) -> np.ndarray:
    """Median key (width, height) from a layout's half-extents."""
    r = np.asarray(radii, dtype=np.float32).reshape(-1, 2)
    return np.maximum(2.0 * np.median(r, axis=0), 1e-6).astype(np.float32)


def to_key_units(points: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Rescale canonical coordinates so one unit is one key.

    Aspect correction expresses position in grid-heights, which is only
    comparable between layouts that have the same number of rows. A 5-row
    keyboard packs its rows into the same unit square as a 3-row one, so an
    identical gesture measures ~1.65x slower in grid-heights per second -- a
    distribution shift precisely where cross-layout transfer is being asked for.

    Dividing by key size removes it, and it is the same normalization
    :func:`key_affinity` already applies, so both feature groups end up on one
    scale. The device aspect ratio cancels out: physical x over physical key
    width is ``x * aspect / (2 * rx * aspect)``.
    """
    return (np.asarray(points, dtype=np.float32).reshape(-1, 2)
            / np.asarray(scale, dtype=np.float32).reshape(1, 2))


def kinematics(points: np.ndarray, t: np.ndarray, aspect: float,
               n: int = N_POINTS, mode: str = "time",
               scale: np.ndarray | None = None) -> np.ndarray:
    """Motion-only features: ``(n, 6)`` with no absolute position.

    Position is deliberately excluded. An encoder that sees raw (x, y) can
    memorize one keyboard; feeding it motion plus a layout-relative affinity
    (see :func:`key_affinity`) leaves it nothing layout-specific to memorize.

    Args:
        scale: key (width, height) from :func:`key_scale`. When given, motion is
            measured in keys-per-second, which is comparable across layouts with
            different row counts -- the right choice for a layout-agnostic
            model. When ``None``, falls back to aspect-corrected grid units.
    """
    if scale is not None:
        pts = to_key_units(points, scale)
    else:
        pts = aspect_correct(points, aspect)
    xy = resample(pts, t, n=n, mode=mode).astype(np.float64)
    duration_s = max(int(t[-1]) - int(t[0]), 1) / 1000.0 if len(t) else 1.0
    dt = duration_s / max(n - 1, 1) if mode == "time" else 1.0 / max(n - 1, 1)

    _, vel, acc = _smooth_derivatives(xy, dt)
    speed = np.linalg.norm(vel, axis=1)
    cross = vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]
    curvature = np.clip(
        cross / (np.power(speed, 3) + _EPS), -CURVATURE_CLIP, CURVATURE_CLIP
    )
    out = np.column_stack(
        [vel[:, 0], vel[:, 1], acc[:, 0], acc[:, 1], speed, curvature]
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def key_affinity(points: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                 n: int = N_POINTS, t: np.ndarray | None = None,
                 mode: str = "time", temperature: float = 1.0) -> np.ndarray:
    """Per-frame Gaussian affinity to every key: ``(n, K)``.

    This is the *only* channel through which the keyboard reaches the model, and
    it is expressed in key-relative units -- offsets are divided by each key's
    own half-extents. That makes it invariant to key size and key aspect, so a
    layout the model has never seen (azerty, dvorak, a different phone) produces
    affinities on the same scale as the training layout.

    No aspect correction is applied here, and none is wanted: dividing by
    ``radii`` already puts both axes in units of that key's own geometry.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if t is not None:
        pts = resample(pts, t, n=n, mode=mode)
    centers = np.asarray(centers, dtype=np.float32).reshape(1, -1, 2)
    radii = np.maximum(np.asarray(radii, dtype=np.float32).reshape(1, -1, 2), 1e-6)

    delta = (pts[:, None, :] - centers) / radii
    return np.exp(-0.5 * np.square(delta).sum(axis=2) / max(temperature, 1e-6)).astype(
        np.float32
    )


def encode_batch(swipes, n: int = N_POINTS, **kwargs) -> np.ndarray:
    """Encode many swipes into a ``(B, n, 8)`` tensor."""
    swipes = list(swipes)
    out = np.zeros((len(swipes), n, N_FEATURES), dtype=np.float32)
    for i, sw in enumerate(swipes):
        out[i] = encode(sw, n=n, **kwargs)
    return out
