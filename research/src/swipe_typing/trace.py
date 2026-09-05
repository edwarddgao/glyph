"""Training-free trace features: the key sequence under a swipe path.

The learned stack reads geometry through an encoder; the training-free stack
reads it through the layout directly. A *key trace* is the sequence of nearest
keys visited by the path — the intended word's letters appear in order among
them (start and end exactly, interior letters modulo sloppiness), and
everything between is transit over keys the finger merely crossed. That string
is what a language model can decode without ever seeing gesture data.

Distances are measured in key units (offset divided by the key's half-extents),
not raw canonical units — a canonical-space Euclidean nearest-key would trade
0.1-wide columns against 0.33-tall rows and misassign vertically.
"""

from __future__ import annotations

import numpy as np

from .features import resample
from .layout import KeyboardLayout
from .schema import Swipe

#: Arclength step between trace samples, in canonical units (a key column is
#: 0.1 wide, so this is one sample per fifth of a key width).
STEP = 0.02


def nearest_keys(points: np.ndarray, kb: KeyboardLayout) -> np.ndarray:
    """Index of the nearest key per point, in key-unit distance. (N,) int."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    # (N, K, 2) offsets scaled by per-key half-extents.
    d = (pts[:, None, :] - kb.centers[None, :, :]) / kb.radii[None, :, :]
    return np.argmin((d * d).sum(-1), axis=1)


def _n_samples(points: np.ndarray, step: float) -> int:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 2
    arclen = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return int(np.clip(round(arclen / step) + 1, 2, 512))


def key_trace(sw: Swipe, kb: KeyboardLayout | None = None,
              mode: str = "collapsed", step: float = STEP) -> str:
    """Nearest-key string for a swipe.

    Args:
        mode: ``"collapsed"`` resamples uniformly along the path and collapses
            consecutive repeats — pure geometry, one letter per key crossed.
            ``"dwell"`` resamples uniformly in *time* (one sample per 20ms)
            and keeps repeats, so lingering on a key shows as a repeated
            letter — the pause/hesitation cue the FUTO paper leans on,
            spelled in a form a text model can read.
    """
    kb = kb or KeyboardLayout.qwerty()
    pts = sw.points
    if mode == "collapsed":
        xy = resample(pts, sw.t, n=_n_samples(pts, step), mode="arclength")
        collapse = True
    elif mode == "dwell":
        n = int(np.clip(round(sw.duration_ms / 20) + 1, 2, 512))
        xy = resample(pts, sw.t, n=n, mode="time")
        collapse = False
    else:
        raise ValueError(f"unknown trace mode {mode!r}")
    idx = nearest_keys(xy, kb)
    if collapse:
        keep = np.concatenate([[True], idx[1:] != idx[:-1]])
        idx = idx[keep]
    return "".join(kb.letters[i] for i in idx)


def template_trace(word: str, kb: KeyboardLayout | None = None,
                   step: float = STEP) -> str:
    """Collapsed key trace of the straight-line template for ``word``.

    Built from geometry alone, so it is safe to use in prompts and few-shot
    examples without touching any gesture corpus.
    """
    from .layout import ideal_trace

    kb = kb or KeyboardLayout.qwerty()
    pts = ideal_trace(word, points_per_key=8, kb=kb)
    xy = resample(pts, np.arange(len(pts)), n=_n_samples(pts, step),
                  mode="arclength")
    idx = nearest_keys(xy, kb)
    keep = np.concatenate([[True], idx[1:] != idx[:-1]])
    return "".join(kb.letters[i] for i in idx[keep])


def collapse(word: str) -> str:
    """Collapse consecutive repeated letters (``hello`` -> ``helo``).

    A swipe visits a doubled letter once, so this is the form a collapsed
    trace can actually witness; doubling must come from the language model.
    """
    return "".join(a for a, b in zip(word, "\0" + word) if a != b)


def is_subsequence(word: str, trace: str) -> bool:
    """Whether ``word``'s letters appear in order within ``trace``.

    Compare ``collapse(word)`` against a collapsed trace for the information
    ceiling of any decoder that only sees the trace string: if the collapsed
    label is not a subsequence, the trace alone cannot produce it.
    """
    it = iter(trace)
    return all(ch in it for ch in word)
