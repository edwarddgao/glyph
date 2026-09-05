"""Loader for "How We Swipe" (Leiva et al., MobileHCI '21) -- https://osf.io/sj67f/

1,338 users, ~8.8M touch points, 11,318 unique English words. About 10x smaller
than FUTO and collected on different apparatus, which is exactly what makes it a
good *held-out* evaluation set: different devices, different users, different
keyboard geometry, so it measures transfer rather than memorization of one
capture pipeline.

Two format details this parser handles, both verified against the release:

1.  Rows are one touch *event*, not one swipe. Gestures must be segmented on
    touchstart -> touchend. Users retry failed words, so a single target word
    can produce several gestures.

2.  Some lines carry extra trailing fields beyond the 12-column header (we see
    up to 31), which are additional per-attempt error flags. Every field this
    loader needs sits at a fixed index 0..10, so we parse positionally and treat
    everything from index 11 on as flags.

Vertical geometry differs from FUTO: ``keyb_height`` here also covers a partial
bottom (space) row, so the letter grid occupies only the top ~82.6%. That factor
comes from ``scripts/calibrate_layout.py``; see ``_GRID_SPAN``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Swipe, is_plausible

#: Fraction of ``keyb_height`` occupied by the 3 letter rows.
#:
#: Calibration measures a row pitch of 0.2752 of keyb_height from the *spacing*
#: of per-row touch medians -- a difference of medians, so the users' systematic
#: downward touch offset cancels out. 3 x 0.2752 = 0.8256. The remaining 17%
#: below is the partial space row.
_GRID_SPAN = 0.8256

#: Top edge of the letter grid. Calibration returns -0.012 after de-biasing
#: against FUTO's measured touch offset; that is within noise of the keyboard's
#: own top edge and a negative value is not physically meaningful, so we clamp.
#: The check that this is right: after mapping, How We Swipe's row centers land
#: at 0.181 / 0.524 / 0.847 against FUTO's 0.191 / 0.531 / 0.864 -- under 5% of
#: a row apart, from two independent corpora.
_GRID_Y0 = 0.0

# Column positions in the .log files (header has 12 names; trailing extras are
# per-attempt error flags).
_SENTENCE, _TIME, _KW, _KH, _EVENT = 0, 1, 2, 3, 4
_X, _Y, _RX, _RY, _ANGLE, _WORD = 5, 6, 7, 8, 9, 10
_N_FIXED = 11


def normalize_word(raw: str) -> str:
    return "".join(ch for ch in raw.lower() if ch.isalpha())


def _flush(buf: list, word: str, sentence: str, kw: float, kh: float,
           uid: str, split: str, flagged: bool) -> Swipe | None:
    if len(buf) < 2:
        return None
    t = np.array([b[0] for b in buf], dtype=np.int64)
    x = np.array([b[1] for b in buf], dtype=np.float32) / kw
    y = np.array([b[2] for b in buf], dtype=np.float32) / kh
    y = (y - _GRID_Y0) / _GRID_SPAN
    return Swipe(
        word=word,
        x=x,
        y=y,
        t=(t - t[0]).astype(np.int32),
        aspect=(kw / (kh * _GRID_SPAN)) if kh > 0 else 0.0,
        session=uid,
        source="how_we_swipe",
        split=split,
        sentence=sentence.replace("_", " "),
        flagged=flagged,
    )


def iter_logfile(path: str | Path, split: str = "test",
                 keep_flagged: bool = False,
                 validate: bool = True) -> Iterator[Swipe]:
    """Stream canonical swipes from one ``swipelogs/<uid>.log`` file."""
    path = Path(path)
    uid = path.stem
    buf: list[tuple[int, float, float]] = []
    open_gesture = False
    word = sentence = ""
    kw = kh = 0.0
    flagged = False

    with open(path, errors="replace") as fh:
        fh.readline()  # header
        for line in fh:
            f = line.rstrip("\n").split(" ")
            if len(f) < _N_FIXED + 1:
                continue
            event = f[_EVENT]
            if event not in ("touchstart", "touchmove", "touchend"):
                continue
            try:
                ts = int(f[_TIME])
                px, py = float(f[_X]), float(f[_Y])
                w, h = float(f[_KW]), float(f[_KH])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue

            if event == "touchstart":
                # An unterminated previous gesture is abandoned, not merged.
                buf = [(ts, px, py)]
                open_gesture = True
                word = normalize_word(f[_WORD])
                sentence = f[_SENTENCE]
                kw, kh = w, h
                flagged = any(v == "1" for v in f[_N_FIXED:])
                continue

            if not open_gesture:
                continue
            buf.append((ts, px, py))
            flagged = flagged or any(v == "1" for v in f[_N_FIXED:])

            if event == "touchend":
                open_gesture = False
                if word and (keep_flagged or not flagged):
                    sw = _flush(buf, word, sentence, kw, kh, uid, split, flagged)
                    if sw is not None and (not validate or is_plausible(sw)):
                        yield sw
                buf = []


def iter_swipes(logdir: str | Path, split: str = "test", limit: int | None = None,
                **kwargs) -> Iterator[Swipe]:
    """Stream canonical swipes from a ``swipelogs/`` directory."""
    files = sorted(Path(logdir).glob("*.log"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        yield from iter_logfile(path, split=split, **kwargs)


def load_participants(logdir: str | Path) -> dict[str, dict]:
    """Per-user demographics from the sidecar ``<uid>.json`` files."""
    out: dict[str, dict] = {}
    for path in sorted(Path(logdir).glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return out
