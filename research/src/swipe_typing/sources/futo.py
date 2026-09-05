"""Loader for ``futo-org/swipe.futo.org`` (MIT).

The largest MIT-licensed swipe corpus: 1,043,789 filtered swipes over 108,759
unique target words from 12k+ donor sessions, split by donor session so
validation actually measures generalization to unseen users.

Coordinates need no rescaling. FUTO logs ``x``/``y`` already normalized to the
canvas, and the canvas *is* the 3-row letter grid -- confirmed by their own
``layouts/qwerty.json``, which is expressed in exactly this space.

Configs:
    swipe-1  1.04M  English QWERTY, split train/dev/test by session
    swipe-2  28k    additional donations
    swipe-3  38k
    swipe-4  50k
    swipe-5  59k    multi-layout + multi-language (10 layouts, 8 languages);
                    carries extra ``layout`` / ``language`` / ``dual_finger``
                    fields, exposed via ``Swipe.meta``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import Swipe, is_plausible

REPO_ID = "futo-org/swipe.futo.org"

#: config -> {split: path within the HF repo}
FILES: dict[str, dict[str, str]] = {
    "swipe-1": {
        "train": "train.jsonl",
        "validation": "dev.jsonl",
        "test": "test.jsonl",
    },
    "swipe-2": {"train": "swipe-2/swipe2.jsonl"},
    "swipe-3": {"train": "swipe-3/swipe3.jsonl"},
    "swipe-4": {"train": "swipe-4/swipe4.jsonl"},
    "swipe-5": {"train": "swipe-5/swipe5.jsonl"},
}

LAYOUT_FILES = "swipe-5/layouts/{name}.json"

_EXTRA_FIELDS = ("language", "layout", "dual_finger", "distance", "orientation")


def download(config: str = "swipe-1", split: str = "train",
             cache_dir: str | None = None) -> Path:
    """Fetch one split from the Hub and return its local path."""
    from huggingface_hub import hf_hub_download

    try:
        filename = FILES[config][split]
    except KeyError:
        avail = FILES.get(config, {})
        raise KeyError(
            f"no split {split!r} for config {config!r}; "
            f"available: {sorted(avail) or sorted(FILES)}"
        ) from None
    return Path(
        hf_hub_download(REPO_ID, filename, repo_type="dataset", cache_dir=cache_dir)
    )


def download_layout(name: str = "qwerty", cache_dir: str | None = None) -> Path:
    """Fetch one of the swipe-5 layout descriptions."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            REPO_ID, LAYOUT_FILES.format(name=name),
            repo_type="dataset", cache_dir=cache_dir,
        )
    )


def normalize_word(raw: str) -> str:
    """Lowercase and drop anything not typeable on the letter grid.

    Targets are drawn from natural sentences, so they carry case and
    punctuation ("The", "don't"). The keyboard has no apostrophe key, so the
    gesture for "don't" is the gesture for "dont" -- strip rather than reject.
    """
    return "".join(ch for ch in raw.lower() if ch.isalpha())


def parse_record(rec: dict, split: str, source: str = "futo") -> Swipe | None:
    """Convert one JSONL record into a canonical :class:`Swipe`.

    Returns ``None`` for anything that is not a single continuous trace. In
    swipe-5, 2,708 records are dual-finger gestures whose ``data`` is
    ``{"L": [[...]], "R": [[...]]}`` rather than a point list -- a different
    input modality, not a malformed record, and out of scope for a
    single-trajectory encoder.
    """
    pts = rec.get("data") or []
    if not isinstance(pts, list) or not pts:
        return None
    head = pts[0]
    if not (isinstance(head, dict) and {"x", "y", "t"} <= head.keys()):
        return None
    word = normalize_word(rec.get("word", ""))
    if not word:
        return None

    x = np.fromiter((p["x"] for p in pts), dtype=np.float32, count=len(pts))
    y = np.fromiter((p["y"] for p in pts), dtype=np.float32, count=len(pts))
    t = np.fromiter((p["t"] for p in pts), dtype=np.int64, count=len(pts))
    t = (t - t[0]).astype(np.int32)

    h = float(rec.get("canvas_height") or 0.0)
    w = float(rec.get("canvas_width") or 0.0)
    aspect = (w / h) if h > 0 else 0.0

    meta = {k: rec[k] for k in _EXTRA_FIELDS if k in rec}
    return Swipe(
        word=word,
        x=x,
        y=y,
        t=t,
        aspect=aspect,
        session=rec.get("session", ""),
        source=source,
        split=split,
        sentence=rec.get("sentence") or "",
        word_idx=int(rec.get("word_idx", -1)),
        flagged=bool(rec.get("potentially_invalid_sentence", False)),
        meta=meta,
    )


def iter_swipes(path: str | Path, split: str = "train", config: str = "swipe-1",
                keep_flagged: bool = False,
                validate: bool = True) -> Iterator[Swipe]:
    """Stream canonical swipes from a downloaded FUTO JSONL file."""
    source = "futo" if config == "swipe-1" else f"futo/{config}"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sw = parse_record(rec, split=split, source=source)
            if sw is None:
                continue
            if sw.flagged and not keep_flagged:
                continue
            if validate and not is_plausible(sw):
                continue
            yield sw


def load(config: str = "swipe-1", split: str = "train", cache_dir: str | None = None,
         **kwargs) -> Iterator[Swipe]:
    """Download (if needed) and stream a FUTO split."""
    return iter_swipes(
        download(config, split, cache_dir), split=split, config=config, **kwargs
    )
