#!/usr/bin/env python3
"""Write a synthetic min-jerk training corpus mirroring futo/train.

    python scripts/gen_minjerk_corpus.py --out data/canonical/minjerk/train

One synthetic gesture per real training swipe, for the same word -- so the
synthetic corpus has the identical word multiset and size, and any difference
a decoder trained on it shows is attributable to the gestures, not the label
distribution. This is the paper's Table 7 experiment (synthetic-only training
nearly matched real for SHARK2) run against a far stronger decoder.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from swipe_typing import cache, minjerk
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--model", default="runs/minjerk_model.json",
                    help="fitted MinJerkModel (from probe_minjerk.py)")
    ap.add_argument("--out", default="data/canonical/minjerk/train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", default=None,
                    choices=["spline", "segments", "random"],
                    help="override the model's trajectory profile")
    ap.add_argument("--dwell-prob", type=float, default=None)
    ap.add_argument("--tremor", type=float, default=None)
    ap.add_argument("--seg-jitter", type=float, default=None)
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    model = minjerk.MinJerkModel.load(args.model)
    for k in ("profile", "dwell_prob", "tremor", "seg_jitter"):
        v = getattr(args, k)
        if v is not None:
            setattr(model, k, v)
    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.limit)
    print(f"generating {len(train):,} synthetic swipes ...")
    t0 = time.time()
    swipes = minjerk.generate_many(model, list(train.words), kb,
                                   seed=args.seed)
    for sw in swipes:
        sw.split = "train"
    cache.write(swipes, args.out)
    print(f"wrote {len(swipes):,} to {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
